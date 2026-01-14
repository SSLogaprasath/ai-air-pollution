"""
Fine-tune the STGraphTransformer on recent OpenAQ data.

This script:
1. Fetches recent hourly data from OpenAQ (last 10 days)
2. Aggregates to daily for model input
3. Fine-tunes the pretrained model
4. Saves improved checkpoint
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar

from data_ingestion import fetch_city_graph, fetch_historical_signals, MASK_TOKEN
from train import PollutionForecaster, MaskedMSELoss


# ============================================================
# Configuration
# ============================================================

CITIES = ['mumbai', 'delhi', 'chennai', 'bangalore']
HOURS_TO_FETCH = 240  # 10 days (under API limit)
INPUT_WINDOW = 7      # Days for fine-tuning (OpenAQ has ~10 days)
CHECKPOINT_PATH = Path("checkpoints/pollution-forecaster-best.ckpt")
FINETUNE_CHECKPOINT_DIR = Path("checkpoints")

# Fine-tuning hyperparameters
BATCH_SIZE = 8
FINETUNE_EPOCHS = 30
LEARNING_RATE = 1e-4  # Lower LR for fine-tuning


# ============================================================
# Dataset for Fine-tuning
# ============================================================

class RecentDataset(Dataset):
    """Dataset from recent OpenAQ data with sliding window."""
    
    def __init__(self, daily_signals, edge_index, input_window=7, model_window=14):
        """
        Args:
            daily_signals: Tensor (num_nodes, num_days, 1)
            edge_index: Graph connectivity
            input_window: Number of days we actually have
            model_window: Number of days the model expects (14)
        """
        self.daily_signals = daily_signals
        self.edge_index = edge_index
        self.input_window = input_window
        self.model_window = model_window
        
        # We use day N-1 to predict day N
        self.num_days = daily_signals.shape[1]
        self.num_samples = max(0, self.num_days - input_window)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Input: days [idx, idx+input_window)
        # Target: day idx+input_window
        x = self.daily_signals[:, idx:idx+self.input_window, :]
        y = self.daily_signals[:, idx+self.input_window, :]
        
        # Pad x to model_window (prepend with MASK_TOKEN)
        if x.shape[1] < self.model_window:
            padding = torch.full(
                (x.shape[0], self.model_window - x.shape[1], 1),
                MASK_TOKEN
            )
            x = torch.cat([padding, x], dim=1)
        
        return x, y, self.edge_index


def fetch_and_prepare_city_data(city: str, hours: int = HOURS_TO_FETCH):
    """Fetch data for a city and prepare for training."""
    print(f"\n{'='*50}")
    print(f"Fetching data for {city.upper()}")
    print('='*50)
    
    # Get graph
    graph = fetch_city_graph(city, radius_km=15)
    num_nodes = len(graph.location_ids)
    
    if num_nodes == 0:
        print(f"No sensors in {city}, skipping...")
        return None
    
    print(f"Found {num_nodes} sensors")
    
    # Fetch hourly data
    signals = fetch_historical_signals(graph.location_ids, hours=hours)
    
    # Check coverage
    hourly_coverage = (signals != MASK_TOKEN).float().mean().item() * 100
    print(f"Hourly coverage: {hourly_coverage:.1f}%")
    
    if hourly_coverage < 30:
        print(f"Coverage too low, skipping {city}")
        return None
    
    # Aggregate to daily
    available_hours = signals.shape[1]
    num_days = available_hours // 24
    
    daily = torch.full((num_nodes, num_days, 1), MASK_TOKEN)
    
    for day in range(num_days):
        start_h = day * 24
        end_h = (day + 1) * 24
        
        for node in range(num_nodes):
            day_data = signals[node, start_h:end_h, 0]
            valid = day_data[day_data != MASK_TOKEN]
            if len(valid) > 0:
                daily[node, day, 0] = valid.mean()
    
    daily_coverage = (daily != MASK_TOKEN).float().mean().item() * 100
    print(f"Daily coverage: {daily_coverage:.1f}% ({num_days} days)")
    
    return {
        'city': city,
        'daily_signals': daily,
        'edge_index': graph.edge_index,
        'location_ids': graph.location_ids,
        'num_nodes': num_nodes,
        'num_days': num_days
    }


def finetune():
    """Main fine-tuning function."""
    print("\n" + "="*60)
    print("  FINE-TUNING ON RECENT OPENAQ DATA")
    print("="*60)
    
    # Step 1: Load pretrained model
    print("\n📦 Loading pretrained model...")
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        return
    
    model = PollutionForecaster.load_from_checkpoint(str(CHECKPOINT_PATH))
    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Lower learning rate for fine-tuning
    model.learning_rate = LEARNING_RATE
    
    # Step 2: Fetch recent data from all cities
    print("\n📡 Fetching recent data from OpenAQ...")
    
    city_data = []
    
    for city in CITIES:
        try:
            data = fetch_and_prepare_city_data(city, HOURS_TO_FETCH)
            if data is None:
                continue
            
            # Check if enough days
            num_days = data['num_days']
            if num_days < INPUT_WINDOW + 1:
                print(f"Not enough days for {city} (need {INPUT_WINDOW+1}, got {num_days})")
                continue
            
            city_data.append(data)
            print(f"Collected {num_days} days from {city}")
            
        except Exception as e:
            print(f"Error processing {city}: {e}")
            continue
    
    if not city_data:
        print("ERROR: No valid data collected!")
        return
    
    # Step 3: Fine-tune using a simple loop (handle different graph sizes)
    print(f"\n🔧 Fine-tuning model on {len(city_data)} cities...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(FINETUNE_EPOCHS):
        epoch_loss = 0.0
        num_batches = 0
        
        for data in city_data:
            daily = data['daily_signals']
            edge_index = data['edge_index']
            num_days = daily.shape[1]
            
            # Create sliding window samples
            for start_day in range(num_days - INPUT_WINDOW):
                # Get input (7 days) and pad to 14 days
                x = daily[:, start_day:start_day + INPUT_WINDOW, :]
                y = daily[:, start_day + INPUT_WINDOW, :]
                
                # Pad to model_window (14 days)
                if x.shape[1] < 14:
                    padding = torch.full((x.shape[0], 14 - x.shape[1], 1), MASK_TOKEN)
                    x = torch.cat([padding, x], dim=1)
                
                # Move to device
                x = x.to(device)
                y = y.to(device)
                edge_index = edge_index.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                preds, _, _ = model.model(x, edge_index, return_attention_weights=True)
                
                # Masked loss
                mask = (y != MASK_TOKEN)
                if mask.sum() == 0:
                    continue
                    
                loss = ((preds - y) ** 2 * mask).sum() / mask.sum()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1:3d}/{FINETUNE_EPOCHS} - Loss: {avg_loss:.4f} (RMSE: {avg_loss**0.5:.2f})")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), FINETUNE_CHECKPOINT_DIR / 'pollution-forecaster-finetuned-best.pt')
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Save final checkpoint in Lightning format
    trainer = pl.Trainer(accelerator='auto', devices=1)
    trainer.strategy.connect(model)
    trainer.save_checkpoint(FINETUNE_CHECKPOINT_DIR / 'pollution-forecaster-finetuned.ckpt')
    
    print("\n" + "="*60)
    print("✅ FINE-TUNING COMPLETE!")
    print("="*60)
    print(f"\nBest loss: {best_loss:.4f} (RMSE: {best_loss**0.5:.2f})")
    print(f"Models saved to: {FINETUNE_CHECKPOINT_DIR}/")


if __name__ == "__main__":
    finetune()
