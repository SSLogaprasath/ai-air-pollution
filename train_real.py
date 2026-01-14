"""
Train STGraphTransformer on Time Series Air Quality Data of India (2010-2023).

This dataset has 454 station files with hourly PM2.5 data from 2016-2023.
We aggregate to daily and train the model on recent years for better accuracy.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar

from model import STGraphTransformer
from train import PollutionForecaster, MaskedMSELoss

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data/2010-2023")
CHECKPOINT_DIR = Path("checkpoints")

# Training hyperparameters
INPUT_WINDOW = 14       # Days of history
BATCH_SIZE = 32
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3

# Model hyperparameters
HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1

# Data filtering
MIN_YEAR = 2020  # Use recent data (2020-2023) for training
MASK_TOKEN = -1.0


# ============================================================
# Dataset
# ============================================================

class StationDataset(Dataset):
    """Dataset from multiple station CSV files."""
    
    def __init__(self, data_dir: Path, min_year: int = 2020, input_window: int = 14):
        self.input_window = input_window
        self.samples = []
        
        print(f"Loading station data from {data_dir}...")
        
        # Get all CSV files
        csv_files = sorted(data_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} station files")
        
        # Group files by state (first 2 chars of filename)
        state_files: Dict[str, List[Path]] = {}
        for f in csv_files:
            state = f.stem[:2]
            if state not in state_files:
                state_files[state] = []
            state_files[state].append(f)
        
        print(f"States: {list(state_files.keys())}")
        
        # Process each state as a graph
        for state, files in state_files.items():
            if len(files) < 2:  # Need at least 2 stations for a graph
                continue
                
            self._process_state(state, files, min_year)
        
        print(f"Total samples: {len(self.samples)}")
    
    def _process_state(self, state: str, files: List[Path], min_year: int):
        """Process all stations in a state as a single graph."""
        
        # Load all station data
        station_data = {}
        for f in files:
            try:
                df = pd.read_csv(f)
                if 'PM2.5 (ug/m3)' not in df.columns or 'From Date' not in df.columns:
                    continue
                
                # Parse dates and filter by year
                df['datetime'] = pd.to_datetime(df['From Date'], errors='coerce')
                df = df.dropna(subset=['datetime'])
                df = df[df['datetime'].dt.year >= min_year]
                
                if len(df) < 24 * 30:  # Need at least 30 days of data
                    continue
                
                # Aggregate to daily
                df['date'] = df['datetime'].dt.date
                daily = df.groupby('date')['PM2.5 (ug/m3)'].mean().reset_index()
                daily.columns = ['date', 'pm25']
                daily = daily.dropna()
                daily = daily.sort_values('date')
                
                if len(daily) >= self.input_window + 1:
                    station_data[f.stem] = daily
                    
            except Exception as e:
                continue
        
        if len(station_data) < 2:
            return
        
        # Find common date range
        all_dates = set()
        for name, df in station_data.items():
            dates = set(df['date'].tolist())
            if len(all_dates) == 0:
                all_dates = dates
            else:
                all_dates = all_dates.intersection(dates)
        
        common_dates = sorted(list(all_dates))
        if len(common_dates) < self.input_window + 1:
            return
        
        # Build aligned data matrix
        station_names = list(station_data.keys())
        num_stations = len(station_names)
        num_days = len(common_dates)
        
        data_matrix = np.full((num_stations, num_days), MASK_TOKEN)
        
        for i, name in enumerate(station_names):
            df = station_data[name]
            date_to_pm25 = dict(zip(df['date'], df['pm25']))
            for j, date in enumerate(common_dates):
                if date in date_to_pm25:
                    val = date_to_pm25[date]
                    if pd.notna(val) and val >= 0:
                        data_matrix[i, j] = val
        
        # Build fully connected graph for this state
        edge_index = []
        for i in range(num_stations):
            for j in range(num_stations):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        
        # Create sliding window samples
        for start_idx in range(num_days - self.input_window):
            x = data_matrix[:, start_idx:start_idx + self.input_window]
            y = data_matrix[:, start_idx + self.input_window]
            
            # Check coverage
            x_valid = (x != MASK_TOKEN).mean()
            y_valid = (y != MASK_TOKEN).mean()
            
            if x_valid >= 0.5 and y_valid >= 0.5:  # At least 50% coverage
                self.samples.append({
                    'x': torch.tensor(x, dtype=torch.float32).unsqueeze(-1),
                    'y': torch.tensor(y, dtype=torch.float32).unsqueeze(-1),
                    'edge_index': edge_index,
                    'state': state
                })
        
        if len(self.samples) > 0 and len(station_data) > 0:
            print(f"  {state}: {len(station_data)} stations, {len(common_dates)} days, "
                  f"{len([s for s in self.samples if s['state'] == state])} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['x'], sample['y'], sample['edge_index']


def collate_fn(batch):
    """Custom collate for variable-sized graphs - returns dict format."""
    x, y, edge_index = batch[0]
    
    # Create mask (True where data is valid)
    mask = (y != MASK_TOKEN)
    
    return {
        'x': x.unsqueeze(0),  # Add batch dim
        'y': y.unsqueeze(0),
        'edge_index': edge_index,
        'mask': mask.unsqueeze(0)
    }


# ============================================================
# Training
# ============================================================

def train():
    print("=" * 60)
    print("  TRAINING ON TIME SERIES DATA (2020-2023)")
    print("=" * 60)
    
    # Set precision for tensor cores
    torch.set_float32_matmul_precision('medium')
    
    # Create dataset
    dataset = StationDataset(DATA_DIR, min_year=MIN_YEAR, input_window=INPUT_WINDOW)
    
    if len(dataset) == 0:
        print("ERROR: No valid samples found!")
        return
    
    # Split train/val/test (70/15/15)
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")
    
    # Create data loaders (batch_size=1 due to variable graph sizes)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, 
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    
    # Create model
    model = PollutionForecaster(
        input_dim=1,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_temporal_layers=NUM_LAYERS,
        max_seq_len=INPUT_WINDOW,
        dropout=DROPOUT,
        learning_rate=LEARNING_RATE
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(CHECKPOINT_DIR),
        filename='pollution-forecaster-best',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_last=True
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        mode='min',
        verbose=True
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stop, RichProgressBar()],
        accelerator='auto',
        devices=1,
        log_every_n_steps=10
    )
    
    # Train
    print("\n🚀 Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    print("\n📊 Testing...")
    results = trainer.test(model, test_loader)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nBest model saved to: {CHECKPOINT_DIR / 'pollution-forecaster-best.ckpt'}")
    print(f"Test RMSE: {results[0].get('test_rmse', 'N/A'):.2f} µg/m³")
    print(f"Test MAE:  {results[0].get('test_mae', 'N/A'):.2f} µg/m³")


if __name__ == "__main__":
    train()
