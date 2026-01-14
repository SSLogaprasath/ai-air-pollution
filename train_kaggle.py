"""
Train STGraphTransformer on Kaggle Air Quality Data in India.

Dataset: city_day.csv (daily PM2.5 data for 26 Indian cities)
Approach: Treat cities as nodes in a graph, predict next-day PM2.5
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar

from model import STGraphTransformer
from train import MaskedMSELoss


# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path("data/city_day.csv")
CHECKPOINT_DIR = Path("checkpoints")
MASK_TOKEN = -1.0

# Model hyperparameters
INPUT_WINDOW = 14      # 14 days of history (2 weeks)
HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_TEMPORAL_LAYERS = 2

# Training hyperparameters
BATCH_SIZE = 32
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.8

# Cities to use (major cities with good data coverage)
TARGET_CITIES = [
    'Delhi', 'Mumbai', 'Bengaluru', 'Chennai', 'Kolkata',
    'Hyderabad', 'Ahmedabad', 'Lucknow', 'Jaipur', 'Patna'
]


# ============================================================
# City Graph (Fixed Topology Based on Geography)
# ============================================================

# Approximate coordinates for building city graph
CITY_COORDS = {
    'Delhi': (28.61, 77.21),
    'Mumbai': (19.08, 72.88),
    'Bengaluru': (12.97, 77.59),
    'Chennai': (13.08, 80.27),
    'Kolkata': (22.57, 88.36),
    'Hyderabad': (17.38, 78.49),
    'Ahmedabad': (23.02, 72.57),
    'Lucknow': (26.85, 80.95),
    'Jaipur': (26.91, 75.79),
    'Patna': (25.59, 85.14),
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_city_graph(cities: list, radius_km: float = 1000):
    """
    Build a graph connecting cities within radius_km of each other.
    For city-level data, we use larger radius since cities are far apart.
    """
    n = len(cities)
    edges_src = []
    edges_dst = []
    
    for i, city_i in enumerate(cities):
        if city_i not in CITY_COORDS:
            continue
        lat_i, lon_i = CITY_COORDS[city_i]
        
        for j, city_j in enumerate(cities):
            if city_j not in CITY_COORDS:
                continue
            lat_j, lon_j = CITY_COORDS[city_j]
            
            dist = haversine_distance(lat_i, lon_i, lat_j, lon_j)
            if dist <= radius_km:
                edges_src.append(i)
                edges_dst.append(j)
    
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return edge_index


# ============================================================
# Dataset
# ============================================================

class CityDayDataset(Dataset):
    """
    Dataset for city-level daily PM2.5 prediction.
    
    Each sample:
    - Input: (num_cities, input_window, 1) - past days PM2.5
    - Target: (num_cities, 1) - next day PM2.5
    """
    
    def __init__(self, df: pd.DataFrame, cities: list, input_window: int = 14):
        self.input_window = input_window
        self.cities = cities
        self.city_to_idx = {city: i for i, city in enumerate(cities)}
        
        # Pivot data: rows=dates, columns=cities, values=PM2.5
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[df['City'].isin(cities)]
        
        pivot = df.pivot_table(
            index='Date', 
            columns='City', 
            values='PM2.5',
            aggfunc='mean'
        )
        
        # Reorder columns to match our city order
        pivot = pivot.reindex(columns=cities)
        
        # Fill NaN with mask token
        pivot = pivot.fillna(MASK_TOKEN)
        
        self.data = torch.tensor(pivot.values, dtype=torch.float32)  # (num_days, num_cities)
        self.dates = pivot.index.tolist()
        
        print(f"Dataset: {len(self.dates)} days, {len(cities)} cities")
        print(f"Date range: {self.dates[0]} to {self.dates[-1]}")
        
        # Check coverage
        valid = (self.data != MASK_TOKEN).float().mean().item() * 100
        print(f"Data coverage: {valid:.1f}%")
    
    def __len__(self):
        return len(self.dates) - self.input_window - 1
    
    def __getitem__(self, idx):
        # Input: days [idx : idx + input_window]
        # Target: day [idx + input_window]
        
        x = self.data[idx : idx + self.input_window, :]  # (window, cities)
        y = self.data[idx + self.input_window, :]        # (cities,)
        
        # Reshape to (cities, window, 1) for model
        x = x.T.unsqueeze(-1)  # (cities, window, 1)
        y = y.unsqueeze(-1)    # (cities, 1)
        
        return x, y


# ============================================================
# Lightning Module
# ============================================================

class CityForecaster(pl.LightningModule):
    """PyTorch Lightning module for city-level PM2.5 forecasting."""
    
    def __init__(
        self,
        num_cities: int,
        edge_index: torch.Tensor,
        input_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        num_temporal_layers: int = NUM_TEMPORAL_LAYERS,
        max_seq_len: int = INPUT_WINDOW,
        learning_rate: float = LEARNING_RATE
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['edge_index'])
        
        self.edge_index = edge_index
        self.learning_rate = learning_rate
        
        self.model = STGraphTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_temporal_layers=num_temporal_layers,
            max_seq_len=max_seq_len
        )
        
        self.loss_fn = MaskedMSELoss(mask_token=MASK_TOKEN)
    
    def forward(self, x):
        edge_index = self.edge_index.to(x.device)
        return self.model(x, edge_index, return_attention_weights=False)
    
    def _shared_step(self, batch, stage: str):
        x, y = batch  # x: (batch, cities, window, 1), y: (batch, cities, 1)
        
        # Process each sample in batch (model expects unbatched input)
        batch_size = x.shape[0]
        predictions = []
        
        edge_index = self.edge_index.to(x.device)
        
        for i in range(batch_size):
            # Model returns (forecast, spatial_weights, temporal_weights)
            y_hat_i, _, _ = self.model(x[i], edge_index, return_attention_weights=False)
            predictions.append(y_hat_i)
        
        y_hat = torch.stack(predictions, dim=0)  # (batch, cities, 1)
        
        loss = self.loss_fn(y_hat, y)
        
        # Calculate metrics on valid (non-masked) values
        mask = y != MASK_TOKEN
        if mask.sum() > 0:
            rmse = torch.sqrt(((y_hat[mask] - y[mask]) ** 2).mean())
            mae = (y_hat[mask] - y[mask]).abs().mean()
        else:
            rmse = torch.tensor(0.0)
            mae = torch.tensor(0.0)
        
        self.log(f'{stage}_loss', loss, prog_bar=True)
        self.log(f'{stage}_rmse', rmse, prog_bar=True)
        self.log(f'{stage}_mae', mae)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, 'test')
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
        )
        return [optimizer], [scheduler]


# ============================================================
# Training
# ============================================================

def train():
    print("="*60)
    print("  TRAINING ON KAGGLE AIR QUALITY DATA")
    print("="*60)
    
    # Load data
    print("\n📊 Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Total records: {len(df)}")
    
    # Filter to target cities
    df_filtered = df[df['City'].isin(TARGET_CITIES)]
    print(f"Records for target cities: {len(df_filtered)}")
    
    # Build city graph
    print("\n🌐 Building city graph...")
    edge_index = build_city_graph(TARGET_CITIES, radius_km=1000)
    print(f"Cities: {len(TARGET_CITIES)}, Edges: {edge_index.shape[1]}")
    
    # Create dataset
    print("\n📦 Creating dataset...")
    full_dataset = CityDayDataset(df_filtered, TARGET_CITIES, INPUT_WINDOW)
    
    # Split into train/val
    total_samples = len(full_dataset)
    train_size = int(total_samples * TRAIN_SPLIT)
    val_size = total_samples - train_size
    
    # Use temporal split (not random) to avoid data leakage
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, total_samples))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\n🧠 Creating model...")
    model = CityForecaster(
        num_cities=len(TARGET_CITIES),
        edge_index=edge_index,
        input_dim=1,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_temporal_layers=NUM_TEMPORAL_LAYERS,
        max_seq_len=INPUT_WINDOW,
        learning_rate=LEARNING_RATE
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Callbacks
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(CHECKPOINT_DIR),
        filename='pollution-forecaster-best',
        save_top_k=1,
        monitor='val_rmse',
        mode='min',
        save_last=True
    )
    
    early_stop = EarlyStopping(
        monitor='val_rmse',
        patience=15,
        mode='min',
        verbose=True
    )
    
    # Trainer
    print("\n🚀 Starting training...")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stop, RichProgressBar()],
        accelerator='auto',
        devices=1,
        log_every_n_steps=5
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    print("\n📈 Final evaluation...")
    results = trainer.test(model, val_loader)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\nBest model saved to: {CHECKPOINT_DIR / 'pollution-forecaster-best.ckpt'}")
    print(f"Final Val RMSE: {results[0].get('test_rmse', 'N/A'):.2f} µg/m³")
    print(f"Final Val MAE: {results[0].get('test_mae', 'N/A'):.2f} µg/m³")
    
    return model


if __name__ == "__main__":
    train()
