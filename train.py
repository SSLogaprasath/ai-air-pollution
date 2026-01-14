"""
Training Pipeline for Air Quality Forecasting.

This module provides PyTorch Lightning-based training infrastructure
for the STGraphTransformer model, including:
- SlidingWindowDataset for creating input/target pairs
- Masked MSE Loss for handling missing data
- PollutionForecaster LightningModule
- Training utilities and callbacks
"""

import math
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger

from model import STGraphTransformer


# Mask token value (must match data_ingestion.py)
MASK_TOKEN: float = -1.0


class SlidingWindowDataset(Dataset):
    """
    Dataset that creates sliding window samples from time series data.
    
    Given a long sequence of air quality measurements, this dataset
    creates (input_window, target) pairs for supervised training.
    
    Example:
        If input_window=24 and we have 100 hours of data:
        - Sample 0: input=hours[0:24], target=hour[24]
        - Sample 1: input=hours[1:25], target=hour[25]
        - ...
    """
    
    def __init__(
        self,
        signals: Tensor,
        edge_index: Tensor,
        input_window: int = 24,
        forecast_horizon: int = 1,
        stride: int = 1
    ):
        """
        Initialize the sliding window dataset.
        
        Args:
            signals: PM2.5 time series of shape (num_nodes, num_hours, 1)
            edge_index: Graph connectivity of shape (2, num_edges)
            input_window: Number of hours to use as input (default: 24)
            forecast_horizon: Number of hours to forecast (default: 1)
            stride: Step size between samples (default: 1)
        """
        super().__init__()
        
        self.signals = signals
        self.edge_index = edge_index
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        
        self.num_nodes, self.num_hours, self.num_features = signals.shape
        
        # Calculate number of valid samples
        self.num_samples = max(
            0,
            (self.num_hours - input_window - forecast_horizon + 1) // stride
        )
        
        if self.num_samples == 0:
            raise ValueError(
                f"Not enough data for sliding window. "
                f"Have {self.num_hours} hours, need at least "
                f"{input_window + forecast_horizon} hours."
            )
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
                - x: Input sequence of shape (num_nodes, input_window, 1)
                - y: Target values of shape (num_nodes, forecast_horizon)
                - edge_index: Graph connectivity
                - mask: Boolean mask for valid (non-missing) targets
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.input_window
        target_start = end_idx
        target_end = target_start + self.forecast_horizon
        
        # Extract input window
        x = self.signals[:, start_idx:end_idx, :]  # (N, T_in, 1)
        
        # Extract target(s)
        y = self.signals[:, target_start:target_end, 0]  # (N, T_out)
        
        # Create mask for valid targets (not missing)
        mask = y != MASK_TOKEN
        
        return {
            'x': x,
            'y': y,
            'edge_index': self.edge_index,
            'mask': mask
        }


def sliding_window_collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """
    Collate function for SlidingWindowDataset.
    
    Since all samples share the same graph structure, we just stack
    the time series data and keep a single copy of edge_index.
    
    For simplicity, this implementation assumes batch_size=1 since
    each sample already contains all nodes. For larger batches,
    we would need to use PyG's Batch mechanism.
    """
    # For now, just return the first (and typically only) sample
    # In practice with batch_size > 1, you'd stack along a batch dimension
    if len(batch) == 1:
        return batch[0]
    
    # Stack multiple time windows
    x = torch.stack([b['x'] for b in batch], dim=0)  # (B, N, T, 1)
    y = torch.stack([b['y'] for b in batch], dim=0)  # (B, N, T_out)
    mask = torch.stack([b['mask'] for b in batch], dim=0)  # (B, N, T_out)
    edge_index = batch[0]['edge_index']  # Same for all samples
    
    return {
        'x': x,
        'y': y,
        'edge_index': edge_index,
        'mask': mask
    }


class MaskedMSELoss(nn.Module):
    """
    Masked Mean Squared Error Loss.
    
    This loss function ignores predictions where the ground truth
    is missing (marked with MASK_TOKEN = -1.0). This prevents the
    model from being penalized for missing measurements.
    """
    
    def __init__(self, mask_token: float = MASK_TOKEN, reduction: str = 'mean'):
        """
        Args:
            mask_token: Value indicating missing data (default: -1.0)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.mask_token = mask_token
        self.reduction = reduction
        
    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute masked MSE loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            mask: Optional boolean mask (True = valid, False = ignore)
                  If None, will be computed from targets != mask_token
                  
        Returns:
            Scalar loss value (or per-element if reduction='none')
        """
        # Create mask if not provided
        if mask is None:
            mask = targets != self.mask_token
        
        # Ensure mask is boolean
        mask = mask.bool()
        
        # Count valid elements
        num_valid = mask.sum()
        
        # Handle edge case: no valid elements
        if num_valid == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Compute squared errors only for valid elements
        squared_errors = (predictions - targets) ** 2
        
        # Apply mask
        masked_errors = squared_errors * mask.float()
        
        if self.reduction == 'none':
            return masked_errors
        elif self.reduction == 'sum':
            return masked_errors.sum()
        else:  # mean
            return masked_errors.sum() / num_valid


class PollutionForecaster(pl.LightningModule):
    """
    PyTorch Lightning module for air quality forecasting.
    
    Wraps the STGraphTransformer model with training, validation,
    and testing logic, including masked loss computation.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_temporal_layers: int = 2,
        max_seq_len: int = 168,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 100
    ):
        """
        Initialize the forecaster.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for model
            num_heads: Number of attention heads
            num_temporal_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            warmup_epochs: Number of warmup epochs for scheduler
            max_epochs: Total training epochs
        """
        super().__init__()
        
        # Save hyperparameters for logging
        self.save_hyperparameters()
        
        # Model
        self.model = STGraphTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_temporal_layers=num_temporal_layers,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # Loss function
        self.loss_fn = MaskedMSELoss()
        
        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        return_attention_weights: bool = False
    ) -> Tuple[Tensor, Optional[Tuple], Optional[Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (num_nodes, seq_len, input_dim)
            edge_index: Graph connectivity
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Model outputs (forecast, spatial_weights, temporal_weights)
        """
        # Handle masked inputs: replace -1.0 with 0.0 for processing
        x_processed = x.clone()
        x_processed[x_processed == MASK_TOKEN] = 0.0
        
        return self.model(x_processed, edge_index, return_attention_weights)
    
    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """
        Training step.
        
        Args:
            batch: Dictionary with 'x', 'y', 'edge_index', 'mask'
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        x = batch['x']
        y = batch['y']
        edge_index = batch['edge_index']
        mask = batch['mask']
        
        # Handle batched input: (B, N, T, F) -> process each sample
        if x.dim() == 4:
            # Batch processing - for simplicity, process first sample
            x = x[0]  # (N, T, F)
            y = y[0]  # (N, T_out)
            mask = mask[0]  # (N, T_out)
        
        # Forward pass (no attention weights during training for speed)
        predictions, _, _ = self(x, edge_index, return_attention_weights=False)
        
        # Compute masked loss
        loss = self.loss_fn(predictions, y, mask)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log percentage of valid data
        valid_pct = mask.float().mean() * 100
        self.log('train_valid_pct', valid_pct, on_step=False, on_epoch=True)
        
        self.training_step_outputs.append(loss.detach())
        
        return loss
    
    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """
        Validation step.
        """
        x = batch['x']
        y = batch['y']
        edge_index = batch['edge_index']
        mask = batch['mask']
        
        # Handle batched input
        if x.dim() == 4:
            x = x[0]
            y = y[0]
            mask = mask[0]
        
        # Forward pass
        predictions, spatial_weights, temporal_weights = self(
            x, edge_index, return_attention_weights=True
        )
        
        # Compute masked loss
        loss = self.loss_fn(predictions, y, mask)
        
        # Compute RMSE for valid predictions
        valid_mask = mask.bool()
        if valid_mask.sum() > 0:
            rmse = torch.sqrt(
                F.mse_loss(predictions[valid_mask], y[valid_mask])
            )
        else:
            rmse = torch.tensor(0.0, device=self.device)
        
        # Compute MAE
        if valid_mask.sum() > 0:
            mae = F.l1_loss(predictions[valid_mask], y[valid_mask])
        else:
            mae = torch.tensor(0.0, device=self.device)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_rmse', rmse, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_epoch=True)
        
        output = {'val_loss': loss, 'val_rmse': rmse, 'val_mae': mae}
        self.validation_step_outputs.append(output)
        
        return output
    
    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """
        Test step - same as validation but with different logging.
        """
        x = batch['x']
        y = batch['y']
        edge_index = batch['edge_index']
        mask = batch['mask']
        
        if x.dim() == 4:
            x = x[0]
            y = y[0]
            mask = mask[0]
        
        predictions, _, _ = self(x, edge_index, return_attention_weights=False)
        
        loss = self.loss_fn(predictions, y, mask)
        
        valid_mask = mask.bool()
        if valid_mask.sum() > 0:
            rmse = torch.sqrt(F.mse_loss(predictions[valid_mask], y[valid_mask]))
            mae = F.l1_loss(predictions[valid_mask], y[valid_mask])
        else:
            rmse = torch.tensor(0.0, device=self.device)
            mae = torch.tensor(0.0, device=self.device)
        
        self.log('test_loss', loss)
        self.log('test_rmse', rmse)
        self.log('test_mae', mae)
        
        return {'test_loss': loss, 'test_rmse': rmse, 'test_mae': mae}
    
    def on_train_epoch_end(self):
        """Aggregate training metrics at epoch end."""
        if self.training_step_outputs:
            avg_loss = torch.stack(self.training_step_outputs).mean()
            self.log('train_loss_epoch', avg_loss)
            self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """Aggregate validation metrics at epoch end."""
        if self.validation_step_outputs:
            avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
            avg_rmse = torch.stack([x['val_rmse'] for x in self.validation_step_outputs]).mean()
            self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Uses AdamW with cosine annealing and warmup.
        """
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Cosine annealing with warmup
        warmup_epochs = self.hparams.warmup_epochs
        max_epochs = self.hparams.max_epochs
        
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine decay
                progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def predict_with_interpretation(
        self,
        x: Tensor,
        edge_index: Tensor,
        node_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Make predictions with interpretability outputs.
        
        Args:
            x: Input tensor of shape (num_nodes, seq_len, 1)
            edge_index: Graph connectivity
            node_names: Optional list of sensor names
            
        Returns:
            Dictionary with predictions and attention interpretations
        """
        self.eval()
        with torch.no_grad():
            predictions, spatial_weights, temporal_weights = self(
                x, edge_index, return_attention_weights=True
            )
        
        interpretation = self.model.get_attention_interpretation(
            spatial_weights, temporal_weights, node_names
        )
        
        return {
            'predictions': predictions.cpu().numpy(),
            'spatial_interpretation': interpretation.get('spatial'),
            'temporal_interpretation': interpretation.get('temporal')
        }


def create_dataloaders(
    signals: Tensor,
    edge_index: Tensor,
    input_window: int = 24,
    forecast_horizon: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 1,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Splits the time series temporally (not randomly) to avoid
    data leakage from future to past.
    
    Args:
        signals: Full time series (num_nodes, num_hours, 1)
        edge_index: Graph connectivity
        input_window: Input window size
        forecast_horizon: Forecast horizon
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    num_hours = signals.shape[1]
    
    # Temporal split points
    train_end = int(num_hours * train_ratio)
    val_end = int(num_hours * (train_ratio + val_ratio))
    
    # Ensure minimum data for each split
    min_hours = input_window + forecast_horizon
    
    train_signals = signals[:, :train_end, :]
    val_signals = signals[:, train_end:val_end, :]
    test_signals = signals[:, val_end:, :]
    
    # Create datasets
    datasets = []
    for split_signals, name in [
        (train_signals, 'train'),
        (val_signals, 'val'),
        (test_signals, 'test')
    ]:
        if split_signals.shape[1] >= min_hours:
            datasets.append(
                SlidingWindowDataset(
                    signals=split_signals,
                    edge_index=edge_index,
                    input_window=input_window,
                    forecast_horizon=forecast_horizon
                )
            )
        else:
            print(f"Warning: {name} split has insufficient data ({split_signals.shape[1]} hours)")
            # Create minimal dummy dataset
            datasets.append(None)
    
    # Create dataloaders
    dataloaders = []
    for dataset, shuffle in zip(datasets, [True, False, False]):
        if dataset is not None:
            dataloaders.append(
                DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    collate_fn=sliding_window_collate_fn,
                    pin_memory=True
                )
            )
        else:
            dataloaders.append(None)
    
    return tuple(dataloaders)


def get_callbacks(
    checkpoint_dir: str = 'checkpoints',
    patience: int = 10
) -> List[pl.Callback]:
    """
    Create training callbacks.
    
    Args:
        checkpoint_dir: Directory for saving checkpoints
        patience: Early stopping patience
        
    Returns:
        List of callbacks
    """
    callbacks = [
        # Model checkpoint - save best model
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='pollution-forecaster-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min',
            verbose=True
        ),
        
        # Learning rate monitor
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # Add progress bar if rich is available
    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass
    
    return callbacks


# ============================================================
# Main Training Script
# ============================================================

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    print("=" * 60)
    print("Air Quality Forecasting - Training Pipeline")
    print("=" * 60)
    
    # Configuration
    NUM_NODES = 10
    NUM_HOURS = 200  # Simulate ~1 week of hourly data
    INPUT_WINDOW = 24  # 24 hours input
    FORECAST_HORIZON = 1  # Predict next hour
    HIDDEN_DIM = 64
    NUM_HEADS = 4
    BATCH_SIZE = 1
    MAX_EPOCHS = 5  # Short for demo
    
    print(f"\nConfiguration:")
    print(f"  Nodes (sensors): {NUM_NODES}")
    print(f"  Hours of data: {NUM_HOURS}")
    print(f"  Input window: {INPUT_WINDOW} hours")
    print(f"  Forecast horizon: {FORECAST_HORIZON} hour(s)")
    
    # ============== Generate Dummy Data ==============
    print("\n" + "-" * 40)
    print("Generating dummy data...")
    
    # Simulate PM2.5 values with some structure
    # Base value + daily cycle + random noise
    t = torch.arange(NUM_HOURS).float()
    daily_cycle = 30 * torch.sin(2 * math.pi * t / 24)  # 24-hour cycle
    base = 80 + 20 * torch.randn(NUM_NODES, 1)  # Different base per sensor
    noise = 10 * torch.randn(NUM_NODES, NUM_HOURS)
    
    signals = (base + daily_cycle.unsqueeze(0) + noise).unsqueeze(-1)  # (N, T, 1)
    signals = torch.clamp(signals, min=0)  # PM2.5 can't be negative
    
    # Add some missing values (mask tokens)
    missing_mask = torch.rand(NUM_NODES, NUM_HOURS) < 0.05  # 5% missing
    signals[missing_mask.unsqueeze(-1).expand_as(signals)] = MASK_TOKEN
    
    print(f"  Signals shape: {signals.shape}")
    print(f"  Missing data: {missing_mask.float().mean():.1%}")
    
    # Create graph (ring + some random edges)
    edge_list = []
    for i in range(NUM_NODES):
        edge_list.append([i, (i + 1) % NUM_NODES])
        edge_list.append([(i + 1) % NUM_NODES, i])
        # Add random edges
        for j in range(2):
            target = torch.randint(0, NUM_NODES, (1,)).item()
            if target != i:
                edge_list.append([i, target])
                edge_list.append([target, i])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    # Remove duplicates
    edge_index = torch.unique(edge_index, dim=1)
    
    print(f"  Edge index shape: {edge_index.shape}")
    print(f"  Number of edges: {edge_index.shape[1]}")
    
    # ============== Create DataLoaders ==============
    print("\n" + "-" * 40)
    print("Creating dataloaders...")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        signals=signals,
        edge_index=edge_index,
        input_window=INPUT_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
        train_ratio=0.7,
        val_ratio=0.15,
        batch_size=BATCH_SIZE
    )
    
    if train_loader:
        print(f"  Train samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"  Val samples: {len(val_loader.dataset)}")
    if test_loader:
        print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Verify shapes with a sample batch
    sample_batch = next(iter(train_loader))
    print(f"\n  Sample batch shapes:")
    print(f"    x: {sample_batch['x'].shape}")
    print(f"    y: {sample_batch['y'].shape}")
    print(f"    mask: {sample_batch['mask'].shape}")
    
    # ============== Create Model ==============
    print("\n" + "-" * 40)
    print("Creating model...")
    
    model = PollutionForecaster(
        input_dim=1,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_temporal_layers=2,
        max_seq_len=INPUT_WINDOW,
        learning_rate=1e-3,
        warmup_epochs=2,
        max_epochs=MAX_EPOCHS
    )
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ============== Quick Forward Pass Test ==============
    print("\n" + "-" * 40)
    print("Testing forward pass...")
    
    model.eval()
    with torch.no_grad():
        x = sample_batch['x']
        edge_index_batch = sample_batch['edge_index']
        
        forecast, spatial_weights, temporal_weights = model(
            x, edge_index_batch, return_attention_weights=True
        )
        
        print(f"  Forecast shape: {forecast.shape}")
        if spatial_weights:
            print(f"  Spatial attention shape: {spatial_weights[1].shape}")
        if temporal_weights is not None:
            print(f"  Temporal attention shape: {temporal_weights.shape}")
    
    # ============== Test Loss Computation ==============
    print("\n" + "-" * 40)
    print("Testing masked loss...")
    
    loss_fn = MaskedMSELoss()
    y = sample_batch['y']
    mask = sample_batch['mask']
    
    loss = loss_fn(forecast, y, mask)
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Valid targets: {mask.sum().item()}/{mask.numel()}")
    
    # ============== Training ==============
    print("\n" + "-" * 40)
    print(f"Starting training ({MAX_EPOCHS} epochs)...")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='auto',  # Use GPU if available
        devices=1,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=3, mode='min'),
        ],
        enable_checkpointing=False,  # Disable for demo
        enable_progress_bar=True,
        log_every_n_steps=1,
        logger=False  # Disable logging for demo
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # ============== Final Evaluation ==============
    print("\nFinal evaluation on test set...")
    
    if test_loader:
        test_results = trainer.test(model, test_loader, verbose=False)
        print(f"  Test Loss: {test_results[0]['test_loss']:.4f}")
        print(f"  Test RMSE: {test_results[0]['test_rmse']:.4f}")
        print(f"  Test MAE: {test_results[0]['test_mae']:.4f}")
    
    # ============== Interpretation Demo ==============
    print("\n" + "-" * 40)
    print("Interpretation demo...")
    
    model.eval()
    result = model.predict_with_interpretation(x, edge_index_batch)
    
    print(f"  Predictions shape: {result['predictions'].shape}")
    
    if result['spatial_interpretation']:
        print(f"  Spatial attention available: Yes")
        print(f"    - Shows which neighbor sensors influenced predictions")
        
    if result['temporal_interpretation']:
        print(f"  Temporal attention available: Yes")
        print(f"    - Shows which past hours influenced the forecast")
    
    print("\n✓ All training pipeline tests passed!")
