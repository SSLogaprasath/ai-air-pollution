"""
Spatiotemporal Graph Transformer for Air Quality Forecasting.

This module implements a hybrid architecture combining Graph Attention Networks
(for spatial dependencies between sensors) with Transformers (for temporal patterns).
The model is designed for interpretability, returning attention weights that explain
which neighboring sensors and which past time steps influenced the prediction.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encodings for temporal sequences.
    
    Unlike fixed sinusoidal encodings, learnable encodings can adapt
    to the specific temporal patterns in air quality data.
    """
    
    def __init__(self, max_len: int, d_model: int):
        """
        Args:
            max_len: Maximum sequence length
            d_model: Embedding dimension
        """
        super().__init__()
        self.positional_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encodings to input tensor.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model) or (num_nodes, seq_len, d_model)
            
        Returns:
            Tensor with positional encodings added
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        pos_encoding = self.positional_embedding(positions)  # (seq_len, d_model)
        return x + pos_encoding.unsqueeze(0)  # Broadcasting over batch/nodes


class SpatialEncoder(nn.Module):
    """
    Graph Attention Network encoder for spatial dependencies.
    
    Uses GATv2Conv which allows dynamic attention based on both source
    and target node features, enabling the model to learn directional
    relationships (e.g., upwind vs downwind sensors).
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True
    ):
        """
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension per head
            num_heads: Number of attention heads
            dropout: Dropout rate for attention weights
            concat: If True, concatenate heads; if False, average them
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        
        self.gat = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=num_heads,
            dropout=dropout,
            concat=concat,
            add_self_loops=True,
            share_weights=False
        )
        
        # Output dimension depends on whether we concatenate heads
        self.out_dim = out_channels * num_heads if concat else out_channels
        
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        return_attention_weights: bool = True
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass through spatial encoder.
        
        Args:
            x: Node features of shape (num_nodes, in_channels)
            edge_index: Graph connectivity of shape (2, num_edges)
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Tuple of:
                - Output features of shape (num_nodes, out_dim)
                - Attention weights tuple (edge_index, attention_weights) if requested
        """
        if return_attention_weights:
            out, (edge_index_out, attention_weights) = self.gat(
                x, edge_index, return_attention_weights=True
            )
            return out, (edge_index_out, attention_weights)
        else:
            out = self.gat(x, edge_index)
            return out, None


class TemporalEncoder(nn.Module):
    """
    Transformer encoder for temporal dependencies.
    
    Processes the time sequence for each node to capture temporal
    patterns like daily cycles, trends, and sudden changes.
    Returns attention weights for interpretability.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Custom transformer layer that stores attention weights
        self.layers = nn.ModuleList([
            TransformerLayerWithAttention(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        x: Tensor,
        return_attention_weights: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass through temporal encoder.
        
        Args:
            x: Input tensor of shape (seq_len, batch, d_model) - Transformer format
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Tuple of:
                - Output tensor of shape (seq_len, batch, d_model)
                - Attention weights of shape (num_layers, batch, num_heads, seq_len, seq_len)
        """
        attention_weights_list = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, return_attention_weights=return_attention_weights)
            if attn_weights is not None:
                attention_weights_list.append(attn_weights)
        
        x = self.norm(x)
        
        if return_attention_weights and attention_weights_list:
            # Stack attention weights from all layers
            attention_weights = torch.stack(attention_weights_list, dim=0)
            return x, attention_weights
        
        return x, None


class TransformerLayerWithAttention(nn.Module):
    """
    Custom Transformer layer that returns attention weights.
    
    Standard nn.TransformerEncoderLayer doesn't expose attention weights,
    so we implement a custom version for interpretability.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False  # We use (seq, batch, features) format
        )
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(
        self,
        x: Tensor,
        return_attention_weights: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass with optional attention weight output.
        
        Args:
            x: Input of shape (seq_len, batch, d_model)
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Tuple of output tensor and optional attention weights
        """
        # Self-attention with residual connection
        attn_output, attn_weights = self.self_attn(
            x, x, x,
            need_weights=return_attention_weights,
            average_attn_weights=False  # Return per-head attention
        )
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feedforward with residual connection
        ff_output = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x, attn_weights


class STGraphTransformer(nn.Module):
    """
    Spatiotemporal Graph Transformer for Air Quality Forecasting.
    
    This architecture combines:
    1. GATv2 for spatial encoding (sensor-to-sensor attention)
    2. Transformer for temporal encoding (time step attention)
    
    Key features for interpretability:
    - Returns spatial attention weights showing which neighbor sensors
      influenced each prediction (e.g., upwind vs downwind)
    - Returns temporal attention weights showing which past hours
      were most important for the forecast
    
    Architecture:
        Input PM2.5 -> Embedding -> Positional Encoding
            -> Spatial Encoder (GAT) -> Temporal Encoder (Transformer)
            -> Decoder -> Forecast
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_temporal_layers: int = 2,
        max_seq_len: int = 168,  # 1 week of hourly data
        dropout: float = 0.1,
        forecast_horizon: int = 1
    ):
        """
        Initialize the Spatiotemporal Graph Transformer.
        
        Args:
            input_dim: Input feature dimension (1 for PM2.5 only)
            hidden_dim: Hidden dimension for embeddings and encoders
            num_heads: Number of attention heads for both GAT and Transformer
            num_temporal_layers: Number of Transformer encoder layers
            max_seq_len: Maximum sequence length for positional encodings
            dropout: Dropout rate
            forecast_horizon: Number of hours to forecast (default: 1)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.forecast_horizon = forecast_horizon
        
        # ============== Embedding Layer ==============
        # Project input PM2.5 values to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Learnable positional encodings for temporal sequence
        self.positional_encoding = LearnablePositionalEncoding(
            max_len=max_seq_len,
            d_model=hidden_dim
        )
        
        # ============== Spatial Encoder (GAT) ==============
        # GATv2 for dynamic spatial attention between sensors
        self.spatial_encoder = SpatialEncoder(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,  # Output per head
            num_heads=num_heads,
            dropout=dropout,
            concat=True  # Concatenate heads -> output = hidden_dim
        )
        
        # Layer norm after spatial encoding
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        
        # ============== Temporal Encoder (Transformer) ==============
        self.temporal_encoder = TemporalEncoder(
            d_model=hidden_dim,
            num_heads=num_heads,
            num_layers=num_temporal_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )
        
        # ============== Decoder Head ==============
        # MLP decoder for forecasting
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_horizon)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        return_attention_weights: bool = True
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]], Optional[Tensor]]:
        """
        Forward pass through the Spatiotemporal Graph Transformer.
        
        Args:
            x: Input PM2.5 time series of shape (num_nodes, seq_len, input_dim)
            edge_index: Graph connectivity of shape (2, num_edges)
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Tuple of:
                - forecast: Predicted PM2.5 values of shape (num_nodes, forecast_horizon)
                - spatial_weights: Tuple of (edge_index, attention_weights) from GAT
                  where attention_weights has shape (num_edges, num_heads)
                - temporal_weights: Attention weights from Transformer
                  of shape (num_layers, num_nodes, num_heads, seq_len, seq_len)
        """
        num_nodes, seq_len, _ = x.shape
        
        # ============== Embedding ==============
        # Project input to hidden dimension: (N, T, 1) -> (N, T, D)
        x = self.input_projection(x)
        
        # Add learnable positional encodings
        x = self.positional_encoding(x)
        
        # ============== Spatial Encoding ==============
        # Process spatial dependencies for each time step
        # Reshape: (N, T, D) -> (N*T, D) for GAT processing
        x_spatial = x.reshape(num_nodes * seq_len, self.hidden_dim)
        
        # Create temporal copies of edge_index for batch processing
        # Each time step has its own copy of the graph
        edge_index_expanded = self._expand_edge_index(edge_index, num_nodes, seq_len)
        
        # Apply spatial attention
        x_spatial, spatial_weights = self.spatial_encoder(
            x_spatial, edge_index_expanded, 
            return_attention_weights=return_attention_weights
        )
        
        # Reshape back: (N*T, D) -> (N, T, D)
        x = x_spatial.reshape(num_nodes, seq_len, self.hidden_dim)
        x = self.spatial_norm(x)
        
        # Average the spatial weights across time steps for interpretability
        if spatial_weights is not None:
            edge_idx, attn_weights = spatial_weights
            # Reshape attention weights to get per-original-edge weights
            # attn_weights shape: (num_edges * seq_len, num_heads)
            num_original_edges = edge_index.size(1)
            if attn_weights.size(0) > 0:
                attn_reshaped = attn_weights.reshape(seq_len, -1, self.num_heads)
                # Average across time: (seq_len, num_edges+self_loops, heads) -> (num_edges+self_loops, heads)
                attn_avg = attn_reshaped.mean(dim=0)
                spatial_weights = (edge_index, attn_avg[:num_original_edges + num_nodes])
            
        # ============== Temporal Encoding ==============
        # Reshape for Transformer: (N, T, D) -> (T, N, D)
        x = x.permute(1, 0, 2)
        
        # Apply temporal attention
        x, temporal_weights = self.temporal_encoder(
            x, return_attention_weights=return_attention_weights
        )
        
        # Reshape back: (T, N, D) -> (N, T, D)
        x = x.permute(1, 0, 2)
        
        # ============== Decoding ==============
        # Use the last time step's representation for forecasting
        x_last = x[:, -1, :]  # (N, D)
        
        # Generate forecast
        forecast = self.decoder(x_last)  # (N, forecast_horizon)
        
        return forecast, spatial_weights, temporal_weights
    
    def _expand_edge_index(
        self,
        edge_index: Tensor,
        num_nodes: int,
        seq_len: int
    ) -> Tensor:
        """
        Expand edge_index to create a batch of graphs for each time step.
        
        This creates seq_len copies of the graph, where nodes in each
        copy are offset by (t * num_nodes) for time step t.
        
        Args:
            edge_index: Original graph connectivity (2, num_edges)
            num_nodes: Number of nodes in the graph
            seq_len: Number of time steps
            
        Returns:
            Expanded edge_index of shape (2, num_edges * seq_len)
        """
        edge_indices = []
        for t in range(seq_len):
            offset = t * num_nodes
            edge_indices.append(edge_index + offset)
        
        return torch.cat(edge_indices, dim=1)
    
    def get_attention_interpretation(
        self,
        spatial_weights: Tuple[Tensor, Tensor],
        temporal_weights: Tensor,
        node_names: Optional[list] = None
    ) -> dict:
        """
        Convert attention weights to interpretable format.
        
        Args:
            spatial_weights: Tuple of (edge_index, attention_weights)
            temporal_weights: Temporal attention tensor
            node_names: Optional list of sensor names
            
        Returns:
            Dictionary with interpretation-friendly attention data
        """
        interpretation = {}
        
        if spatial_weights is not None:
            edge_index, attn = spatial_weights
            # Average attention across heads
            attn_avg = attn.mean(dim=-1)  # (num_edges,)
            
            interpretation['spatial'] = {
                'edge_index': edge_index.cpu().numpy(),
                'attention_weights': attn_avg.cpu().detach().numpy(),
                'description': 'Higher values indicate stronger spatial influence'
            }
            
        if temporal_weights is not None:
            # Use the last layer's attention for interpretation
            # Shape: (num_layers, num_nodes, num_heads, seq_len, seq_len)
            last_layer_attn = temporal_weights[-1]  # (N, heads, T, T)
            # Average across heads
            attn_avg = last_layer_attn.mean(dim=1)  # (N, T, T)
            # Get attention to the last time step (what influenced the forecast)
            forecast_attention = attn_avg[:, -1, :]  # (N, T) - attention from last step to all others
            
            interpretation['temporal'] = {
                'attention_matrix': attn_avg.cpu().detach().numpy(),
                'forecast_attention': forecast_attention.cpu().detach().numpy(),
                'description': 'Shows which past hours influenced the forecast'
            }
            
        return interpretation


class STGraphTransformerForecaster(nn.Module):
    """
    Complete forecasting model wrapper with training utilities.
    
    This wraps STGraphTransformer with additional functionality
    for handling masked inputs and computing losses.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_temporal_layers: int = 2,
        max_seq_len: int = 168,
        dropout: float = 0.1,
        mask_token: float = -1.0
    ):
        super().__init__()
        
        self.mask_token = mask_token
        
        self.model = STGraphTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_temporal_layers=num_temporal_layers,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        return_attention_weights: bool = True
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]], Optional[Tensor]]:
        """
        Forward pass with mask handling.
        
        Replaces mask tokens with zeros before processing.
        """
        # Create mask for valid (non-masked) values
        mask = x != self.mask_token
        
        # Replace mask tokens with zeros for processing
        x_processed = x.clone()
        x_processed[~mask] = 0.0
        
        return self.model(x_processed, edge_index, return_attention_weights)
    
    def compute_loss(
        self,
        predictions: Tensor,
        targets: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute masked MSE loss.
        
        Args:
            predictions: Model predictions (num_nodes, 1)
            targets: Ground truth values (num_nodes, 1)
            mask: Optional mask for valid targets
            
        Returns:
            Scalar loss value
        """
        if mask is None:
            mask = targets != self.mask_token
            
        if mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)
            
        loss = F.mse_loss(predictions[mask], targets[mask])
        return loss


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing STGraphTransformer")
    print("=" * 60)
    
    # Create sample data
    num_nodes = 10
    seq_len = 12
    input_dim = 1
    
    # Random PM2.5 values
    x = torch.randn(num_nodes, seq_len, input_dim) * 50 + 100  # Simulate PM2.5 values
    
    # Create a simple graph (ring topology for testing)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    ], dtype=torch.long)
    
    # Add reverse edges for undirected graph
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Number of edges: {edge_index.shape[1]}")
    
    # Create model
    model = STGraphTransformer(
        input_dim=input_dim,
        hidden_dim=64,
        num_heads=4,
        num_temporal_layers=2
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        forecast, spatial_weights, temporal_weights = model(x, edge_index)
    
    print(f"\nForecast shape: {forecast.shape}")
    
    if spatial_weights is not None:
        edge_idx, attn = spatial_weights
        print(f"Spatial attention shape: {attn.shape}")
        print(f"  - Shows attention from each edge (which neighbors matter)")
        
    if temporal_weights is not None:
        print(f"Temporal attention shape: {temporal_weights.shape}")
        print(f"  - Shape: (layers, nodes, heads, seq_len, seq_len)")
    
    # Get interpretable attention
    interpretation = model.get_attention_interpretation(spatial_weights, temporal_weights)
    
    print("\n" + "=" * 60)
    print("Interpretation Summary")
    print("=" * 60)
    
    if 'spatial' in interpretation:
        print(f"\nSpatial Attention (neighbor influence):")
        print(f"  Edge index shape: {interpretation['spatial']['edge_index'].shape}")
        print(f"  Attention weights shape: {interpretation['spatial']['attention_weights'].shape}")
        
    if 'temporal' in interpretation:
        print(f"\nTemporal Attention (past hour influence):")
        print(f"  Forecast attention shape: {interpretation['temporal']['forecast_attention'].shape}")
        print(f"  - Shows which of the {seq_len} past hours influenced the forecast")
    
    print("\n✓ Model test passed!")
