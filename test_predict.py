"""Quick test to verify the model can make predictions."""

import torch
from data_ingestion import fetch_city_graph, fetch_historical_signals
from model import STGraphTransformer

def test_prediction():
    print("=" * 50)
    print("Testing Model Prediction Capability")
    print("=" * 50)
    
    # Step 1: Fetch live graph
    print("\n1. Fetching Mumbai sensor graph...")
    graph = fetch_city_graph('mumbai', radius_km=15)
    print(f"   ✓ Found {len(graph.location_ids)} sensors")
    print(f"   ✓ Created {graph.edge_index.shape[1]} edges")
    
    # Step 2: Fetch historical signals
    print("\n2. Fetching last 24 hours of PM2.5 data...")
    signals = fetch_historical_signals(graph.location_ids, hours=24)
    print(f"   ✓ Signal shape: {signals.shape}")
    
    # Check data coverage
    valid = (signals != -1.0).float().mean().item() * 100
    print(f"   ✓ Data coverage: {valid:.1f}%")
    
    # Step 3: Create model
    print("\n3. Creating STGraphTransformer model...")
    model = STGraphTransformer(
        input_dim=1, 
        hidden_dim=64, 
        num_heads=4, 
        num_temporal_layers=2, 
        max_seq_len=24
    )
    model.eval()
    print(f"   ✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Step 4: Run prediction
    print("\n4. Running inference...")
    with torch.no_grad():
        predictions, spatial_attn, temporal_attn = model(
            signals, 
            graph.edge_index, 
            return_attention_weights=True
        )
    
    print(f"   ✓ Output shape: {predictions.shape}")
    print(f"   ✓ Spatial attention: {type(spatial_attn)}")
    print(f"   ✓ Temporal attention shape: {temporal_attn.shape}")
    
    # Step 5: Display results
    print("\n5. Predicted PM2.5 values (µg/m³):")
    preds = predictions.flatten().numpy()
    for i, (loc_id, pred) in enumerate(zip(graph.location_ids, preds)):
        print(f"   Station {loc_id}: {pred:.2f}")
    
    print("\n" + "=" * 50)
    print("✓ SUCCESS - Model can make predictions!")
    print("=" * 50)
    print("\nNote: These are random predictions from an untrained model.")
    print("Train the model first for meaningful forecasts.")

if __name__ == "__main__":
    test_prediction()
