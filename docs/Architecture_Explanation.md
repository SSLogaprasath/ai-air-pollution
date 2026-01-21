# Air Quality Forecasting Model - Complete Explanation

## Table of Contents
1. [Architecture: STGraphTransformer](#1-architecture-stgraphtransformer)
2. [Training Process](#2-training-process)
3. [How It Works (Inference)](#3-how-it-works-inference)
4. [Key Innovations](#4-key-innovations)
5. [Results Summary](#5-results-summary)

---

## 1. Architecture: STGraphTransformer

The model is a **hybrid architecture** that captures two types of dependencies in air quality data:

### Why Two Types of Dependencies?

1. **Spatial**: Pollution at one sensor affects nearby sensors (wind carries pollutants)
2. **Temporal**: Today's pollution depends on past days (weather patterns, weekly cycles)

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                                   │
│   Shape: (N sensors × 14 days × 1 feature)                          │
│   Example: 13 sensors in Delhi, 14 days of PM2.5 readings           │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    INPUT EMBEDDING                                   │
│   Linear(1 → 64)                                                     │
│   Expands PM2.5 value into 64-dimensional representation            │
│   Shape: (N × 14 × 64)                                              │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 POSITIONAL ENCODING                                  │
│   Learnable embeddings for each day position (day 1, day 2, ...)    │
│   Tells model "this is 3 days ago" vs "this is yesterday"           │
│   Shape: (N × 14 × 64)                                              │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  SPATIAL ENCODER (GATv2)                             │
│                                                                      │
│   Graph Attention Network v2                                         │
│   • 4 attention heads                                                │
│   • Each sensor "attends" to its neighbors                           │
│   • Learns: "How much does Sensor A influence Sensor B?"             │
│                                                                      │
│   Example attention weights:                                         │
│     Industrial_Area → Residential: 0.35 (high - upwind)              │
│     Park_Sensor → Residential: 0.12 (low - cleaner area)             │
│                                                                      │
│   Output: Spatially-aware features for each sensor                   │
│   Shape: (N × 14 × 64)                                              │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                TEMPORAL ENCODER (Transformer)                        │
│                                                                      │
│   2 Transformer Layers, each with:                                   │
│   • Multi-head self-attention (4 heads)                              │
│   • Feedforward network (64 → 256 → 64)                              │
│   • Layer normalization + residual connections                       │
│                                                                      │
│   Learns: "Which past days matter for tomorrow's prediction?"        │
│                                                                      │
│   Example attention weights:                                         │
│     Yesterday (day 14): 0.25 (most important)                        │
│     2 days ago: 0.18                                                 │
│     7 days ago: 0.12 (weekly pattern)                                │
│   Shape: (N × 14 × 64)                                              │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       DECODER (MLP)                                  │
│   Takes the LAST time step (day 14) representation                  │
│   Linear(64 → 32) → ReLU → Linear(32 → 1)                           │
│   Output: Next-day PM2.5 prediction for each sensor                 │
│   Shape: (N × 1)                                                    │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          OUTPUT                                      │
│   • Predicted PM2.5 for each sensor (N values)                      │
│   • Spatial attention weights (which sensors influenced which)      │
│   • Temporal attention weights (which days mattered most)           │
└─────────────────────────────────────────────────────────────────────┘
```

### Model Parameters Summary

| Component | Details |
|-----------|---------|
| Total Parameters | 115,969 |
| Hidden Dimension | 64 |
| Attention Heads | 4 |
| Temporal Layers | 2 |
| Input Window | 14 days |
| Forecast Horizon | 1 day |

---

## 2. Training Process

### Data Preparation

```
Raw Data: CSV files from 454 stations across India (2010-2023)
    │
    ▼
Filter: Keep only 2020-2023 (modern pollution patterns)
    │
    ▼
Group by State: Each state = one graph (sensors are nodes)
    │
    ▼
Sliding Window: Create 14-day windows with next day as target
    │
    ▼
Build Graph: Connect sensors within 50km of each other
    │
    ▼
Dataset: 4,292 training samples from 21 states
```

### Graph Construction

```
Sensors in Maharashtra:
    
    [Mumbai_Andheri] ←─────→ [Mumbai_Bandra]     (5km apart: connected)
          │                        │
          │                        │
          ▼                        ▼
    [Mumbai_Worli] ←──────→ [Mumbai_Colaba]     (8km apart: connected)
          
          ✗ Not connected to [Pune_Station] (150km away)

Edge Rule: Connect if distance < 50km
```

### Loss Function: Masked MSE

```python
# Problem: Not all sensors have data every day
# Solution: Only compute loss on valid (non-masked) values

predictions = [45.2, 67.8, MASK, 34.1, MASK]
targets     = [48.0, 62.5, MASK, 38.0, MASK]

# Only compute MSE for positions 0, 1, 3 (ignore MASK positions)
loss = MSE([45.2, 67.8, 34.1], [48.0, 62.5, 38.0])
```

### Training Loop (PyTorch Lightning)

```
For each epoch:
    │
    ├─► Training batches (3,004 samples)
    │   • Forward pass → predictions
    │   • Compute masked MSE loss
    │   • Backpropagate gradients
    │   • Update weights (AdamW optimizer, lr=0.001)
    │
    ├─► Validation (643 samples)
    │   • Compute val_loss (no gradient updates)
    │   • Early stopping if no improvement for 10 epochs
    │
    └─► Save best checkpoint based on val_loss
    
Total: ~30 epochs, ~30 minutes on RTX 4050
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.01 |
| Scheduler | ReduceLROnPlateau |
| Batch Size | 32 |
| Early Stopping | 10 epochs patience |
| Max Epochs | 100 |

---

## 3. How It Works (Inference)

### Step-by-Step Prediction Flow

**Step 1: FETCH SENSOR LOCATIONS (OpenAQ API)**
```
GET /locations?city=delhi&radius=15km
Returns: 13 sensors with lat/lon coordinates
```

**Step 2: BUILD SPATIAL GRAPH**
```
For each pair of sensors:
  if distance < threshold: add edge
Result: edge_index tensor [[0,0,1,1,2,...], [1,2,0,2,0,...]]
```

**Step 3: FETCH HISTORICAL DATA (OpenAQ API)**
```
GET /measurements?location_id=XXX&hours=240
Returns: ~10 days of hourly PM2.5 readings per sensor
Aggregate to daily averages → 14-day window
```

**Step 4: HANDLE MISSING DATA**
```
If sensor has no data for a day: fill with MASK_TOKEN (-999)
Model is trained to ignore MASK positions
```

**Step 5: RUN MODEL**
```
Input: x (13 sensors × 14 days × 1), edge_index
Output: predictions (13 sensors × 1), attention weights
```

**Step 6: GENERATE EXPLANATION**
```
From spatial attention: "Sensor A strongly influences B"
From temporal attention: "Yesterday was most important"
```

**Step 7: HEALTH IMPACT ASSESSMENT**
```
PM2.5 → AQI category → Health recommendations
Example: 150 µg/m³ → "Very Unhealthy" → "Avoid outdoors"
```

**Step 8: RETURN TO UI**
```
• Map markers colored by AQI
• Prediction values
• Model explanation
• Health warnings
```

### Example: Delhi Prediction

**Input: Last 14 days of PM2.5 for each sensor**
```
Anand Vihar:  [180, 195, 210, 185, 220, 245, 230, 205, 190, 200, 215, 225, 240, 235]
ITO:          [165, 175, 190, 170, 205, 220, 210, 185, 175, 185, 195, 205, 220, 215]
Dwarka:       [145, 155, 170, 150, 180, 195, 185, 165, 155, 165, 175, 185, 195, 190]
                ↑                                                              ↑
            Day 1 (oldest)                                              Day 14 (today)
```

**Output: Tomorrow's PM2.5 prediction**
```
┌─────────────────────────────────────────────────────────┐
│  Sensor           Today (actual)    Tomorrow (predicted) │
├─────────────────────────────────────────────────────────┤
│  Anand Vihar      235 µg/m³    →    248 µg/m³           │
│  ITO              215 µg/m³    →    228 µg/m³           │
│  Dwarka           190 µg/m³    →    201 µg/m³           │
│  Rohini           205 µg/m³    →    218 µg/m³           │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Key Innovations

### Why GATv2 (not GAT)?

```
GAT (v1):  attention(i,j) = f(W·hᵢ || W·hⱼ)  # Static ranking
GATv2:     attention(i,j) = f(W·[hᵢ || hⱼ])  # Dynamic - depends on BOTH nodes

Result: GATv2 can learn directional relationships
  • "Industrial area affects residential" ≠ "Residential affects industrial"
  • Better for pollution flow modeling (upwind vs downwind)
```

### Why Transformer (not LSTM)?

```
LSTM: Sequential processing, forgets long-term patterns
  Day 1 → Day 2 → Day 3 → ... → Day 14 (information degrades)

Transformer: Direct attention to ANY past day
  Day 14 can directly attend to Day 1, Day 7, etc.
  Better for: weekly patterns, sudden events
```

### Masked Training for Sparse Data

```
Real-world problem: Sensors have gaps (maintenance, failures)

Solution: MASK_TOKEN = -999
  • Input: [45, MASK, 52, MASK, MASK, 48, ...]
  • Model learns to ignore MASK positions
  • Loss computed only on valid predictions
```

---

## 5. Results Summary

### Overall Performance

| Metric | Value | Meaning |
|--------|-------|---------|
| MAE | 42.16 µg/m³ | Average prediction error |
| RMSE | 78.09 µg/m³ | Penalizes large errors more |
| MAPE | 27.6% | Percentage error |
| Correlation | 0.941 | Predictions track actual trends well |

### Per-City Performance

| City | Performance | Notes |
|------|-------------|-------|
| Mumbai | ⭐ Excellent | Errors: -5.7 to +26.0 µg/m³ |
| Bangalore | ⭐ Excellent | Errors: -13.3 to +0.5 µg/m³ |
| Chennai | ✅ Good | Most errors <20 µg/m³ |
| Delhi | ⚠️ Challenging | Winter pollution 300-400+ is hard to predict |

### Health Impact Categories

| AQI Level | PM2.5 Range | Recommendation |
|-----------|-------------|----------------|
| Good | 0-30 µg/m³ | Air quality is satisfactory |
| Moderate | 31-60 µg/m³ | Acceptable for most people |
| Unhealthy (Sensitive) | 61-90 µg/m³ | Sensitive groups limit outdoor activity |
| Unhealthy | 91-120 µg/m³ | Everyone reduce outdoor exertion |
| Very Unhealthy | 121-250 µg/m³ | Avoid outdoor activities |
| Hazardous | 250+ µg/m³ | Emergency - stay indoors |

---

## Project Files

| File | Description |
|------|-------------|
| model.py | STGraphTransformer architecture |
| train.py | PyTorch Lightning training module |
| train_real.py | Training script for 2020-2023 data |
| data_ingestion.py | OpenAQ API integration |
| app.py | FastAPI backend |
| static/index.html | Interactive Leaflet map UI |
| checkpoints/pollution-forecaster-best-v2.ckpt | Trained model weights |

---

*Document generated: January 2026*
*Model Version: v2 (trained on 2020-2023 data)*
