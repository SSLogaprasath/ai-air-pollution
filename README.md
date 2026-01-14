# 🌫️ Air Quality Forecasting with STGraphTransformer

A deep learning system for predicting PM2.5 concentrations across Indian cities using a hybrid **Graph Attention Network + Transformer** architecture.

## 🏗️ Architecture

**STGraphTransformer** - A spatiotemporal model that combines:
- **GATv2Conv** (4 heads): Learns sensor-to-sensor spatial relationships
- **Transformer** (2 layers, 4 heads): Captures temporal patterns over 14 days
- **Explainable AI**: Returns attention weights showing which sensors/days influenced predictions

```
Input (14 days × N sensors)
    ↓
Embedding → Positional Encoding
    ↓
Spatial Encoder (GATv2) → learns neighbor influence
    ↓
Temporal Encoder (Transformer) → learns time patterns
    ↓
Decoder (MLP) → Next-day PM2.5 prediction
```

**Model Stats**: 115,969 parameters

## 📊 Performance

Tested against live OpenAQ sensor data (January 2026):

| Metric | Value |
|--------|-------|
| MAE | 42.16 µg/m³ |
| RMSE | 78.09 µg/m³ |
| MAPE | 27.6% |
| Correlation | 0.941 |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torch_geometric pytorch-lightning fastapi uvicorn openaq numpy pandas
```

### 2. Train the Model

```bash
# Download "Time Series Air Quality Data of India (2010-2023)" from Kaggle
# Place CSV files in data/2010-2023/

python train_real.py
```

### 3. Run the API

```bash
uvicorn app:app --reload
```

Open http://localhost:8000 for the interactive map UI.

## 📁 Project Structure

| File | Description |
|------|-------------|
| `model.py` | STGraphTransformer architecture |
| `train.py` | PyTorch Lightning training module |
| `train_real.py` | Training script for 2020-2023 data |
| `data_ingestion.py` | OpenAQ API integration |
| `app.py` | FastAPI backend with health impact |
| `static/index.html` | Interactive Leaflet map UI |

## 🔑 Configuration

Set your OpenAQ API key in `data_ingestion.py`:
```python
API_KEY = "your_api_key_here"
```

Get a free API key at https://openaq.org

## 🌍 Supported Cities

The model is city-agnostic. Add new cities by editing `CITY_BBOXES` in `data_ingestion.py`:

```python
CITY_BBOXES = {
    "mumbai": (19.0760, 72.8777),
    "delhi": (28.6139, 77.2090),
    "kolkata": (22.5726, 88.3639),  # Add new cities here
    # ...
}
```

No retraining needed - the model generalizes to any city with OpenAQ sensors.

## 🏥 Health Impact Assessment

The API translates PM2.5 predictions to health recommendations:

| AQI Level | PM2.5 Range | Recommendation |
|-----------|-------------|----------------|
| Good | 0-30 | Air quality is satisfactory |
| Moderate | 31-60 | Acceptable for most people |
| Unhealthy (Sensitive) | 61-90 | Sensitive groups should limit outdoor activity |
| Unhealthy | 91-120 | Everyone should reduce prolonged outdoor exertion |
| Very Unhealthy | 121-250 | Avoid outdoor activities |
| Hazardous | 250+ | Emergency conditions - stay indoors |

## 📜 License

MIT License
