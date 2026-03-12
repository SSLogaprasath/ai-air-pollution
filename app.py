"""
Air Quality Forecasting API.

FastAPI backend for real-time PM2.5 predictions using
the STGraphTransformer model with interpretable outputs.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import csv
import io
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from data_ingestion import (
    fetch_city_graph,
    fetch_historical_signals,
    CITY_BBOXES,
    MASK_TOKEN
)
from indian_cities import search_cities, get_city_bbox, get_city_center
from model import STGraphTransformer
from train import PollutionForecaster
from auth import (
    register_user,
    login_user,
    get_current_user,
    get_optional_user,
)
from database import (
    save_location,
    get_saved_locations,
    delete_saved_location,
    upsert_alert,
    get_alerts,
    delete_alert,
    upsert_health_profile,
    get_health_profile,
    add_forecast_history,
    get_forecast_history,
)
from alert_service import alert_scheduler


# ============================================================
# Auth Pydantic Models
# ============================================================

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=30)
    email: str = Field(..., max_length=120)
    password: str = Field(..., min_length=6)

class LoginRequest(BaseModel):
    email: str
    password: str

class SaveLocationRequest(BaseModel):
    city_name: str
    state: str = ""
    lat: float
    lon: float

class AlertRequest(BaseModel):
    city_name: str
    threshold_pm25: float = Field(default=55.4, ge=1, le=500)
    enabled: bool = True

class HealthProfileRequest(BaseModel):
    age_group: str = Field(default="adult")
    conditions: List[str] = Field(default_factory=list)
    outdoor_hours: float = Field(default=2.0, ge=0, le=24)


# ============================================================
# Configuration
# ============================================================

MODEL_CHECKPOINT_PATH = "checkpoints/pollution-forecaster-best-v2.ckpt"
INPUT_WINDOW = 14  # Days of historical data (matches training)
HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_TEMPORAL_LAYERS = 2

# Global model instance (loaded once at startup)
model: Optional[PollutionForecaster] = None
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Pydantic Models for API
# ============================================================

class SensorPrediction(BaseModel):
    """Prediction for a single sensor."""
    location_id: int = Field(..., description="OpenAQ location ID")
    name: str = Field(..., description="Station name")
    latitude: float = Field(..., description="Sensor latitude")
    longitude: float = Field(..., description="Sensor longitude")
    predicted_pm25: float = Field(..., description="Predicted PM2.5 value (µg/m³)")
    data_coverage: float = Field(..., description="Percentage of valid input data")


class SpatialInfluence(BaseModel):
    """Information about spatial influence from neighboring sensors."""
    source_id: int = Field(..., description="Source sensor ID")
    target_id: int = Field(..., description="Target sensor ID")
    attention_weight: float = Field(..., description="Attention weight (0-1)")


class TemporalInfluence(BaseModel):
    """Information about temporal influence from past hours."""
    hour_offset: int = Field(..., description="Hours ago (0 = most recent)")
    attention_weight: float = Field(..., description="Attention weight (0-1)")


class HealthImpact(BaseModel):
    """Health impact assessment based on PM2.5 level."""
    aqi_category: str = Field(..., description="AQI category (Good, Moderate, Unhealthy, etc.)")
    aqi_color: str = Field(..., description="Color code for the AQI level")
    health_warning: str = Field(..., description="Health warning message")
    sensitive_groups: List[str] = Field(..., description="Groups most at risk")
    recommended_exposure: str = Field(..., description="Maximum recommended outdoor exposure time")
    health_effects: List[str] = Field(..., description="Potential health effects")
    precautions: List[str] = Field(..., description="Recommended precautions")
    long_term_effects: List[str] = Field(..., description="Effects of prolonged exposure")


class PredictionExplanation(BaseModel):
    """Explanation of what influenced the prediction."""
    summary: str = Field(..., description="Human-readable explanation")
    dominant_spatial_influence: Optional[str] = Field(None, description="Most influential neighbor")
    dominant_temporal_influence: Optional[str] = Field(None, description="Most influential time period")
    top_spatial_influences: List[Dict[str, Any]] = Field(default_factory=list)
    top_temporal_influences: List[Dict[str, Any]] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    """Complete prediction response."""
    city: str = Field(..., description="City name")
    num_sensors: int = Field(..., description="Number of sensors in the graph")
    num_edges: int = Field(..., description="Number of connections between sensors")
    forecast_pm25: List[SensorPrediction] = Field(..., description="Predictions for each sensor")
    city_average_pm25: float = Field(..., description="Average predicted PM2.5 across city")
    health_impact: HealthImpact = Field(..., description="Health impact assessment")
    explanation: PredictionExplanation = Field(..., description="Interpretation of predictions")
    model_version: str = Field(default="1.0.0", description="Model version")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    available_cities: List[str] = Field(..., description="Cities available for prediction")
    device: str = Field(..., description="Compute device (cpu/cuda)")


# ============================================================
# Model Loading
# ============================================================

def get_health_impact(pm25: float) -> HealthImpact:
    """
    Get health impact assessment based on PM2.5 level.
    Based on US EPA and WHO guidelines.
    
    Args:
        pm25: PM2.5 concentration in µg/m³
        
    Returns:
        HealthImpact with detailed health information
    """
    if pm25 <= 12:
        return HealthImpact(
            aqi_category="Good",
            aqi_color="#00e676",
            health_warning="Air quality is satisfactory with little or no risk.",
            sensitive_groups=["None - safe for all groups"],
            recommended_exposure="Unlimited outdoor activity is safe",
            health_effects=[
                "No health effects expected",
                "Ideal conditions for outdoor exercise"
            ],
            precautions=[
                "No precautions needed",
                "Enjoy outdoor activities"
            ],
            long_term_effects=[
                "No long-term effects at this level",
                "Continued exposure at this level is safe"
            ]
        )
    elif pm25 <= 35.4:
        return HealthImpact(
            aqi_category="Moderate",
            aqi_color="#ffeb3b",
            health_warning="Air quality is acceptable. Some pollutants may pose a moderate health concern for a very small number of people.",
            sensitive_groups=[
                "People with respiratory diseases (asthma, COPD)",
                "Children and elderly",
                "People with heart disease"
            ],
            recommended_exposure="8-12 hours outdoor exposure is generally safe",
            health_effects=[
                "Possible mild respiratory irritation in sensitive individuals",
                "May cause slight discomfort for people with pre-existing conditions",
                "Most people will not be affected"
            ],
            precautions=[
                "Sensitive individuals should monitor symptoms",
                "Consider reducing prolonged outdoor exertion if you experience symptoms",
                "Keep windows closed if you're sensitive"
            ],
            long_term_effects=[
                "Minimal risk with occasional exposure",
                "Chronic exposure may slightly increase respiratory risks"
            ]
        )
    elif pm25 <= 55.4:
        return HealthImpact(
            aqi_category="Unhealthy for Sensitive Groups",
            aqi_color="#ff9800",
            health_warning="Members of sensitive groups may experience health effects. General public is less likely to be affected.",
            sensitive_groups=[
                "Children and teenagers",
                "Elderly (65+)",
                "People with asthma or respiratory diseases",
                "People with heart or lung disease",
                "Pregnant women",
                "Outdoor workers"
            ],
            recommended_exposure="Limit outdoor exposure to 2-4 hours for sensitive groups",
            health_effects=[
                "Increased respiratory symptoms in sensitive groups",
                "Aggravation of asthma and lung disease",
                "Possible breathing discomfort",
                "Increased coughing and throat irritation",
                "Reduced lung function in children"
            ],
            precautions=[
                "Sensitive groups should reduce prolonged outdoor exertion",
                "Consider wearing N95 masks outdoors",
                "Keep windows and doors closed",
                "Use air purifiers indoors",
                "Avoid outdoor exercise, especially near traffic"
            ],
            long_term_effects=[
                "Increased risk of respiratory infections",
                "Potential worsening of chronic lung conditions",
                "Children may experience reduced lung development",
                "Increased cardiovascular stress"
            ]
        )
    elif pm25 <= 150.4:
        return HealthImpact(
            aqi_category="Unhealthy",
            aqi_color="#f44336",
            health_warning="Everyone may begin to experience health effects. Sensitive groups may experience more serious effects.",
            sensitive_groups=[
                "Everyone is at risk",
                "Higher risk: Children, elderly, pregnant women",
                "Higher risk: People with heart/lung disease",
                "Higher risk: Outdoor workers"
            ],
            recommended_exposure="Maximum 1-2 hours outdoors. Avoid outdoor exercise.",
            health_effects=[
                "Significant aggravation of heart or lung disease",
                "Increased respiratory symptoms in general population",
                "Decreased exercise tolerance",
                "Headaches and fatigue",
                "Irritation of eyes, nose, and throat",
                "Increased risk of heart attacks in vulnerable individuals"
            ],
            precautions=[
                "Everyone should reduce outdoor exposure",
                "Wear N95/P100 masks if going outside",
                "Avoid all outdoor exercise",
                "Keep all windows and doors sealed",
                "Run HEPA air purifiers continuously",
                "Consider staying home if possible",
                "Sensitive groups should remain indoors"
            ],
            long_term_effects=[
                "Accelerated lung aging",
                "Increased risk of lung cancer with chronic exposure",
                "Higher rates of cardiovascular disease",
                "Potential neurological effects",
                "Reduced life expectancy with prolonged exposure"
            ]
        )
    elif pm25 <= 250.4:
        return HealthImpact(
            aqi_category="Very Unhealthy",
            aqi_color="#9c27b0",
            health_warning="Health alert: Everyone may experience serious health effects.",
            sensitive_groups=[
                "EVERYONE is at serious risk",
                "Extreme risk: Elderly and children",
                "Extreme risk: Anyone with respiratory conditions",
                "Extreme risk: People with cardiovascular disease"
            ],
            recommended_exposure="Avoid ANY outdoor exposure. Stay indoors with air filtration.",
            health_effects=[
                "Serious aggravation of heart and lung disease",
                "Significant respiratory effects in general population",
                "Increased emergency room visits",
                "Severe breathing difficulties",
                "Chest pain and discomfort",
                "Irregular heartbeat",
                "Dizziness and nausea"
            ],
            precautions=[
                "AVOID all outdoor activities",
                "Remain indoors with air purification",
                "Seal windows and doors with tape if needed",
                "Wear N95 masks even indoors if no air filtration",
                "Have emergency medications ready (inhalers, etc.)",
                "Seek medical attention if experiencing symptoms",
                "Consider temporary relocation if possible"
            ],
            long_term_effects=[
                "Significant reduction in life expectancy",
                "High risk of chronic respiratory disease",
                "Increased cancer risk",
                "Permanent lung damage possible",
                "Cardiovascular damage",
                "Potential cognitive decline"
            ]
        )
    else:
        return HealthImpact(
            aqi_category="Hazardous",
            aqi_color="#7b1fa2",
            health_warning="HEALTH EMERGENCY: Entire population is affected. Serious health effects on everyone.",
            sensitive_groups=[
                "ENTIRE POPULATION at emergency risk",
                "Life-threatening for elderly and children",
                "Critical risk for anyone with health conditions"
            ],
            recommended_exposure="DO NOT go outside. This is a health emergency.",
            health_effects=[
                "Emergency conditions: Serious illness and premature death",
                "Severe respiratory distress",
                "Heart attacks and strokes",
                "Hospital admissions surge",
                "Breathing may be difficult even indoors",
                "Immediate symptoms: burning eyes, severe coughing",
                "Risk of death for vulnerable individuals"
            ],
            precautions=[
                "STAY INDOORS - this is an emergency",
                "Run all available air purifiers on maximum",
                "Create a clean room with sealed doors/windows",
                "Wear N95 masks even indoors",
                "Have emergency contacts ready",
                "Call emergency services if experiencing severe symptoms",
                "Consider emergency evacuation to cleaner areas",
                "DO NOT exercise or exert yourself"
            ],
            long_term_effects=[
                "Severe and permanent lung damage",
                "Significantly shortened lifespan",
                "High probability of chronic disease",
                "Permanent cardiovascular damage",
                "Neurological damage possible",
                "Increased mortality even after exposure ends"
            ]
        )


def load_model() -> PollutionForecaster:
    """
    Load the trained model from checkpoint or create a new one.
    
    Returns:
        Loaded PollutionForecaster model
    """
    forecaster = PollutionForecaster(
        input_dim=1,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_temporal_layers=NUM_TEMPORAL_LAYERS,
        max_seq_len=INPUT_WINDOW,
        dropout=0.1
    )
    
    # Try to load checkpoint if it exists
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"Loading model from checkpoint: {MODEL_CHECKPOINT_PATH}")
        checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
        forecaster.load_state_dict(checkpoint['state_dict'])
        print("Model loaded successfully!")
    else:
        print(f"No checkpoint found at {MODEL_CHECKPOINT_PATH}")
        print("Using untrained model (for demo purposes)")
    
    forecaster.to(device)
    forecaster.eval()
    
    return forecaster


# ============================================================
# Interpretation Utilities
# ============================================================

def generate_explanation(
    spatial_weights: Optional[tuple],
    temporal_weights: Optional[torch.Tensor],
    location_ids: List[int],
    edge_index: torch.Tensor,
    num_heads: int = NUM_HEADS,
    location_names: Optional[List[str]] = None
) -> PredictionExplanation:
    """
    Generate human-readable explanation from attention weights.
    
    Args:
        spatial_weights: Tuple of (edge_index, attention_weights) from GAT
        temporal_weights: Attention tensor from Transformer
        location_ids: List of sensor location IDs
        edge_index: Graph connectivity
        num_heads: Number of attention heads
        
    Returns:
        PredictionExplanation with interpretable outputs
    """
    explanation_parts = []
    top_spatial = []
    top_temporal = []
    dominant_spatial = None
    dominant_temporal = None
    
    # Process spatial attention
    if spatial_weights is not None:
        edge_idx, attn = spatial_weights
        
        # Average across heads
        if attn.dim() > 1:
            attn_avg = attn.mean(dim=-1)  # (num_edges,)
        else:
            attn_avg = attn
        
        attn_np = attn_avg.cpu().detach().numpy()
        edge_idx_np = edge_idx.cpu().numpy()
        
        # Find top spatial influences (excluding self-loops)
        num_edges = min(len(attn_np), edge_idx_np.shape[1])
        
        edge_weights = []
        for i in range(num_edges):
            if i < edge_idx_np.shape[1]:
                src, tgt = edge_idx_np[0, i], edge_idx_np[1, i]
                if src != tgt and src < len(location_ids) and tgt < len(location_ids):
                    src_name = location_names[src] if location_names and src < len(location_names) else f"Station {location_ids[src]}"
                    tgt_name = location_names[tgt] if location_names and tgt < len(location_names) else f"Station {location_ids[tgt]}"
                    edge_weights.append({
                        'source_id': int(location_ids[src]),
                        'target_id': int(location_ids[tgt]),
                        'source_name': src_name,
                        'target_name': tgt_name,
                        'weight': float(attn_np[i]) if i < len(attn_np) else 0.0
                    })
        
        # Sort by weight and take top 5
        edge_weights.sort(key=lambda x: x['weight'], reverse=True)
        top_spatial = edge_weights[:5]
        
        if top_spatial:
            top = top_spatial[0]
            dominant_spatial = f"{top['source_name']} \u2192 {top['target_name']} (weight: {top['weight']:.3f})"
            explanation_parts.append(
                f"Dominant spatial influence: {top['source_name']} "
                f"strongly affects {top['target_name']}"
            )
    
    # Process temporal attention
    if temporal_weights is not None:
        # temporal_weights shape: (layers, nodes, heads, seq_len, seq_len)
        # Get last layer, average across nodes and heads
        last_layer = temporal_weights[-1]  # (nodes, heads, seq_len, seq_len)
        
        # Average across nodes and heads
        attn_avg = last_layer.mean(dim=(0, 1))  # (seq_len, seq_len)
        
        # Get attention from the last time step to all previous
        forecast_attn = attn_avg[-1, :].cpu().detach().numpy()  # (seq_len,)
        
        # Create temporal influence list
        seq_len = len(forecast_attn)
        temporal_influences = []
        for i, weight in enumerate(forecast_attn):
            hours_ago = seq_len - 1 - i
            temporal_influences.append({
                'hours_ago': hours_ago,
                'weight': float(weight),
                'label': f"{hours_ago}h ago" if hours_ago > 0 else "current"
            })
        
        # Sort by weight
        temporal_influences.sort(key=lambda x: x['weight'], reverse=True)
        top_temporal = temporal_influences[:5]
        
        if top_temporal:
            top = top_temporal[0]
            dominant_temporal = f"{top['label']} (weight: {top['weight']:.3f})"
            explanation_parts.append(
                f"Most influential time: {top['hours_ago']} hours ago "
                f"(attention weight: {top['weight']:.3f})"
            )
    
    # Build summary
    if explanation_parts:
        summary = ". ".join(explanation_parts) + "."
    else:
        summary = "Prediction based on spatiotemporal patterns in sensor network."
    
    return PredictionExplanation(
        summary=summary,
        dominant_spatial_influence=dominant_spatial,
        dominant_temporal_influence=dominant_temporal,
        top_spatial_influences=top_spatial,
        top_temporal_influences=top_temporal
    )


# ============================================================
# FastAPI Application
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    global model
    print("=" * 60)
    print("Starting Air Quality Forecasting API...")
    print("=" * 60)
    
    # Load model on startup
    model = load_model()
    print(f"Model loaded on device: {device}")
    print(f"Available cities: {list(CITY_BBOXES.keys())}")
    print("API ready!")
    print("=" * 60)
    
    # Start background alert checker
    import asyncio
    alert_task = asyncio.create_task(alert_scheduler())
    
    yield
    
    # Cleanup on shutdown
    alert_task.cancel()
    print("Shutting down API...")


app = FastAPI(
    title="Air Quality Forecasting API",
    description="""
    Real-time PM2.5 predictions using a Spatiotemporal Graph Transformer.
    
    This API provides:
    - Live air quality forecasts for Indian cities
    - Interpretable predictions with attention-based explanations
    - Spatial analysis showing which sensors influence each other
    - Temporal analysis showing which past hours matter most
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============================================================
# API Endpoints
# ============================================================

@app.get("/", tags=["UI"])
async def root():
    """
    Serve the main UI page.
    """
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Air Quality Forecasting API", "docs": "/docs"}


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        available_cities=list(CITY_BBOXES.keys()),
        device=str(device)
    )


@app.post("/predict/{city_name}", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    city_name: str,
    radius_km: int = Query(default=15, ge=1, le=100, description="Radius in km for connecting sensors"),
    hours: int = Query(default=INPUT_WINDOW * 24, ge=1, le=720, description="Hours of historical data to use"),
    user: dict | None = Depends(get_optional_user),
):
    """
    Get PM2.5 forecast for a city.
    
    This endpoint:
    1. Fetches the live sensor graph for the requested city
    2. Retrieves the last N days of PM2.5 measurements
    3. Runs the STGraphTransformer model for prediction
    4. Returns forecasts with interpretable explanations
    
    Args:
        city_name: Name of the city (delhi, mumbai, bangalore, chennai, tiruchirappalli)
        radius_km: Maximum distance for connecting sensors (default: 15km)
        hours: Hours of historical data to use (default: 336 = 14 days)
        
    Returns:
        PredictionResponse with forecasts and explanations
    """
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    city_lower = city_name.lower()

    # Look up bounding box: first hardcoded, then dynamic cities DB
    bbox = CITY_BBOXES.get(city_lower)
    if bbox is None:
        bbox = get_city_bbox(city_name)
    if bbox is None:
        raise HTTPException(
            status_code=400,
            detail=f"City '{city_name}' not found. Use /api/cities/search?q=... to find available cities."
        )
    
    try:
        # Step 1: Fetch city graph
        print(f"Fetching graph for {city_name}...")
        graph_data = fetch_city_graph(city=city_lower, radius_km=radius_km, bbox=bbox)
        
        if len(graph_data.location_ids) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No sensors found in {city_name}"
            )
        
        location_ids = graph_data.location_ids
        coords = graph_data.coords
        edge_index = graph_data.edge_index
        location_names = graph_data.location_names
        
        print(f"Found {len(location_ids)} sensors with {edge_index.shape[1]} edges")
        
        # Step 2: Fetch historical signals (hourly data)
        # We need 14 days of data, fetch hourly and aggregate to daily
        hours_needed = INPUT_WINDOW * 24  # 14 days * 24 hours = 336 hours
        # But API limits to 1000, so fetch what we can (about 10 days max)
        hours_to_fetch = min(hours_needed, 240)  # ~10 days worth
        
        print(f"Fetching last {hours_to_fetch} hours of data...")
        signals = fetch_historical_signals(location_ids, hours=hours_to_fetch)
        
        # Check data availability for hourly data
        valid_mask_hourly = signals != MASK_TOKEN
        hourly_coverage = valid_mask_hourly.float().mean().item() * 100
        print(f"Hourly data coverage: {hourly_coverage:.1f}%")
        
        # Step 3: Aggregate hourly data to daily averages
        # signals shape: (num_nodes, hours, 1) -> (num_nodes, days, 1)
        num_nodes = signals.shape[0]
        available_hours = signals.shape[1]
        num_days = min(available_hours // 24, INPUT_WINDOW)
        
        daily_signals = torch.full((num_nodes, INPUT_WINDOW, 1), MASK_TOKEN)
        
        for day in range(num_days):
            # Get hours for this day (most recent days first)
            start_h = available_hours - (day + 1) * 24
            end_h = available_hours - day * 24
            
            if start_h < 0:
                start_h = 0
            
            for node in range(num_nodes):
                day_data = signals[node, start_h:end_h, 0]
                valid = day_data[day_data != MASK_TOKEN]
                if len(valid) > 0:
                    # Store in reverse order (oldest day first)
                    day_idx = INPUT_WINDOW - 1 - day
                    daily_signals[node, day_idx, 0] = valid.mean()
        
        # Check daily data coverage
        valid_mask = daily_signals != MASK_TOKEN
        data_coverage = valid_mask.float().mean().item() * 100
        print(f"Daily data coverage: {data_coverage:.1f}% ({num_days} days available)")
        
        if data_coverage < 10:
            print("Warning: Very low data coverage, predictions may be unreliable")
        
        # Step 4: Prepare input for model
        # daily_signals shape: (num_nodes, 14, 1)
        x = daily_signals.to(device)
        edge_index_tensor = edge_index.to(device)
        
        # Step 4: Run inference
        print("Running model inference...")
        model.eval()
        with torch.no_grad():
            predictions, spatial_weights, temporal_weights = model(
                x, edge_index_tensor, return_attention_weights=True
            )
        
        predictions_np = predictions.cpu().numpy().flatten()
        
        # Step 6: Generate explanation
        explanation = generate_explanation(
            spatial_weights=spatial_weights,
            temporal_weights=temporal_weights,
            location_ids=location_ids,
            edge_index=edge_index,
            num_heads=NUM_HEADS,
            location_names=location_names
        )
        
        # Step 7: Build response
        sensor_predictions = []
        valid_predictions = []
        
        for i, (loc_id, coord, pred) in enumerate(zip(location_ids, coords, predictions_np)):
            # Calculate per-sensor data coverage from daily aggregated data
            sensor_valid = valid_mask[i].float().mean().item() * 100
            
            # Clamp predictions to reasonable range (PM2.5 should be positive)
            pred_clamped = float(max(0, min(pred, 999)))
            
            sensor_name = location_names[i] if i < len(location_names) else f"Station {loc_id}"
            
            sensor_predictions.append(SensorPrediction(
                location_id=loc_id,
                name=sensor_name,
                latitude=coord[0],
                longitude=coord[1],
                predicted_pm25=round(pred_clamped, 2),
                data_coverage=round(sensor_valid, 1)
            ))
            
            # Only include sensors with data for average calculation
            if sensor_valid > 0:
                valid_predictions.append(pred_clamped)
        
        # Calculate city average from valid predictions
        if valid_predictions:
            city_avg = float(np.mean(valid_predictions))
        else:
            # Fallback: use all predictions if no sensors have data
            city_avg = float(np.mean(predictions_np)) if len(predictions_np) > 0 else 0.0
        
        # Get health impact assessment
        health_impact = get_health_impact(city_avg)
        
        print(f"Prediction complete. City average PM2.5: {city_avg:.2f} µg/m³ ({health_impact.aqi_category})")
        
        # Save to forecast history if user is logged in
        if user:
            try:
                add_forecast_history(
                    user["id"], city_name.title(), round(city_avg, 2),
                    health_impact.aqi_category, len(location_ids)
                )
            except Exception:
                pass  # Don't fail prediction if history save fails
        
        return PredictionResponse(
            city=city_name.title(),
            num_sensors=len(location_ids),
            num_edges=edge_index.shape[1],
            forecast_pm25=sensor_predictions,
            city_average_pm25=round(city_avg, 2),
            health_impact=health_impact,
            explanation=explanation,
            model_version="1.0.0"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/cities", tags=["Info"])
async def list_cities():
    """
    List all available cities with their bounding boxes.
    """
    cities = []
    for city, bbox in CITY_BBOXES.items():
        cities.append({
            "name": city.title(),
            "code": city,
            "bounding_box": {
                "min_lat": bbox[0],
                "min_lon": bbox[1],
                "max_lat": bbox[2],
                "max_lon": bbox[3]
            }
        })
    return {"cities": cities}


@app.get("/api/cities/search", tags=["Info"])
async def city_search(
    q: str = Query(..., min_length=1, max_length=100, description="City name to search for"),
    limit: int = Query(default=10, ge=1, le=30, description="Max results")
):
    """
    Search Indian cities by name. Returns matching cities with coordinates.
    Use this for the city autocomplete in the UI.
    """
    results = search_cities(q, limit=limit)
    return {
        "query": q,
        "count": len(results),
        "cities": [
            {
                "name": c["name"],
                "state": c["state"],
                "lat": c["lat"],
                "lon": c["lon"],
                "code": c["name"].lower(),
            }
            for c in results
        ]
    }


# ============================================================
# Auth Endpoints
# ============================================================

@app.post("/api/auth/register", tags=["Auth"])
async def api_register(req: RegisterRequest):
    """Register a new user account."""
    return register_user(req.username, req.email, req.password)


@app.post("/api/auth/login", tags=["Auth"])
async def api_login(req: LoginRequest):
    """Login with email and password. Returns JWT token."""
    return login_user(req.email, req.password)


@app.get("/api/auth/me", tags=["Auth"])
async def api_me(user: dict = Depends(get_current_user)):
    """Get current user info."""
    return {"user": user}


# ============================================================
# Saved Locations (Requires Login)
# ============================================================

@app.get("/api/user/locations", tags=["User Features"])
async def api_get_locations(user: dict = Depends(get_current_user)):
    """Get user's saved locations."""
    return {"locations": get_saved_locations(user["id"])}


@app.post("/api/user/locations", tags=["User Features"])
async def api_save_location(req: SaveLocationRequest, user: dict = Depends(get_current_user)):
    """Save a city to favorites."""
    loc_id = save_location(user["id"], req.city_name, req.state, req.lat, req.lon)
    return {"message": f"Saved {req.city_name}", "id": loc_id}


@app.delete("/api/user/locations/{city_name}", tags=["User Features"])
async def api_delete_location(city_name: str, user: dict = Depends(get_current_user)):
    """Remove a saved city."""
    deleted = delete_saved_location(user["id"], city_name)
    if not deleted:
        raise HTTPException(status_code=404, detail="Location not found")
    return {"message": f"Removed {city_name}"}


# ============================================================
# Alert Preferences (Requires Login)
# ============================================================

@app.get("/api/user/alerts", tags=["User Features"])
async def api_get_alerts(user: dict = Depends(get_current_user)):
    """Get user's alert preferences."""
    return {"alerts": get_alerts(user["id"])}


@app.post("/api/user/alerts", tags=["User Features"])
async def api_upsert_alert(req: AlertRequest, user: dict = Depends(get_current_user)):
    """Create or update an alert for a city."""
    alert_id = upsert_alert(user["id"], req.city_name, req.threshold_pm25, req.enabled)
    return {"message": f"Alert set for {req.city_name}", "id": alert_id}


@app.delete("/api/user/alerts/{city_name}", tags=["User Features"])
async def api_delete_alert(city_name: str, user: dict = Depends(get_current_user)):
    """Remove an alert for a city."""
    deleted = delete_alert(user["id"], city_name)
    if not deleted:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": f"Alert removed for {city_name}"}


# ============================================================
# Health Profile (Requires Login)
# ============================================================

@app.get("/api/user/health-profile", tags=["User Features"])
async def api_get_health_profile(user: dict = Depends(get_current_user)):
    """Get user's health profile."""
    profile = get_health_profile(user["id"])
    return {"profile": profile}


@app.post("/api/user/health-profile", tags=["User Features"])
async def api_upsert_health_profile(req: HealthProfileRequest, user: dict = Depends(get_current_user)):
    """Create or update health profile."""
    valid_age_groups = ["child", "teen", "adult", "senior"]
    if req.age_group not in valid_age_groups:
        raise HTTPException(status_code=400, detail=f"age_group must be one of: {valid_age_groups}")
    valid_conditions = [
        "asthma", "copd", "heart_disease", "diabetes",
        "pregnancy", "allergies", "lung_disease", "none"
    ]
    for c in req.conditions:
        if c not in valid_conditions:
            raise HTTPException(status_code=400, detail=f"Invalid condition: {c}. Valid: {valid_conditions}")
    upsert_health_profile(user["id"], req.age_group, req.conditions, req.outdoor_hours)
    return {"message": "Health profile updated"}


# ============================================================
# Forecast History (Requires Login)
# ============================================================

@app.get("/api/user/history", tags=["User Features"])
async def api_get_history(
    limit: int = Query(default=30, ge=1, le=200),
    user: dict = Depends(get_current_user),
):
    """Get user's forecast history."""
    return {"history": get_forecast_history(user["id"], limit=limit)}


@app.get("/api/user/history/export", tags=["User Features"])
async def api_export_history(
    format: str = Query(default="csv", regex="^(csv|json)$"),
    user: dict = Depends(get_current_user),
):
    """Export forecast history as CSV or JSON."""
    history = get_forecast_history(user["id"], limit=500)
    if format == "json":
        return {"history": history}

    # CSV export
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date", "City", "Avg PM2.5", "AQI Category", "Sensors"])
    for h in history:
        writer.writerow([h["created_at"], h["city_name"], h["avg_pm25"], h["aqi_category"], h["num_sensors"]])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=forecast_history.csv"},
    )


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Air Quality Forecasting API Server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative Docs: http://localhost:8000/redoc")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )