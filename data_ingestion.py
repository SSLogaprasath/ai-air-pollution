"""
Data Ingestion Module for Air Quality Forecasting System.

This module provides functions to fetch sensor data from OpenAQ API
and build graph structures for GNN-based forecasting models.
"""

import time
import math
import atexit
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data
from openaq import OpenAQ


# OpenAQ API Key
OPENAQ_API_KEY: str = "dff6c602f4c1d99c37ee7b77a0278070c47508e244d5db7d3aa28fdafce97337"

# Singleton OpenAQ client (workaround for library bug where closing
# the client breaks subsequent new client instances)
_openaq_client: Optional[OpenAQ] = None


def _get_openaq_client() -> OpenAQ:
    """
    Get or create the singleton OpenAQ client.
    
    Note: The openaq library has a bug where closing a client
    breaks subsequent client instances. We work around this
    by using a singleton that is only closed at program exit.
    """
    global _openaq_client
    if _openaq_client is None:
        _openaq_client = OpenAQ(api_key=OPENAQ_API_KEY)
    return _openaq_client


def _cleanup_client():
    """Cleanup the OpenAQ client on program exit."""
    global _openaq_client
    if _openaq_client is not None:
        try:
            _openaq_client.close()
        except Exception:
            pass
        _openaq_client = None


# Register cleanup on program exit
atexit.register(_cleanup_client)


# City bounding boxes (min_lat, min_lon, max_lat, max_lon)
CITY_BBOXES: dict[str, tuple[float, float, float, float]] = {
    # South India
    "tiruchirappalli": (10.7000, 78.6000, 10.9500, 78.8500),
    "chennai": (12.9000, 80.1000, 13.2500, 80.3500),
    "bangalore": (12.8500, 77.4500, 13.1500, 77.7500),
    "hyderabad": (17.2500, 78.2500, 17.5500, 78.6500),
    "coimbatore": (10.9000, 76.9000, 11.1000, 77.1000),
    "kochi": (9.9000, 76.2000, 10.1000, 76.4000),
    "visakhapatnam": (17.6000, 83.1500, 17.8500, 83.4000),
    "madurai": (9.8500, 78.0500, 10.0500, 78.2500),
    "thiruvananthapuram": (8.4000, 76.8500, 8.6000, 77.0500),
    # North India
    "delhi": (28.4000, 76.8000, 28.9000, 77.4000),
    "lucknow": (26.7500, 80.8500, 27.0000, 81.0500),
    "jaipur": (26.8000, 75.7000, 27.0000, 75.9000),
    "chandigarh": (30.6500, 76.6800, 30.8000, 76.8500),
    "varanasi": (25.2500, 82.9000, 25.4000, 83.1000),
    "agra": (27.1000, 77.9000, 27.2500, 78.1000),
    "patna": (25.5500, 85.0500, 25.7000, 85.2500),
    "amritsar": (31.5500, 74.8000, 31.7500, 74.9500),
    # West India
    "mumbai": (18.8900, 72.7500, 19.2700, 73.0500),
    "pune": (18.4000, 73.7500, 18.6500, 74.0000),
    "ahmedabad": (22.9500, 72.5000, 23.1500, 72.7000),
    "nagpur": (21.0500, 79.0000, 21.2500, 79.1500),
    "goa": (15.3500, 73.8500, 15.5500, 74.0500),
    # East India
    "kolkata": (22.4000, 88.2000, 22.7000, 88.5000),
    "bhubaneswar": (20.2000, 85.7500, 20.3500, 85.9000),
    "guwahati": (26.1000, 91.6500, 26.2500, 91.8500),
    "ranchi": (23.2500, 85.2500, 23.4500, 85.4500),
    # Central India
    "bhopal": (23.1500, 77.3000, 23.3500, 77.5000),
    "indore": (22.6000, 75.7500, 22.8500, 75.9500),
    "raipur": (21.1800, 81.5500, 21.3500, 81.7500),
}

# API rate limit configuration
RATE_LIMIT_DELAY: float = 1.0  # seconds between API calls
MAX_RETRIES: int = 3
RETRY_BACKOFF: float = 2.0  # exponential backoff multiplier

# Mask token for missing data (not 0 since 0 implies clean air)
MASK_TOKEN: float = -1.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the Haversine distance between two points on Earth.

    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees

    Returns:
        Distance in kilometers
    """
    R = 6371.0  # Earth's radius in kilometers

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def _api_call_with_retry(api_func, *args, **kwargs):
    """
    Execute an API call with retry logic and rate limiting.

    Args:
        api_func: The API function to call
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        API response

    Raises:
        Exception: If all retries are exhausted
    """
    last_exception = None

    for attempt in range(MAX_RETRIES):
        try:
            # Rate limit before each attempt
            if attempt > 0:
                time.sleep(RATE_LIMIT_DELAY)
            return api_func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in str(e) or "too many" in error_str:
                wait_time = RATE_LIMIT_DELAY * (RETRY_BACKOFF ** attempt)
                print(f"Rate limited. Waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES}")
                time.sleep(wait_time)
            else:
                raise e

    raise last_exception


def build_distance_matrix(
    coords: list[tuple[float, float]]
) -> np.ndarray:
    """
    Build a distance matrix using Haversine distance.

    Args:
        coords: List of (latitude, longitude) tuples

    Returns:
        Symmetric distance matrix of shape (n, n)
    """
    n = len(coords)
    distance_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            dist = haversine_distance(
                coords[i][0], coords[i][1],
                coords[j][0], coords[j][1]
            )
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix


def distance_matrix_to_edge_index(
    distance_matrix: np.ndarray,
    radius_km: float
) -> torch.Tensor:
    """
    Convert distance matrix to PyG edge_index tensor.

    Creates edges between nodes that are within radius_km of each other.

    Args:
        distance_matrix: Symmetric distance matrix
        radius_km: Maximum distance for edge connection

    Returns:
        Edge index tensor of shape (2, num_edges)
    """
    n = distance_matrix.shape[0]
    sources = []
    targets = []

    for i in range(n):
        for j in range(n):
            if i != j and distance_matrix[i, j] <= radius_km:
                sources.append(i)
                targets.append(j)

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    return edge_index


def fetch_city_graph(
    city: str = "delhi",
    radius_km: int = 10,
    bbox: Optional[tuple[float, float, float, float]] = None
) -> Data:
    """
    Fetch sensor locations for a city and build a graph structure.

    Uses the OpenAQ API to find all sensor locations within a bounding box
    for the given city, extracts static features, builds a distance matrix,
    and creates a PyTorch Geometric Data object.

    Args:
        city: City name (default: 'delhi').
        radius_km: Maximum distance in km for connecting sensors (default: 10)
        bbox: Optional explicit bounding box (min_lat, min_lon, max_lat, max_lon).
              If provided, overrides the city lookup.

    Returns:
        PyG Data object containing:
            - x: Node features tensor (num_nodes, 3) with [id, lat, lon]
            - edge_index: Edge connectivity tensor (2, num_edges)
            - location_ids: List of OpenAQ location IDs
            - coords: List of (lat, lon) tuples

    Raises:
        ValueError: If city is not in CITY_BBOXES and no bbox provided
    """
    if bbox is not None:
        min_lat, min_lon, max_lat, max_lon = bbox
    else:
        city_lower = city.lower()
        if city_lower not in CITY_BBOXES:
            # Try the dynamic cities database as fallback
            try:
                from indian_cities import get_city_bbox
                dynamic_bbox = get_city_bbox(city)
                if dynamic_bbox is not None:
                    min_lat, min_lon, max_lat, max_lon = dynamic_bbox
                else:
                    raise ValueError(
                        f"City '{city}' not found in any city database."
                    )
            except ImportError:
                raise ValueError(
                    f"City '{city}' not supported. "
                    f"Available cities: {list(CITY_BBOXES.keys())}"
                )
        else:
            min_lat, min_lon, max_lat, max_lon = CITY_BBOXES[city_lower]

    # Get singleton OpenAQ client
    client = _get_openaq_client()
    
    print(f"Fetching sensor locations for {city}...")

    locations_response = _api_call_with_retry(
        client.locations.list,
        bbox=(min_lon, min_lat, max_lon, max_lat),
        limit=1000
    )

    # Extract location data
    locations = locations_response.results
    if not locations:
        print(f"No sensors found in {city}. Returning empty graph.")
        return Data(
            x=torch.zeros((0, 3), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            location_ids=[],
            coords=[]
        )

    print(f"Found {len(locations)} sensor locations")

    # Extract static features: ID, Latitude, Longitude, Name
    location_ids = []
    coords = []
    features = []
    location_names = []

    for loc in locations:
        loc_id = loc.id
        lat = loc.coordinates.latitude
        lon = loc.coordinates.longitude
        name = getattr(loc, 'name', None) or f"Station {loc_id}"

        if lat is not None and lon is not None:
            location_ids.append(loc_id)
            coords.append((lat, lon))
            features.append([float(loc_id), lat, lon])
            location_names.append(name)

    if not features:
        print("No valid coordinates found. Returning empty graph.")
        return Data(
            x=torch.zeros((0, 3), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            location_ids=[],
            coords=[]
        )

    # Build node feature tensor
    x = torch.tensor(features, dtype=torch.float32)

    # Build distance matrix
    print("Building distance matrix...")
    distance_matrix = build_distance_matrix(coords)

    # Convert to edge_index
    print(f"Creating edges for sensors within {radius_km} km...")
    edge_index = distance_matrix_to_edge_index(distance_matrix, radius_km)

    print(f"Graph built: {x.shape[0]} nodes, {edge_index.shape[1]} edges")

    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        location_ids=location_ids,
        coords=coords,
        location_names=location_names
    )

    return data


def fetch_historical_signals(
    location_ids: list[int],
    hours: int = 24
) -> torch.Tensor:
    """
    Fetch historical PM2.5 measurements for specified sensor locations.

    Args:
        location_ids: List of OpenAQ location IDs
        hours: Number of hours of historical data to fetch (default: 24)

    Returns:
        Tensor of shape (num_nodes, num_hours, 1) containing PM2.5 values.
        Missing data is filled with MASK_TOKEN (-1.0).
    """
    if not location_ids:
        return torch.zeros((0, hours, 1), dtype=torch.float32)

    num_nodes = len(location_ids)

    # Initialize output tensor with mask token
    signals = torch.full((num_nodes, hours, 1), MASK_TOKEN, dtype=torch.float32)

    # Calculate time range
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours)

    # Get singleton OpenAQ client
    client = _get_openaq_client()

    for node_idx, loc_id in enumerate(location_ids):
        print(f"Fetching data for location {loc_id} ({node_idx + 1}/{num_nodes})...")
        
        try:
            # Fetch sensors for this location
            time.sleep(RATE_LIMIT_DELAY)
            sensors_response = client.locations.sensors(locations_id=loc_id)

            sensors = sensors_response.results
            if not sensors:
                print(f"  No sensors found for location {loc_id}")
                continue

            # Find PM2.5 sensor
            pm25_sensor_id = None
            for sensor in sensors:
                param = sensor.parameter
                if param:
                    # Handle both dict and object access patterns
                    param_name = param.get('name', '') if isinstance(param, dict) else getattr(param, 'name', '')
                    if param_name and param_name.lower() in ["pm25", "pm2.5"]:
                        pm25_sensor_id = sensor.id
                        break

            if pm25_sensor_id is None:
                print(f"  No PM2.5 sensor found for location {loc_id}")
                continue

            # Fetch recent measurements (without strict date filter to get latest available data)
            # OpenAQ API limit is 1000, so cap the request
            request_limit = min(hours * 4, 1000)
            time.sleep(RATE_LIMIT_DELAY)
            hourly_response = client.measurements.list(
                sensors_id=pm25_sensor_id,
                limit=request_limit
            )

            measurements = hourly_response.results
            if not measurements:
                print(f"  No PM2.5 measurements found for location {loc_id}")
                continue

            # Sort measurements by time (most recent first) and take the requested hours
            sorted_measurements = sorted(
                [m for m in measurements if m.value is not None and m.period and m.period.datetime_from],
                key=lambda m: m.period.datetime_from.utc,
                reverse=True
            )[:hours * 4]

            # Organize measurements by hour slot (0 = oldest, hours-1 = most recent)
            hourly_values: dict[int, list[float]] = {h: [] for h in range(hours)}
            
            if sorted_measurements:
                # Use the most recent measurement's time as reference
                ref_time_str = sorted_measurements[0].period.datetime_from.utc
                from dateutil import parser as date_parser
                ref_time = date_parser.parse(ref_time_str)
                if ref_time.tzinfo is None:
                    ref_time = ref_time.replace(tzinfo=timezone.utc)

                for m in sorted_measurements:
                    meas_time_str = m.period.datetime_from.utc
                    meas_time = date_parser.parse(meas_time_str)
                    if meas_time.tzinfo is None:
                        meas_time = meas_time.replace(tzinfo=timezone.utc)

                    hours_ago = int((ref_time - meas_time).total_seconds() / 3600)

                    if 0 <= hours_ago < hours:
                        hour_idx = hours - 1 - hours_ago  # Most recent is last
                        hourly_values[hour_idx].append(m.value)

            # Average multiple measurements per hour
            for hour_idx, values in hourly_values.items():
                if values:
                    avg_value = sum(values) / len(values)
                    signals[node_idx, hour_idx, 0] = avg_value

            filled_hours = sum(1 for h in range(hours) if signals[node_idx, h, 0] != MASK_TOKEN)
            print(f"  Retrieved {filled_hours}/{hours} hours of data")

        except Exception as e:
            print(f"  Error fetching data for location {loc_id}: {e}")
            continue

    # Report data coverage
    total_values = num_nodes * hours
    missing_values = (signals == MASK_TOKEN).sum().item()
    coverage = (total_values - missing_values) / total_values * 100
    print(f"Data coverage: {coverage:.1f}% ({total_values - int(missing_values)}/{total_values} values)")

    return signals


def fetch_latest_signals(
    location_ids: list[int]
) -> torch.Tensor:
    """
    Fetch the latest PM2.5 measurement for each location.

    Args:
        location_ids: List of OpenAQ location IDs

    Returns:
        Tensor of shape (num_nodes, 1) containing latest PM2.5 values.
        Missing data is filled with MASK_TOKEN (-1.0).
    """
    if not location_ids:
        return torch.zeros((0, 1), dtype=torch.float32)

    num_nodes = len(location_ids)
    signals = torch.full((num_nodes, 1), MASK_TOKEN, dtype=torch.float32)

    # Get singleton OpenAQ client
    client = _get_openaq_client()

    for node_idx, loc_id in enumerate(location_ids):
        try:
            time.sleep(RATE_LIMIT_DELAY)
            latest_response = client.locations.latest(locations_id=loc_id)

            results = latest_response.results
            if results:
                for sensor in results:
                    param = sensor.parameter
                    if param:
                        # Handle both dict and object access patterns
                        param_name = param.get('name', '') if isinstance(param, dict) else getattr(param, 'name', '')
                        if param_name and param_name.lower() in ["pm25", "pm2.5"]:
                            if sensor.latest and sensor.latest.value is not None:
                                signals[node_idx, 0] = sensor.latest.value
                            break

        except Exception as e:
            print(f"Error fetching latest for location {loc_id}: {e}")
            continue

    return signals


# Example usage
if __name__ == "__main__":
    from datetime import datetime
    
    # Fetch graph for Mumbai (most active sensors as of Jan 2026)
    print("=" * 60)
    print("Fetching city graph...")
    print("=" * 60)
    graph_data = fetch_city_graph(city="mumbai", radius_km=15)

    print(f"\nGraph Summary:")
    print(f"  Nodes: {graph_data.x.shape[0]}")
    print(f"  Edges: {graph_data.edge_index.shape[1]}")
    print(f"  Node features shape: {graph_data.x.shape}")

    if graph_data.location_ids:
        # Use known active location IDs from Mumbai for testing
        # These sensors have been reporting data recently:
        # - Colaba (6927), Mahape (6943), Kurla (6945)
        active_ids = [6927, 6943, 6945]
        # Filter to only use IDs that are in our graph
        sample_ids = [lid for lid in active_ids if lid in graph_data.location_ids]
        if not sample_ids:
            sample_ids = graph_data.location_ids[:3]
        
        print("\n" + "=" * 60)
        print(f"Fetching historical signals for {len(sample_ids)} active sensors...")
        print("=" * 60)

        signals = fetch_historical_signals(sample_ids, hours=12)

        print(f"\nSignals tensor shape: {signals.shape}")
        print(f"Mask token value: {MASK_TOKEN}")
        
        # Show sample of the data tensor
        print(f"\nSample signals (all locations, first 6 hours):")
        print(signals[:, :6, 0])
