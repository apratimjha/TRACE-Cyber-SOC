"""
Phase 2: Feature Engineering Pipeline
======================================
Computes dynamic sequential features from raw access logs:
  - geo_velocity_kmh        - haversine distance / time between consecutive events
  - failed_auth_count_5m    - sliding-window count of failed auth attempts
  - resource_rarity_score   - inverse-frequency rarity of each accessed resource
  - fingerprint_mismatch_flag - binary flag for device fingerprint deviation
  - Temporal features       - hour_of_day, day_of_week, is_weekend, is_off_hours
  - Categorical encodings   - LabelEncoded entity_type, auth_method, geo, resource
  - command_count           - number of commands in the JSON command_sequence
"""

import os
import json
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ---------------------------------------------------------------------------
# Geo-coordinate lookup  (must match data_generator.GEO_LOCATIONS)
# ---------------------------------------------------------------------------
GEO_COORDS = {
    "New York, US":      (40.7128,  -74.0060),
    "San Francisco, US": (37.7749, -122.4194),
    "London, UK":        (51.5074,   -0.1278),
    "Tokyo, JP":         (35.6762,  139.6503),
    "Sydney, AU":       (-33.8688,  151.2093),
    "Berlin, DE":        (52.5200,   13.4050),
    "Mumbai, IN":        (19.0760,   72.8777),
    "Toronto, CA":       (43.6532,  -79.3832),
    "Singapore, SG":     ( 1.3521,  103.8198),
    "Sao Paulo, BR":    (-23.5505,  -46.6333),
    "Dubai, AE":         (25.2048,   55.2708),
    "Seoul, KR":         (37.5665,  126.9780),
    "Chicago, US":       (41.8781,  -87.6298),
    "Austin, US":        (30.2672,  -97.7431),
    "Seattle, US":       (47.6062, -122.3321),
    "Denver, US":        (39.7392, -104.9903),
    "Moscow, RU":        (55.7558,   37.6173),
    "Paris, FR":         (48.8566,    2.3522),
    "Beijing, CN":       (39.9042,  116.4074),
    "Lagos, NG":         ( 6.5244,    3.3792),
}


# ===================================================================
# Vectorised haversine
# ===================================================================
def _haversine_vec(lat1, lon1, lat2, lon2):
    """Haversine distance in km - fully vectorised with NumPy."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = (
        np.radians(lat1), np.radians(lon1),
        np.radians(lat2), np.radians(lon2),
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + (
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ===================================================================
# Feature functions
# ===================================================================
def _compute_geo_velocity(df):
    """Vectorised geo-velocity (km/h) between consecutive per-entity events."""
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    # Map geo → lat/lon
    df["_lat"] = df["geo_location"].map(
        lambda g: GEO_COORDS.get(g, (0.0, 0.0))[0])
    df["_lon"] = df["geo_location"].map(
        lambda g: GEO_COORDS.get(g, (0.0, 0.0))[1])

    # Previous event (within same entity)
    df["_prev_lat"] = df.groupby("entity_id")["_lat"].shift(1)
    df["_prev_lon"] = df.groupby("entity_id")["_lon"].shift(1)
    df["_prev_ts"]  = df.groupby("entity_id")["timestamp"].shift(1)

    has_prev = df["_prev_lat"].notna()

    dist_km = np.zeros(len(df))
    dist_km[has_prev] = _haversine_vec(
        df.loc[has_prev, "_prev_lat"].values,
        df.loc[has_prev, "_prev_lon"].values,
        df.loc[has_prev, "_lat"].values,
        df.loc[has_prev, "_lon"].values,
    )

    time_h = np.zeros(len(df))
    time_h[has_prev] = (
        (df.loc[has_prev, "timestamp"] - df.loc[has_prev, "_prev_ts"])
        .dt.total_seconds().values / 3600.0
    )

    # Avoid division by near-zero
    df["geo_velocity_kmh"] = np.where(
        time_h > 0.001, dist_km / time_h, 0.0)

    df.drop(columns=["_lat", "_lon", "_prev_lat", "_prev_lon", "_prev_ts"],
            inplace=True)
    return df


def _compute_failed_auth_rolling(df, window_minutes=5):
    """Sliding-window count of failed auth attempts per entity (O(n))."""
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    df["_is_failed"] = (df["session_duration"] < 1.0).astype(np.int8)

    counts = np.zeros(len(df), dtype=np.int32)
    window_ns = np.timedelta64(window_minutes, "m")

    for _, grp in df.groupby("entity_id"):
        idx = grp.index.values
        ts  = grp["timestamp"].values
        fl  = grp["_is_failed"].values

        left = 0
        running = 0
        for right in range(len(idx)):
            if fl[right]:
                running += 1
            # shrink left side of window
            while left < right and (ts[right] - ts[left]) > window_ns:
                if fl[left]:
                    running -= 1
                left += 1
            counts[idx[right]] = running

    df["failed_auth_count_5m"] = counts
    df.drop(columns=["_is_failed"], inplace=True)
    return df


def _compute_resource_rarity(df):
    """Resource rarity score ∈ [0, 1]  (1 = rarest)."""
    freq = df["resource_accessed"].value_counts(normalize=True)
    df["resource_rarity_score"] = df["resource_accessed"].map(
        lambda r: 1.0 - freq.get(r, 0.0))
    return df


def _compute_fingerprint_mismatch(df):
    """Binary flag: 1 if fingerprint differs from entity's modal baseline."""
    mode_fp = (
        df.groupby("entity_id")["device_fingerprint"]
        .agg(lambda x: x.value_counts().index[0])
    )
    df["fingerprint_mismatch_flag"] = (
        df.apply(
            lambda r: int(
                r["device_fingerprint"]
                != mode_fp.get(r["entity_id"], r["device_fingerprint"])),
            axis=1,
        )
    ).astype(np.int8)
    return df


def _extract_temporal_features(df):
    """Hour of day, day of week, weekend flag, off-hours flag."""
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(np.int8)
    df["is_off_hours"] = (
        (df["hour_of_day"] < 7) | (df["hour_of_day"] > 19)
    ).astype(np.int8)
    return df


def _extract_command_count(df):
    """Number of commands in the JSON command_sequence."""
    def _safe_len(s):
        try:
            return len(json.loads(s))
        except (json.JSONDecodeError, TypeError):
            return 0
    df["command_count"] = df["command_sequence"].apply(_safe_len)
    return df


def _encode_categoricals(df):
    """LabelEncode categorical columns; return encoders dict."""
    label_encoders = {}
    for col in ["entity_type", "auth_method",
                "geo_location", "resource_accessed"]:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders


# ===================================================================
# Public API
# ===================================================================
def extract_features(csv_path=None):
    """Load raw logs, compute all features, and return enriched DataFrame.

    Parameters
    ----------
    csv_path : str, optional
        Path to the CSV.  Defaults to ``data/synthetic_access_logs.csv``.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with all engineered features appended.
    label_encoders : dict
        Mapping {column_name: fitted LabelEncoder}.
    """
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "synthetic_access_logs.csv")

    print("[Phase 2] Extracting features ...")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    print(f"  Loaded {len(df):,} rows from {csv_path}")

    # Sequential / behavioural features
    df = _compute_geo_velocity(df)
    print("  [PASS] geo_velocity_kmh")

    df = _compute_failed_auth_rolling(df)
    print("  [PASS] failed_auth_count_5m")

    df = _compute_resource_rarity(df)
    print("  [PASS] resource_rarity_score")

    df = _compute_fingerprint_mismatch(df)
    print("  [PASS] fingerprint_mismatch_flag")

    # Temporal features
    df = _extract_temporal_features(df)
    print("  [PASS] hour_of_day, day_of_week, is_weekend, is_off_hours")

    # Command count
    df = _extract_command_count(df)
    print("  [PASS] command_count")

    # Categorical encoding
    df, label_encoders = _encode_categoricals(df)
    print("  [PASS] Categorical encodings (entity_type, auth_method, "
          "geo_location, resource_accessed)")

    print(f"  Feature matrix shape: {df.shape}")
    return df, label_encoders


# ===================================================================
if __name__ == "__main__":
    df, le = extract_features()
    print("\nFeature columns added:")
    added = [
        "geo_velocity_kmh", "failed_auth_count_5m",
        "resource_rarity_score", "fingerprint_mismatch_flag",
        "hour_of_day", "day_of_week", "is_weekend", "is_off_hours",
        "command_count", "entity_type_encoded", "auth_method_encoded",
        "geo_location_encoded", "resource_accessed_encoded",
    ]
    for col in added:
        lo = df[col].min()
        hi = df[col].max()
        print(f"  {col:>30s}  range=[{lo:.4f}, {hi:.4f}]")
