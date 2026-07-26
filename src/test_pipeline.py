"""
Phase 4: System Integration Test
==================================
End-to-end validation of the full Cyber SOC pipeline:
  1. Data generation   - schema, row counts, anomaly rate, cold-start %
  2. Feature extraction - new feature columns, value ranges
  3. Model training     - model files, inference latency (< 20 ms)
  4. Classification     - Precision / Recall / F1 / FPR per class
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support,
)

# Ensure sibling modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ===================================================================
# Test 1: Data Generation & Schema Validation
# ===================================================================
def test_data_generation():
    print("\n" + "=" * 64)
    print("  TEST 1 - Data Generation & Schema Validation")
    print("=" * 64)

    from data_generator import generate_dataset

    t0 = time.perf_counter()
    df, profiles = generate_dataset()
    elapsed = time.perf_counter() - t0
    print(f"\n  Generation time: {elapsed:.2f} s")

    # Row count
    assert len(df) >= 15_000, (
        f"FAIL: expected >= 15 000 rows, got {len(df)}")
    print(f"  [PASS] Row count: {len(df):,} (>= 15 000)")

    # Schema
    required = [
        "entity_id", "entity_type", "timestamp", "source_ip",
        "geo_location", "resource_accessed", "auth_method",
        "session_duration", "command_sequence", "device_fingerprint",
        "label",
    ]
    for col in required:
        assert col in df.columns, f"FAIL: missing column '{col}'"
    print(f"  [PASS] All {len(required)} required columns present")

    # Entity types
    types = set(df["entity_type"].unique())
    assert types == {"user", "service_account", "edge_device"}, (
        f"FAIL: unexpected entity_types {types}")
    print(f"  [PASS] Entity types: {sorted(types)}")

    # Anomaly rate (accept 2-10 % to handle stochastic variation)
    anom_rate = (df["label"] != "normal").mean()
    assert 0.02 <= anom_rate <= 0.10, (
        f"FAIL: anomaly rate {anom_rate:.2%} out of [2 %, 10 %]")
    print(f"  [PASS] Anomaly rate: {anom_rate:.2%}")

    # Cold-start entities
    n_cold = sum(1 for p in profiles.values() if p["is_coldstart"])
    pct_cold = n_cold / len(profiles)
    assert 0.03 <= pct_cold <= 0.10, (
        f"FAIL: cold-start pct {pct_cold:.2%} out of [3 %, 10 %]")
    print(f"  [PASS] Cold-start entities: {n_cold} ({pct_cold:.1%})")

    # Taxonomy JSON
    tax_path = os.path.join(DATA_DIR, "data_taxonomy_documentation.json")
    assert os.path.isfile(tax_path), "FAIL: taxonomy JSON missing"
    print(f"  [PASS] Taxonomy documentation saved")

    # No nulls
    nulls = int(df.isnull().sum().sum())
    assert nulls == 0, f"FAIL: {nulls} null values detected"
    print(f"  [PASS] Zero null values")

    # Label diversity (all 7 anomaly types + normal)
    unique_labels = set(df["label"].unique())
    expected_labels = {
        "normal", "brute_force", "impossible_travel",
        "credential_stuffing", "lateral_movement",
        "device_spoofing", "low_and_slow_exfil", "insider_drift",
    }
    assert expected_labels.issubset(unique_labels), (
        f"FAIL: missing labels {expected_labels - unique_labels}")
    print(f"  [PASS] All 8 label classes present")

    return df, profiles


# ===================================================================
# Test 2: Feature Extraction
# ===================================================================
def test_feature_extraction():
    print("\n" + "=" * 64)
    print("  TEST 2 - Feature Extraction")
    print("=" * 64)

    from feature_engine import extract_features

    t0 = time.perf_counter()
    df, label_encoders = extract_features()
    elapsed = time.perf_counter() - t0
    print(f"\n  Feature extraction time: {elapsed:.2f} s")

    # New feature columns
    expected_feats = [
        "geo_velocity_kmh", "failed_auth_count_5m",
        "resource_rarity_score", "fingerprint_mismatch_flag",
        "hour_of_day", "day_of_week", "is_weekend", "is_off_hours",
        "command_count",
        "entity_type_encoded", "auth_method_encoded",
        "geo_location_encoded", "resource_accessed_encoded",
    ]
    for feat in expected_feats:
        assert feat in df.columns, f"FAIL: missing feature '{feat}'"
    print(f"  [PASS] All {len(expected_feats)} engineered features present")

    # Value-range sanity checks
    assert df["geo_velocity_kmh"].min() >= 0, "FAIL: negative velocity"
    assert df["resource_rarity_score"].between(0, 1).all(), (
        "FAIL: rarity score out of [0, 1]")
    assert set(df["fingerprint_mismatch_flag"].unique()).issubset(
        {0, 1}), "FAIL: mismatch flag not binary"
    assert df["hour_of_day"].between(0, 23).all(), (
        "FAIL: hour out of range")
    assert df["day_of_week"].between(0, 6).all(), (
        "FAIL: day_of_week out of range")
    print(f"  [PASS] Feature value ranges validated")

    # Label encoders populated
    assert len(label_encoders) == 4, "FAIL: expected 4 label encoders"
    print(f"  [PASS] Label encoders: {list(label_encoders.keys())}")

    return df, label_encoders


# ===================================================================
# Test 3: Model Training & Inference
# ===================================================================
def test_model_training():
    print("\n" + "=" * 64)
    print("  TEST 3 - Model Training & Inference")
    print("=" * 64)

    from model_engine import run_training_pipeline, FEATURE_COLUMNS

    t0 = time.perf_counter()
    df, iso_forest, xgb_clf, attack_le, explainer, report = (
        run_training_pipeline())
    elapsed = time.perf_counter() - t0
    print(f"\n  Full training pipeline time: {elapsed:.2f} s")

    # Model artefact files
    expected_files = [
        "baseline_profiler.pkl", "anomaly_detector.pkl",
        "attack_classifier.pkl", "shap_explainer.pkl",
    ]
    for fn in expected_files:
        fpath = os.path.join(MODEL_DIR, fn)
        assert os.path.isfile(fpath), f"FAIL: {fn} not found"
        sz = os.path.getsize(fpath)
        assert sz > 0, f"FAIL: {fn} is empty"
    print(f"  [PASS] All {len(expected_files)} model files saved & non-empty")

    # Anomaly score range
    assert "anomaly_score" in df.columns, "FAIL: anomaly_score missing"
    assert df["anomaly_score"].between(0, 1).all(), (
        "FAIL: anomaly_score out of [0, 1]")
    print(f"  [PASS] anomaly_score in [0.0, 1.0]")

    # -- Inference latency benchmark --
    feat_cols = FEATURE_COLUMNS + ["anomaly_score"]
    X_bench = df[feat_cols].iloc[:100].values.astype(np.float32)
    X_if    = df[FEATURE_COLUMNS].iloc[:100].values.astype(np.float32)

    # Warm-up
    _ = iso_forest.predict(X_if[:1])
    _ = xgb_clf.predict(X_bench[:1])

    t0 = time.perf_counter()
    for i in range(100):
        _ = iso_forest.decision_function(X_if[i:i+1])
        _ = xgb_clf.predict(X_bench[i:i+1])
    ms_per_call = (time.perf_counter() - t0) / 100 * 1000

    print(f"  Inference latency: {ms_per_call:.2f} ms / sample "
          f"(target < 20 ms)")
    assert ms_per_call < 20, (
        f"FAIL: inference too slow ({ms_per_call:.2f} ms)")
    print(f"  [PASS] Latency passed: {ms_per_call:.2f} ms < 20 ms")

    # Deserialization test
    with open(os.path.join(MODEL_DIR, "attack_classifier.pkl"), "rb") as f:
        loaded = pickle.load(f)
    assert "model" in loaded and "label_encoder" in loaded
    preds = loaded["model"].predict(X_bench[:5])
    assert len(preds) == 5
    print("  [PASS] Model deserialization & inference OK")

    return df, report, attack_le, xgb_clf, iso_forest


# ===================================================================
# Test 4: Classification Metrics & FPR
# ===================================================================
def test_classification_metrics(report, attack_le, df,
                                xgb_clf, iso_forest):
    print("\n" + "=" * 64)
    print("  TEST 4 - Classification Metrics")
    print("=" * 64)

    from model_engine import FEATURE_COLUMNS

    # -- Per-class metrics --
    print(f"\n  {'Class':<25s} {'Precision':>10s} "
          f"{'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
    print(f"  {'-' * 65}")
    for cls in attack_le.classes_:
        m = report[cls]
        print(f"  {cls:<25s} {m['precision']:10.4f} "
              f"{m['recall']:10.4f} {m['f1-score']:10.4f} "
              f"{m['support']:10.0f}")

    print(f"  {'-' * 65}")
    for avg_key in ["macro avg", "weighted avg"]:
        if avg_key in report:
            m = report[avg_key]
            print(f"  {avg_key:<25s} {m['precision']:10.4f} "
                  f"{m['recall']:10.4f} {m['f1-score']:10.4f}")

    acc = report.get("accuracy", 0)
    print(f"\n  Overall Accuracy: {acc:.4f}")

    # ── False Positive Rate (binary: normal vs any anomaly) ──
    feat_cols = FEATURE_COLUMNS + ["anomaly_score"]
    X_all = df[feat_cols].values.astype(np.float32)
    y_pred_all = xgb_clf.predict(X_all)

    normal_idx = int(
        np.where(attack_le.classes_ == "normal")[0][0])
    y_true_bin = (df["label"] != "normal").astype(int).values
    y_pred_bin = (y_pred_all != normal_idx).astype(int)

    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"\n  Binary Anomaly Detection (normal vs anomaly):")
    print(f"    True Positives:  {tp:>8,}")
    print(f"    True Negatives:  {tn:>8,}")
    print(f"    False Positives: {fp:>8,}")
    print(f"    False Negatives: {fn:>8,}")
    print(f"    False Positive Rate (FPR): {fpr:.4f}")
    print(f"    False Negative Rate (FNR): {fnr:.4f}")

    return report


# ===================================================================
# Entry-point
# ===================================================================
def main():
    print("+" + "-" * 62 + "+")
    print("|  CYBER SOC PLATFORM - SYSTEM INTEGRATION TEST               |")
    print("+" + "-" * 62 + "+")

    errors = []
    try:
        # Phase 1
        df_raw, profiles = test_data_generation()

        # Phase 2
        df_feat, le = test_feature_extraction()

        # Phase 3
        df, report, attack_le, xgb_clf, iso_forest = (
            test_model_training())

        # Phase 4: metrics
        test_classification_metrics(
            report, attack_le, df, xgb_clf, iso_forest)

    except AssertionError as ae:
        errors.append(str(ae))
    except Exception as exc:
        errors.append(f"RUNTIME ERROR: {exc}")
        import traceback
        traceback.print_exc()

    # -- Summary --
    print("\n" + "=" * 64)
    if errors:
        print("  [FAIL] TESTS FAILED")
        for e in errors:
            print(f"    -> {e}")
        sys.exit(1)
    else:
        print("  [PASS] ALL TESTS PASSED - Zero runtime crashes")
    print("=" * 64)


if __name__ == "__main__":
    main()
