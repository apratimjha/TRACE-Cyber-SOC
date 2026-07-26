import os
import sys
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# Ensure we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
# pyrefly: ignore [missing-import]
from feature_engine import extract_features
# pyrefly: ignore [missing-import]
from model_engine import FEATURE_COLUMNS, AnomalyDetectorWrapper

st.set_page_config(
    page_title="Cyber SOC Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enterprise styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 5px;
        border-left: 5px solid #0078D7;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #FFFFFF;
    }
    .metric-label {
        font-size: 14px;
        color: #A0A0A0;
    }
    .stDataFrame {
        font-family: 'Courier New', Courier, monospace;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

@st.cache_data
def load_and_score_data():
    df, le = extract_features(os.path.join(DATA_DIR, "synthetic_access_logs.csv"))
    
    with open(os.path.join(MODEL_DIR, "anomaly_detector.pkl"), "rb") as f:
        anomaly_detector = pickle.load(f)
        
    with open(os.path.join(MODEL_DIR, "attack_classifier.pkl"), "rb") as f:
        attack_dict = pickle.load(f)
        xgb_clf = attack_dict["model"]
        attack_le = attack_dict["label_encoder"]

    # Score anomalies
    # Because we need history, for the dashboard we can just use the decision_function
    # Since our Wrapper expects a 2D array and expands it to seq_len=1 for simple latency scoring,
    # it might not give the exact true sequential score if passed all at once without groups.
    # To be accurate to the dashboard, let's just use the df as-is and we will fake the sequence
    # by just passing it to the wrapper which pads it.
    X_if = df[FEATURE_COLUMNS].values.astype(np.float32)
    scores = anomaly_detector.decision_function(X_if)
    df["anomaly_score"] = scores
    
    # Predict attack types
    feat_cols = FEATURE_COLUMNS + ["anomaly_score"]
    X_attack = df[feat_cols].values.astype(np.float32)
    preds = xgb_clf.predict(X_attack)
    
    df["predicted_attack"] = attack_le.inverse_transform(preds)
    
    return df

def main():
    st.title("Enterprise SOC Dashboard")
    st.markdown("---")

    with st.spinner("Loading telemetry and ML models..."):
        try:
            df = load_and_score_data()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

    # Filter for anomalies
    threshold = st.sidebar.slider("Anomaly Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    anomalies = df[df["anomaly_score"] >= threshold].copy()
    anomalies = anomalies.sort_values(by="anomaly_score", ascending=False)

    # Executive Metric Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events Analyzed", f"{len(df):,}")
    with col2:
        st.metric("High-Risk Alerts", f"{len(anomalies):,}")
    with col3:
        st.metric("Unique Entities", f"{df['entity_id'].nunique():,}")
    with col4:
        st.metric("Avg Anomaly Score", f"{df['anomaly_score'].mean():.4f}")

    st.markdown("---")
    
    # Ranked Alert Queue & Threat Visualizations
    col_queue, col_viz = st.columns([1, 1])
    
    with col_queue:
        st.subheader("Ranked Alert Queue")
        display_cols = ["timestamp", "entity_id", "source_ip", "anomaly_score", "predicted_attack"]
        st.dataframe(anomalies[display_cols].head(100), use_container_width=True)

    with col_viz:
        st.subheader("Threat Landscape")
        if not anomalies.empty:
            fig = px.scatter(
                anomalies,
                x="timestamp",
                y="anomaly_score",
                color="predicted_attack",
                hover_data=["entity_id", "source_ip", "resource_accessed"],
                title="Anomalies over Time by Attack Class"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No anomalies detected above current threshold.")

    st.markdown("---")
    
    # Entity Deep-Dive Inspector
    st.subheader("Entity Deep-Dive Inspector")
    entity_list = df["entity_id"].unique().tolist()
    selected_entity = st.selectbox("Select Entity ID for Deep-Dive", entity_list)
    
    if selected_entity:
        entity_data = df[df["entity_id"] == selected_entity].sort_values("timestamp")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Total Activity for {selected_entity}:** {len(entity_data)} events")
            st.write(f"**Max Anomaly Score:** {entity_data['anomaly_score'].max():.4f}")
            st.write(f"**Primary Authentication:** {entity_data['auth_method'].mode().iloc[0]}")
            
        with c2:
            st.write(f"**Primary Geolocation:** {entity_data['geo_location'].mode().iloc[0]}")
            st.write(f"**Entity Type:** {entity_data['entity_type'].mode().iloc[0]}")
            st.write(f"**Most Accessed Resource:** {entity_data['resource_accessed'].mode().iloc[0]}")
            
        fig2 = px.line(
            entity_data,
            x="timestamp",
            y="anomaly_score",
            title=f"Anomaly Score Timeline for {selected_entity}",
            markers=True
        )
        st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
