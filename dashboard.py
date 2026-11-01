import streamlit as st
import pandas as pd
import json
import os
import time

st.set_page_config(page_title="FL Experiment Dashboard", layout="wide")
st.title("Federated Learning Live Monitor")

METRICS_FILE = "metrics.json"

auto_refresh = st.sidebar.checkbox("Auto-refresh live", value=True)

if not os.path.exists(METRICS_FILE):
    st.info("Waiting for training to start... Run bridge_server.py or server.py.")
else:
    try:
        with open(METRICS_FILE, "r") as f:
            data = json.load(f)
    except Exception:
        data = []

    if data:
        # Deduplicate rounds in case multiple runs wrote to the file, keeping latest
        df = pd.DataFrame(data).drop_duplicates(subset=["round"], keep="last").set_index("round")
        latest = df.iloc[-1]
        latest_round = int(df.index[-1])

        # Top summary cards
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Round", f"{latest_round}")
        col2.metric("Loss", f"{latest['loss']:.4f}")
        col3.metric("Accuracy", f"{min(100.0, float(latest['accuracy'])):.2f}%")

        # Live training charts
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Aggregated Loss")
            st.line_chart(df["loss"])

        with col_right:
            st.subheader("Aggregated Accuracy (%)")
            st.line_chart(df["accuracy"])

        st.subheader("Round History")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Connecting to coordinator... Waiting for round 1 to complete.")

if auto_refresh:
    time.sleep(1.5)
    st.rerun()
