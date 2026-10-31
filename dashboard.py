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
    st.info("Waiting for training to start... Run server.py and your clients.")
else:
    try:
        with open(METRICS_FILE, "r") as f:
            data = json.load(f)
    except Exception:
        data = []

    if data:
        df = pd.DataFrame(data).set_index("round")
        latest = data[-1]

        # Top summary cards
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Round", f"{latest['round']} / 5")
        col2.metric("Loss", f"{latest['loss']}")
        col3.metric("Accuracy", f"{latest['accuracy']}%")

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
