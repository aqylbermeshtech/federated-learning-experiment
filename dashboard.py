import streamlit as st
import pandas as pd
import json
import os
import time

st.set_page_config(page_title="FL Experiment Dashboard", layout="wide")
st.title("Federated Learning Live Monitor & DP Benchmark")

METRICS_FILE = "metrics.json"
CONVERGENCE_PLOT = os.path.join("results", "convergence_comparison.png")
TRADEOFF_PLOT = os.path.join("results", "privacy_utility_tradeoff.png")
SUMMARY_JSON = os.path.join("results", "summary.json")

auto_refresh = st.sidebar.checkbox("Auto-refresh live", value=True)

tab_live, tab_bench = st.tabs([
    "Live Federated Training Monitor",
    "Benchmark Comparison (FedAvg vs. DP-FedAvg)",
])

with tab_live:
    if not os.path.exists(METRICS_FILE):
        st.info("Waiting for training to start... Run bridge_server.py or server.py.")
    else:
        try:
            with open(METRICS_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            data = []

        if data:
            df = pd.DataFrame(data).drop_duplicates(subset=["round"], keep="last").set_index("round")
            latest = df.iloc[-1]
            latest_round = int(df.index[-1])
            dp_sigma = float(latest.get("dp_sigma", 0.0)) if "dp_sigma" in latest else 0.0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Round", f"{latest_round}")
            col2.metric("Loss", f"{latest['loss']:.4f}")
            col3.metric("Accuracy", f"{min(100.0, float(latest['accuracy'])):.2f}%")
            col4.metric("Privacy Noise (DP σ)", f"{dp_sigma:.2f}" if dp_sigma > 0 else "0.00 (Plain FedAvg)")

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

with tab_bench:
    st.subheader("Controlled R&D Benchmark: Privacy–Utility Trade-off Under Extreme Non-IID MNIST")
    st.markdown(
        "**Experimental Setup:** 2 clients with disjoint label distributions "
        "(Client 1: digits `0–4`, Client 2: digits `5–9`), 10 communication rounds, fixed seed `42`, "
        "comparing Baseline FedAvg against Client-Side DP-FedAvg ($L_2$ clip norm $C=1.0$, $\\sigma \\in \\{0.01, 0.05, 0.10\\}$)."
    )

    if os.path.exists(CONVERGENCE_PLOT):
        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            st.image(CONVERGENCE_PLOT, caption="Global Convergence Across Noise Multipliers", use_container_width=True)
        with col_fig2:
            if os.path.exists(TRADEOFF_PLOT):
                st.image(TRADEOFF_PLOT, caption="Privacy-Utility Trade-off (Global vs. Worst-Partition Accuracy)", use_container_width=True)
    else:
        st.warning("Benchmark figures not found yet. Run `python run_benchmark.py && python plot_results.py`.")

    if os.path.exists(SUMMARY_JSON):
        try:
            with open(SUMMARY_JSON, "r") as f:
                summary_data = json.load(f)
            configs_df = pd.DataFrame(summary_data.get("configurations", []))
            if not configs_df.empty:
                st.subheader("Quantitative Summary Table")
                st.dataframe(configs_df, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load benchmark summary: {e}")

if auto_refresh:
    time.sleep(1.5)
    st.rerun()
