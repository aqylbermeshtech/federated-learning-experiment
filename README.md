# Federated Learning Edge Experiment (PyTorch + Core ML + SwiftUI)

An end-to-end Federated Learning testbed and TRL 3–4 research proof-of-concept demonstrating privacy-preserving distributed training across Python clients and native iOS edge devices on extreme Non-IID MNIST data.

## Architecture
- **Controlled R&D Benchmark (`run_benchmark.py` & `plot_results.py`):** Reproducible evaluation of the **privacy–utility trade-off under extreme Non-IID data** (disjoint label partitions `0–4` vs. `5–9`), comparing standard **FedAvg** against client-side **Differentially Private FedAvg (DP-FedAvg)** with $L_2$ update clipping and calibrated Gaussian noise.
- **Flower Simulation & Server (`server.py` & `client.py`):** Multi-client Non-IID MNIST training (0–4 vs. 5–9 partitions) achieving >85% global accuracy via FedAvg without sharing raw image data.
- **FastAPI Aggregation Bridge (`bridge_server.py`):** Lightweight HTTP coordinator that collects weight updates from heterogeneous edge clients, performs FedAvg (with optional live Differential Privacy noise injection via `DP_SIGMA`), and logs live metrics.
- **Python Edge Client (`bridge_client.py`):** Headless edge client that simulates local updates and syncs with the bridge coordinator.
- **Streamlit Live Dashboard (`dashboard.py`):** Real-time monitoring of global loss, accuracy, and privacy noise across federated rounds, paired with a dedicated **Benchmark Comparison** tab displaying empirical convergence and privacy–utility trade-off figures.
- **iOS Edge Client (`EdgeFLClient` — SwiftUI + Core ML):** On-device neural network training using `MLUpdateTask` on the Apple Neural Engine/GPU, complete with an interactive 28x28 digit drawing canvas, configurable coordinator URL, and local weight persistence.

---

## Experimental Evaluation & Reproducibility

### 1. Research Hypothesis & Technical Uncertainty
When edge clients hold **extreme Non-IID data** (e.g., Client 1 only observes digits `0–4` while Client 2 only observes digits `5–9`), local gradient updates diverge significantly in direction and magnitude. Adding client-side Differential Privacy ($L_2$ sensitivity clipping + Gaussian noise injection) protects against gradient inversion and membership inference attacks, but introduces a critical technical uncertainty: **how severely does client-side DP noise degrade global convergence and minority/worst-partition accuracy under disjoint label distributions?**

- **Baseline Method:** Standard Federated Averaging (**FedAvg**, $\sigma = 0.0$, no norm clipping).
- **Proposed Method:** Client-side **DP-FedAvg** with $L_2$ weight-update clipping ($C = 1.0$) and calibrated Gaussian noise injection $\mathcal{N}(0, (\sigma \cdot C)^2 \mathbf{I})$ evaluated across noise multipliers $\sigma \in \{0.01, 0.05, 0.10\}$.

### 2. Experimental Setup & Hardware
- **Dataset & Partitioning:** MNIST (60,000 train / 10,000 test images), partitioned into two completely disjoint Non-IID shards:
  - **Client 1 Partition:** Digits `0, 1, 2, 3, 4` (30,596 training samples)
  - **Client 2 Partition:** Digits `5, 6, 7, 8, 9` (29,404 training samples)
- **Model Architecture:** 3-layer MLP (`784 -> 128 -> 64 -> 10` with ReLU activations), totaling **109,386 parameters** (**427.29 KB** float32 weight payload per client per round).
- **Training Hyperparameters:** $N = 10$ federated communication rounds, 1 local epoch per round, batch size $32$, Adam optimizer ($\text{lr} = 0.002$), fixed random seed `seed = 42` across all configurations.
- **Hardware & Runtime Environment:** Apple Silicon GPU (`mps` backend), PyTorch 2.8.0, Python 3.9 (`fl_env`).

### 3. Empirical Results (10 Federated Rounds)

| Configuration | Noise Multiplier ($\sigma$) | Clip Norm ($C$) | Final Global Acc (%) | Peak Global Acc (%) | Part (0–4) Acc (%) | Part (5–9) Acc (%) | Worst-Partition Acc (%) | Final Global Loss | Avg Round Latency (s) | Client Payload (KB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FedAvg (Baseline)** | 0.00 | None | **85.48%** | **87.93%** | 95.15% | 75.25% | **75.25%** | **0.4873** | 15.53s | 427.29 KB |
| **DP-FedAvg ($\sigma=0.01$)** | 0.01 | 1.0 | **77.74%** | **77.74%** | 94.45% | 60.07% | **60.07%** | **1.3819** | 15.23s | 427.29 KB |
| **DP-FedAvg ($\sigma=0.05$)** | 0.05 | 1.0 | **28.81%** | **33.05%** | 38.22% | 18.86% | **18.86%** | **1.9989** | 15.21s | 427.29 KB |
| **DP-FedAvg ($\sigma=0.10$)** | 0.10 | 1.0 | **14.84%** | **16.37%** | 14.89% | 14.79% | **14.79%** | **13.4373** | 15.21s | 427.29 KB |

#### Key Findings
1. **Low-Noise Regime ($\sigma = 0.01$, $C = 1.0$):** Preserves strong global utility (**77.74%** final accuracy vs. **85.48%** baseline, a **7.74 percentage point** trade-off) while bounding every client's update sensitivity to $C = 1.0$. However, Non-IID disparity widens: partition `0–4` reaches **94.45%**, whereas the harder partition `5–9` drops from **75.25%** to **60.07%**.
2. **Medium & High Noise Regimes ($\sigma \ge 0.05$):** Because noise is injected per parameter across all $d = 109,386$ weights, the expected $L_2$ norm of the Gaussian noise vector scales as $\sqrt{d} \cdot \sigma C \approx 330.7 \cdot \sigma$. At $\sigma = 0.05$ and $\sigma = 0.10$, noise overwhelms the clipped signal norm ($\le 1.0$), causing severe accuracy degradation (**28.81%** and **14.84%**, respectively). This empirically motivates parameter-efficient updates (such as updating only the 640-parameter classification head `dense_3` as implemented in our Core ML iOS client) to reduce ambient dimension $d$ under strict DP budgets.
3. **System Overhead:** DP clipping and noise generation add negligible runtime latency (~15.2s per round across all runs) and zero communication overhead (**427.29 KB** per client update).

### 4. Reproducing the Benchmark & Figures
Run the following commands to execute the 40-round benchmark from scratch and regenerate all CSV/JSON artifacts and 300 DPI plots in `results/`:

```bash
# 1. Run the controlled Non-IID benchmark (saves results/benchmark_metrics.csv and results/summary.json)
/Users/nurtore/fl_env/bin/python run_benchmark.py

# 2. Generate publication-ready figures (results/convergence_comparison.png and results/privacy_utility_tradeoff.png)
/Users/nurtore/fl_env/bin/python plot_results.py
```

---

## Quick Start (Live Demo & iOS Client)

### 1. Start the Monitoring Dashboard
```bash
/Users/nurtore/fl_env/bin/streamlit run dashboard.py
```
*(Switch between the **Live Federated Training Monitor** tab and the **Benchmark Comparison** tab in the UI).*

### 2. Run the FastAPI Weight Bridge
```bash
# Standard FedAvg (default DP_SIGMA=0.0)
/Users/nurtore/fl_env/bin/python bridge_server.py

# Or enable live Differential Privacy noise during the demo (e.g. DP_SIGMA=0.01 or 0.05)
DP_SIGMA=0.01 /Users/nurtore/fl_env/bin/python bridge_server.py
```
*(By default, `bridge_server.py` runs with `MIN_CLIENTS=1` for single-client dev testing. Run `MIN_CLIENTS=2 DP_SIGMA=0.01 /Users/nurtore/fl_env/bin/python bridge_server.py` to require multi-client consensus before DP-FedAvg aggregation).*

### 3. Run the iOS Edge Client
Open `EdgeFLClient/EdgeFLClient.xcodeproj` in Xcode, run on an iOS Simulator or physical device. If testing on a physical iPhone on local Wi-Fi, tap **Device (`192.168.100.187`)** or enter your Mac's LAN IP into the Coordinator URL field, sketch a digit on the canvas, select its label, and tap **Train & Submit**.

### 4. Optional: Run the Python Bridge Client
To test multi-client federated training without needing a second physical device:
```bash
/Users/nurtore/fl_env/bin/python bridge_client.py
```
