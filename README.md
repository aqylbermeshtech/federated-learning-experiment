# Federated Learning Edge Experiment (PyTorch + Core ML + SwiftUI)

An end-to-end Federated Learning testbed demonstrating privacy-preserving distributed training across Python clients and native iOS edge devices on Non-IID MNIST data.

## Architecture
- **Flower Simulation & Server:** Multi-client Non-IID MNIST training (0-4 vs 5-9 partitions) achieving >85% global accuracy via FedAvg without sharing raw image data (`server.py` + `client.py`).
- **FastAPI Aggregation Bridge:** Lightweight HTTP coordinator (`bridge_server.py`) that collects weight updates from heterogeneous edge clients, performs FedAvg, and logs live metrics.
- **Python Edge Client:** Headless edge client (`bridge_client.py`) that can simulate local updates and sync with the bridge coordinator.
- **Streamlit Live Dashboard:** Real-time monitoring (`dashboard.py`) of global loss and accuracy across federated rounds.
- **iOS Edge Client (SwiftUI + Core ML):** On-device neural network training using `MLUpdateTask` on the Apple Neural Engine/GPU, complete with an interactive 28x28 digit drawing canvas, configurable coordinator URL, and local weight persistence.

## Quick Start

### 1. Start the Monitoring Dashboard

    streamlit run dashboard.py

### 2. Run the FastAPI Weight Bridge

    python bridge_server.py

*(By default, `bridge_server.py` runs with `MIN_CLIENTS=1` for single-client dev testing. Run `MIN_CLIENTS=2 python bridge_server.py` to require multi-client consensus before FedAvg).*

### 3. Run the iOS Edge Client
Open `EdgeFLClient/EdgeFLClient.xcodeproj` in Xcode, run on an iOS Simulator or physical device. If testing on a physical iPhone on local Wi-Fi, enter your Mac's LAN IP (e.g. `http://192.168.1.50:8000`) into the Coordinator URL field, sketch a digit on the canvas, select its label, and tap **Train & Submit**.

### 4. Optional: Run the Python Bridge Client
To test multi-client federated training without needing a second device:

    python bridge_client.py
