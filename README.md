# Federated Learning Edge Experiment (PyTorch + Core ML + SwiftUI)

An end-to-end Federated Learning testbed demonstrating privacy-preserving distributed training across Python clients and native iOS edge devices on Non-IID MNIST data.

## Architecture
- **Flower Simulation & Server:** Multi-client Non-IID MNIST training (0-4 vs 5-9 partitions) achieving >85% global accuracy via FedAvg without sharing raw image data.
- **FastAPI Aggregation Bridge:** Lightweight HTTP coordinator (bridge_server.py) that collects weight updates from heterogeneous clients and logs live metrics.
- **Streamlit Live Dashboard:** Real-time monitoring (dashboard.py) of global loss and accuracy across federated rounds.
- **iOS Edge Client (SwiftUI + Core ML):** On-device neural network training using MLUpdateTask on the Apple Neural Engine/GPU, complete with an interactive 28x28 digit drawing canvas and local weight persistence.

## Quick Start

### 1. Start the Monitoring Dashboard

    streamlit run dashboard.py

### 2. Run the FastAPI Weight Bridge

    python bridge_server.py

### 3. Run the iOS Edge Client
Open EdgeFLClient/EdgeFLClient.xcodeproj in Xcode, run on an iOS Simulator or physical device, sketch a digit on the canvas, select its label, and tap Train & Submit.
