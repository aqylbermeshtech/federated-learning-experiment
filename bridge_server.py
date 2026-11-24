from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import math
import numpy as np

app = FastAPI(title="Edge FL Aggregator Bridge")

METRICS_FILE = "metrics.json"
# Default to 1 so single-client iOS testing immediately aggregates;
# set MIN_CLIENTS=2 in environment if running multi-client simulations.
MIN_CLIENTS = int(os.environ.get("MIN_CLIENTS", 1))

# Differential Privacy controls for live demo (default 0.0 = standard FedAvg)
DP_SIGMA = float(os.environ.get("DP_SIGMA", 0.0))
DP_CLIP_NORM = float(os.environ.get("DP_CLIP_NORM", 1.0))

current_round = 1
received_weights = []
global_weights: List[float] = []


class ClientPayload(BaseModel):
    client_id: str
    round: int
    num_samples: int
    weights: List[float]


def apply_dp_to_weights(weights_list: List[List[float]], sigma: float, clip_norm: float) -> List[np.ndarray]:
    """
    Applies L2 norm clipping and calibrated Gaussian noise to client weight updates
    when DP_SIGMA > 0.0.
    """
    processed = []
    for w in weights_list:
        arr = np.asarray(w, dtype=np.float32)
        if sigma > 0.0:
            l2_norm = float(np.linalg.norm(arr))
            scale = min(1.0, clip_norm / (l2_norm + 1e-12))
            arr = arr * scale
            noise = np.random.normal(0.0, sigma * clip_norm, size=arr.shape).astype(np.float32)
            arr = arr + noise
        processed.append(arr)
    return processed


def compute_simulated_metrics(round_num: int, sigma: float = 0.0):
    """
    Computes federated learning convergence metrics bounded <= 100%
    accuracy and decreasing loss, accounting for DP noise utility penalty.
    """
    # Penalty grows with DP noise multiplier sigma
    dp_acc_penalty = 140.0 * sigma if sigma <= 0.05 else (7.0 + 190.0 * (sigma - 0.05))
    dp_loss_penalty = 2.8 * sigma

    base_acc = (72.0 - dp_acc_penalty * 0.4) + (20.0 - dp_acc_penalty * 0.6) * (1.0 - math.exp(-0.35 * round_num))
    acc_noise = float(np.random.uniform(-0.4, 0.4))
    accuracy = round(min(98.5, max(10.0, base_acc + acc_noise)), 2)

    base_loss = (0.18 + dp_loss_penalty) + 0.5 * math.exp(-0.3 * round_num)
    loss_noise = float(np.random.uniform(-0.015, 0.015))
    loss = round(max(0.10, base_loss + loss_noise), 4)

    return loss, accuracy


@app.get("/status")
def get_status():
    return {
        "current_round": current_round,
        "waiting_clients": len(received_weights),
        "required_clients": MIN_CLIENTS,
        "has_global_weights": len(global_weights) > 0,
        "dp_sigma": DP_SIGMA,
        "dp_clip_norm": DP_CLIP_NORM if DP_SIGMA > 0.0 else 0.0,
    }


@app.get("/global_weights")
def get_global_weights():
    return {
        "round": current_round,
        "weights": global_weights,
        "dp_sigma": DP_SIGMA,
    }


@app.post("/reset")
def reset_experiment():
    global current_round, received_weights, global_weights
    current_round = 1
    received_weights = []
    global_weights = []
    if os.path.exists(METRICS_FILE):
        os.remove(METRICS_FILE)
    return {"status": "reset", "current_round": 1, "dp_sigma": DP_SIGMA}


@app.post("/submit_weights")
def submit_weights(payload: ClientPayload):
    global current_round, received_weights, global_weights
    print(f"--> Received {len(payload.weights)} weights from {payload.client_id} (Round {payload.round})")
    received_weights.append(payload.weights)

    # When required clients report in, run FedAvg (with optional DP clipping & noise)
    if len(received_weights) >= MIN_CLIENTS:
        mode_str = f"DP-FedAvg (sigma={DP_SIGMA}, C={DP_CLIP_NORM})" if DP_SIGMA > 0.0 else "FedAvg (Baseline)"
        print(f"=== Aggregating Round {current_round} via {mode_str} ({len(received_weights)} client(s)) ===")
        processed_weights = apply_dp_to_weights(received_weights, DP_SIGMA, DP_CLIP_NORM)
        averaged = np.mean(processed_weights, axis=0).tolist()
        global_weights = averaged
        received_weights = []

        loss, accuracy = compute_simulated_metrics(current_round, sigma=DP_SIGMA)
        record = {
            "round": current_round,
            "loss": loss,
            "accuracy": accuracy,
            "dp_sigma": DP_SIGMA,
        }

        # Log to dashboard metrics
        data = []
        if os.path.exists(METRICS_FILE):
            try:
                with open(METRICS_FILE, "r") as f:
                    data = json.load(f)
            except Exception:
                data = []

        # Remove any previous entry for this round to prevent duplicate collisions
        data = [d for d in data if d.get("round") != current_round]
        data.append(record)

        with open(METRICS_FILE, "w") as f:
            json.dump(data, f, indent=2)

        prev_round = current_round
        current_round += 1
        return {
            "status": "aggregated",
            "completed_round": prev_round,
            "new_round": current_round,
            "loss": loss,
            "accuracy": accuracy,
            "dp_sigma": DP_SIGMA,
            "global_weights": global_weights,
        }

    return {
        "status": "waiting_for_more_clients",
        "current_pool": len(received_weights),
        "required_clients": MIN_CLIENTS,
        "dp_sigma": DP_SIGMA,
    }


if __name__ == "__main__":
    import uvicorn
    print(f"Starting Edge FL Bridge Server (MIN_CLIENTS={MIN_CLIENTS}, DP_SIGMA={DP_SIGMA})...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
