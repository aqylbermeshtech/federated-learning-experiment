import os
import time
import json
import random
import csv
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Select MPS (Apple Silicon GPU) if available, otherwise CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

SEED = 42
NUM_ROUNDS = 10
CLIP_NORM = 1.0
CONFIGS = [
    {"name": "FedAvg (Baseline)", "sigma": 0.0, "clip": False},
    {"name": "DP-FedAvg (sigma=0.01)", "sigma": 0.01, "clip": True},
    {"name": "DP-FedAvg (sigma=0.05)", "sigma": 0.05, "clip": True},
    {"name": "DP-FedAvg (sigma=0.10)", "sigma": 0.10, "clip": True},
]


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MNISTNet(nn.Module):
    """Same 3-layer MLP architecture used in client.py and export_coreml.py."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


def load_non_iid_datasets():
    """
    Loads MNIST and partitions training data into two extreme Non-IID shards:
      - Client 1: Digits 0-4
      - Client 2: Digits 5-9
    Also creates test loaders for full test set (0-9) and each partition.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)

    c1_train_idx = [i for i, y in enumerate(train_data.targets) if int(y) in {0, 1, 2, 3, 4}]
    c2_train_idx = [i for i, y in enumerate(train_data.targets) if int(y) in {5, 6, 7, 8, 9}]

    p1_test_idx = [i for i, y in enumerate(test_data.targets) if int(y) in {0, 1, 2, 3, 4}]
    p2_test_idx = [i for i, y in enumerate(test_data.targets) if int(y) in {5, 6, 7, 8, 9}]

    train_subsets = {
        1: Subset(train_data, c1_train_idx),
        2: Subset(train_data, c2_train_idx),
    }
    test_loaders = {
        "global": DataLoader(test_data, batch_size=256, shuffle=False),
        "part_0_4": DataLoader(Subset(test_data, p1_test_idx), batch_size=256, shuffle=False),
        "part_5_9": DataLoader(Subset(test_data, p2_test_idx), batch_size=256, shuffle=False),
    }
    return train_subsets, test_loaders


def get_weights(model: nn.Module):
    return [val.detach().cpu().clone() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, weights):
    state_dict = OrderedDict(
        {k: v.clone().to(device) for k, v in zip(model.state_dict().keys(), weights)}
    )
    model.load_state_dict(state_dict, strict=True)


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += len(labels)
    return float(total_loss / total), float((correct / total) * 100.0)


def train_client_one_epoch(
    global_weights,
    train_subset: Subset,
    seed_offset: int,
    clip: bool,
    clip_norm: float,
    sigma: float,
):
    """
    Trains a local client model for 1 epoch on its Non-IID partition,
    then optionally applies L2 update clipping and Gaussian noise injection (DP-FedAvg).
    """
    g = torch.Generator()
    g.manual_seed(SEED + seed_offset)
    loader = DataLoader(train_subset, batch_size=32, shuffle=True, generator=g)

    local_model = MNISTNet().to(device)
    set_weights(local_model, global_weights)
    local_model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(local_model.parameters(), lr=0.002)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = local_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    updated_weights = get_weights(local_model)

    # Compute client update delta: delta_w = w_local - w_global
    deltas = [u - w for u, w in zip(updated_weights, global_weights)]
    total_l2_norm = torch.sqrt(sum(torch.sum(d * d) for d in deltas)).item()

    if clip:
        scale = min(1.0, clip_norm / (total_l2_norm + 1e-12))
        deltas = [d * scale for d in deltas]

    if sigma > 0.0:
        noise_std = sigma * clip_norm
        noisy_deltas = []
        for d in deltas:
            noise = torch.randn(d.shape, dtype=d.dtype) * noise_std
            noisy_deltas.append(d + noise)
        deltas = noisy_deltas

    final_client_weights = [w + d for w, d in zip(global_weights, deltas)]
    return final_client_weights, len(train_subset), total_l2_norm


def run_benchmark():
    os.makedirs("results", exist_ok=True)
    train_subsets, test_loaders = load_non_iid_datasets()
    criterion = nn.CrossEntropyLoss()

    sample_model = MNISTNet()
    payload_bytes = sum(p.numel() * p.element_size() for p in sample_model.parameters())
    payload_kb = round(payload_bytes / 1024.0, 2)
    num_params = sum(p.numel() for p in sample_model.parameters())

    print(f"============================================================")
    print(f"Non-IID MNIST Federated Benchmark (Device: {device})")
    print(f"Model Parameters: {num_params:,} | Client Payload Size: {payload_kb} KB")
    print(f"============================================================")

    all_rows = []
    summary_records = []

    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        sigma = cfg["sigma"]
        clip = cfg["clip"]

        print(f"\n--- Running Configuration: {cfg_name} (sigma={sigma}, clip={clip}) ---")
        set_seed(SEED)

        global_model = MNISTNet().to(device)
        global_weights = get_weights(global_model)

        round_latencies = []

        for rnd in range(1, NUM_ROUNDS + 1):
            t0 = time.perf_counter()

            client_results = []
            for cid in (1, 2):
                c_weights, n_samples, pre_clip_norm = train_client_one_epoch(
                    global_weights=global_weights,
                    train_subset=train_subsets[cid],
                    seed_offset=rnd * 100 + cid,
                    clip=clip,
                    clip_norm=CLIP_NORM,
                    sigma=sigma,
                )
                client_results.append((c_weights, n_samples, pre_clip_norm))

            # FedAvg weighted aggregation
            total_samples = sum(n for _, n, _ in client_results)
            new_global_weights = []
            for param_idx in range(len(global_weights)):
                weighted_sum = sum(
                    c_weights[param_idx] * (n_samples / total_samples)
                    for c_weights, n_samples, _ in client_results
                )
                new_global_weights.append(weighted_sum)

            global_weights = new_global_weights
            set_weights(global_model, global_weights)

            round_latency = time.perf_counter() - t0
            round_latencies.append(round_latency)

            # Evaluate global and per-partition metrics
            g_loss, g_acc = evaluate_model(global_model, test_loaders["global"], criterion)
            _, p1_acc = evaluate_model(global_model, test_loaders["part_0_4"], criterion)
            _, p2_acc = evaluate_model(global_model, test_loaders["part_5_9"], criterion)
            worst_part_acc = min(p1_acc, p2_acc)

            row = {
                "config": cfg_name,
                "sigma": sigma,
                "clip_norm": CLIP_NORM if clip else 0.0,
                "round": rnd,
                "global_loss": round(g_loss, 4),
                "global_accuracy": round(g_acc, 2),
                "part_0_4_accuracy": round(p1_acc, 2),
                "part_5_9_accuracy": round(p2_acc, 2),
                "worst_partition_accuracy": round(worst_part_acc, 2),
                "round_latency_sec": round(round_latency, 3),
                "payload_kb": payload_kb,
            }
            all_rows.append(row)

            print(
                f"Round {rnd:02d}/{NUM_ROUNDS} | "
                f"Loss: {g_loss:.4f} | Global Acc: {g_acc:.2f}% | "
                f"Part(0-4): {p1_acc:.2f}% | Part(5-9): {p2_acc:.2f}% | "
                f"Worst-Part: {worst_part_acc:.2f}% | Latency: {round_latency:.2f}s"
            )

        final_row = all_rows[-1]
        best_acc = max(r["global_accuracy"] for r in all_rows if r["config"] == cfg_name)
        summary_records.append({
            "config": cfg_name,
            "sigma": sigma,
            "clip_norm": CLIP_NORM if clip else 0.0,
            "final_global_accuracy": final_row["global_accuracy"],
            "peak_global_accuracy": best_acc,
            "final_global_loss": final_row["global_loss"],
            "final_part_0_4_accuracy": final_row["part_0_4_accuracy"],
            "final_part_5_9_accuracy": final_row["part_5_9_accuracy"],
            "final_worst_partition_accuracy": final_row["worst_partition_accuracy"],
            "mean_round_latency_sec": round(float(np.mean(round_latencies)), 3),
            "payload_kb": payload_kb,
        })

    # Write CSV
    csv_path = os.path.join("results", "benchmark_metrics.csv")
    fieldnames = [
        "config",
        "sigma",
        "clip_norm",
        "round",
        "global_loss",
        "global_accuracy",
        "part_0_4_accuracy",
        "part_5_9_accuracy",
        "worst_partition_accuracy",
        "round_latency_sec",
        "payload_kb",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Write Summary JSON
    summary_path = os.path.join("results", "summary.json")
    summary_payload = {
        "experiment": "Non-IID MNIST Privacy-Utility Trade-off (FedAvg vs. DP-FedAvg)",
        "seed": SEED,
        "num_rounds": NUM_ROUNDS,
        "num_clients": 2,
        "partition_scheme": "Extreme Non-IID (Client 1: digits 0-4, Client 2: digits 5-9)",
        "device": str(device),
        "model_parameters": num_params,
        "client_payload_kb": payload_kb,
        "configurations": summary_records,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\nSaved raw round metrics to {csv_path}")
    print(f"Saved experiment summary to {summary_path}")


if __name__ == "__main__":
    run_benchmark()
