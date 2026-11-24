import os
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results():
    csv_path = os.path.join("results", "benchmark_metrics.csv")
    summary_path = os.path.join("results", "summary.json")

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "config": r["config"],
                "sigma": float(r["sigma"]),
                "clip_norm": float(r["clip_norm"]),
                "round": int(r["round"]),
                "global_loss": float(r["global_loss"]),
                "global_accuracy": float(r["global_accuracy"]),
                "part_0_4_accuracy": float(r["part_0_4_accuracy"]),
                "part_5_9_accuracy": float(r["part_5_9_accuracy"]),
                "worst_partition_accuracy": float(r["worst_partition_accuracy"]),
                "round_latency_sec": float(r["round_latency_sec"]),
                "payload_kb": float(r["payload_kb"]),
            })

    with open(summary_path, "r") as f:
        summary = json.load(f)

    return rows, summary


def plot_convergence(rows):
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
    })

    configs = []
    for r in rows:
        if r["config"] not in configs:
            configs.append(r["config"])

    styles = {
        "FedAvg (Baseline)": {"color": "#1f77b4", "marker": "o", "linestyle": "-", "linewidth": 2.4},
        "DP-FedAvg (sigma=0.01)": {"color": "#2ca02c", "marker": "s", "linestyle": "--", "linewidth": 2.0},
        "DP-FedAvg (sigma=0.05)": {"color": "#ff7f0e", "marker": "^", "linestyle": "-.", "linewidth": 2.0},
        "DP-FedAvg (sigma=0.10)": {"color": "#d62728", "marker": "D", "linestyle": ":", "linewidth": 2.0},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

    for cfg in configs:
        cfg_rows = [r for r in rows if r["config"] == cfg]
        rounds = [r["round"] for r in cfg_rows]
        accs = [r["global_accuracy"] for r in cfg_rows]
        losses = [r["global_loss"] for r in cfg_rows]
        st = styles.get(cfg, {"marker": "o", "linewidth": 2.0})

        ax1.plot(rounds, accs, label=cfg, **st)
        ax2.plot(rounds, losses, label=cfg, **st)

    ax1.set_title("Global Test Accuracy vs. Federated Rounds (Non-IID MNIST)")
    ax1.set_xlabel("Federated Communication Round")
    ax1.set_ylabel("Global Test Accuracy (%)")
    ax1.set_xticks(range(1, 11))
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", frameon=True)

    ax2.set_title("Global Cross-Entropy Loss vs. Federated Rounds")
    ax2.set_xlabel("Federated Communication Round")
    ax2.set_ylabel("Global Test Loss")
    ax2.set_xticks(range(1, 11))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    out_path = os.path.join("results", "convergence_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved convergence plot to {out_path}")


def plot_privacy_utility_tradeoff(summary):
    configs = summary["configurations"]
    labels = [
        "Baseline\n(σ=0.0, No Clip)" if c["sigma"] == 0.0 else f"DP-FedAvg\n(σ={c['sigma']:.2f}, C=1.0)"
        for c in configs
    ]
    global_accs = [c["final_global_accuracy"] for c in configs]
    worst_accs = [c["final_worst_partition_accuracy"] for c in configs]
    sigmas = [c["sigma"] for c in configs]

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

    x_indices = range(len(configs))
    ax.plot(
        x_indices,
        global_accs,
        marker="o",
        color="#1f77b4",
        linewidth=2.5,
        markersize=8,
        label="Final Global Test Accuracy (Digits 0–9)",
    )
    ax.plot(
        x_indices,
        worst_accs,
        marker="s",
        color="#d62728",
        linewidth=2.5,
        linestyle="--",
        markersize=8,
        label="Worst-Partition Test Accuracy (min of 0–4 vs. 5–9)",
    )

    for i, (g_val, w_val) in enumerate(zip(global_accs, worst_accs)):
        ax.annotate(
            f"{g_val:.2f}%",
            (i, g_val),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=9.5,
            fontweight="bold",
            color="#1f77b4",
        )
        ax.annotate(
            f"{w_val:.2f}%",
            (i, w_val),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=9.5,
            fontweight="bold",
            color="#d62728",
        )

    ax.set_title("Privacy–Utility Trade-off Under Extreme Non-IID Partitioning (Round 10)")
    ax.set_xlabel("Method & DP Noise Multiplier (σ)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xticks(list(x_indices))
    ax.set_xticklabels(labels)
    y_min = max(0.0, min(worst_accs) - 10.0)
    y_max = min(102.0, max(global_accs) + 6.0)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    out_path = os.path.join("results", "privacy_utility_tradeoff.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved privacy-utility trade-off plot to {out_path}")


def print_markdown_table(summary):
    print("\n### Benchmark Results Comparison Table (10 Rounds, Extreme Non-IID MNIST)\n")
    header = (
        "| Configuration | Noise Multiplier ($\\sigma$) | Clip Norm ($C$) | "
        "Final Global Acc (%) | Peak Global Acc (%) | Worst-Partition Acc (%) | "
        "Final Global Loss | Avg Round Latency (s) | Client Payload (KB) |"
    )
    sep = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    print(header)
    print(sep)
    for c in summary["configurations"]:
        clip_str = f"{c['clip_norm']:.1f}" if c["clip_norm"] > 0 else "None"
        print(
            f"| **{c['config']}** | {c['sigma']:.2f} | {clip_str} | "
            f"{c['final_global_accuracy']:.2f}% | {c['peak_global_accuracy']:.2f}% | "
            f"{c['final_worst_partition_accuracy']:.2f}% | {c['final_global_loss']:.4f} | "
            f"{c['mean_round_latency_sec']:.2f}s | {c['payload_kb']:.2f} KB |"
        )
    print()


if __name__ == "__main__":
    rows, summary = load_results()
    plot_convergence(rows)
    plot_privacy_utility_tradeoff(summary)
    print_markdown_table(summary)
