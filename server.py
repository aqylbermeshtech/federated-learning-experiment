import flwr as fl
import json
import os

METRICS_FILE = "metrics.json"

if os.path.exists(METRICS_FILE):
    os.remove(METRICS_FILE)

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

class LiveMetricsStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self, server_round, results, failures):
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        if loss_aggregated is not None:
            accuracy = float(metrics_aggregated.get("accuracy", 0.0)) if metrics_aggregated else 0.0
            record = {
                "round": server_round,
                "loss": round(float(loss_aggregated), 4),
                "accuracy": round(accuracy * 100, 2)
            }
            
            data = []
            if os.path.exists(METRICS_FILE):
                try:
                    with open(METRICS_FILE, "r") as f:
                        data = json.load(f)
                except Exception:
                    data = []
            
            data.append(record)
            with open(METRICS_FILE, "w") as f:
                json.dump(data, f, indent=2)

        return loss_aggregated, metrics_aggregated

strategy = LiveMetricsStrategy(
    fraction_fit=1.0,
    min_fit_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
)

print("--- Flower Server is listening on port 8080 ---")
fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)
