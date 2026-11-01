import sys
import time
import requests
import numpy as np

SERVER_URL = "http://127.0.0.1:8000"

def run_client(client_id: str = "python_client_1", num_rounds: int = 5):
    print(f"Starting Bridge FL Client '{client_id}' connecting to {SERVER_URL}...")
    
    # 640 weights corresponding to dense_3 (10 classes x 64 features)
    weights = np.random.randn(640).astype(np.float32) * 0.05
    
    for r in range(1, num_rounds + 1):
        print(f"\n[Round {r}] Simulating local training on client partition...")
        time.sleep(1.0)
        # Apply simulated gradient update
        gradient_step = np.random.randn(640).astype(np.float32) * 0.01
        weights = weights - gradient_step

        payload = {
            "client_id": client_id,
            "round": r,
            "num_samples": 32,
            "weights": weights.tolist()
        }

        try:
            res = requests.post(f"{SERVER_URL}/submit_weights", json=payload)
            res.raise_for_status()
            data = res.json()
            status = data.get("status")
            print(f"--> Server response: {status}")
            if status == "aggregated":
                print(f"    Round {data.get('completed_round')} aggregated successfully!")
                print(f"    New Round: {data.get('new_round')} | Loss: {data.get('loss')} | Acc: {data.get('accuracy')}%")
                if "global_weights" in data and data["global_weights"]:
                    weights = np.array(data["global_weights"], dtype=np.float32)
            elif status == "waiting_for_more_clients":
                print(f"    Waiting for other clients ({data.get('current_pool')}/{data.get('required_clients')})...")
        except Exception as e:
            print(f"Error communicating with bridge server: {e}")
            break

if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else "python_client_1"
    run_client(client_id=cid)

