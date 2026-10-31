import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset

# Select MPS (Mac GPU) if available, otherwise CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Define a simple classifier
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        return self.fc(x)

# 2. Define the Flower Client
class FLClient(fl.client.NumPyClient):
    def __init__(self, cid: str, loader: DataLoader):
        self.cid = cid
        self.loader = loader
        self.model = Net().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):
            for x, y in self.loader:
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config={}), len(self.loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in self.loader:
                x, y = x.to(device), y.to(device)
                out = self.model(x)
                loss += self.criterion(out, y).item() * len(y)
                correct += (out.argmax(1) == y).sum().item()
        total = len(self.loader.dataset)
        return float(loss / total), total, {"accuracy": float(correct / total)}

# 3. Create synthetic partitioned data for 3 simulated clients
def get_data_loaders(num_clients=3):
    loaders = []
    for _ in range(num_clients):
        x = torch.randn(150, 8)
        y = (x.sum(dim=1) > 0).long()
        loaders.append(DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True))
    return loaders

loaders = get_data_loaders(3)

def client_fn(cid: str) -> fl.client.Client:
    return FLClient(cid=cid, loader=loaders[int(cid)]).to_client()

# 4. Run the simulation
if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
    )

    print(f"Starting Flower simulation on {device}...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
