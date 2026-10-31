import sys
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Simple neural network for 28x28 grayscale images
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x)

# 2. Prepare Non-IID MNIST data
def load_partitioned_mnist(client_id: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Downloads dataset to a local ./data folder automatically
    train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)

    # Client 1 gets digits 0-4; Client 2 gets digits 5-9
    if client_id == 1:
        target_digits = {0, 1, 2, 3, 4}
    else:
        target_digits = {5, 6, 7, 8, 9}

    train_indices = [i for i, label in enumerate(train_data.targets) if label.item() in target_digits]
    
    train_loader = DataLoader(Subset(train_data, train_indices), batch_size=32, shuffle=True)
    # Both clients test against all digits 0-9 to measure global accuracy
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    return train_loader, test_loader

# 3. Client Implementation
class MNISTClient(fl.client.NumPyClient):
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.model = MNISTNet().to(device)
        self.train_loader, self.test_loader = load_partitioned_mnist(client_id)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.002)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        # Train 1 local epoch per federated round
        for images, labels in self.train_loader:
            images, labels = images.to(device), labels.to(device)
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()

        print(f"[Client {self.client_id}] Finished local training pass.")
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item() * len(labels)
                correct += (outputs.argmax(1) == labels).sum().item()
        
        total = len(self.test_loader.dataset)
        accuracy = float(correct / total)
        return float(loss / total), total, {"accuracy": accuracy}

if __name__ == "__main__":
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"--- Client {cid} initialized (Partition: {'Digits 0-4' if cid == 1 else 'Digits 5-9'}) ---")
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=MNISTClient(cid).to_client(),
    )
