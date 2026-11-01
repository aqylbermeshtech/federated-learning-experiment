import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder, SgdParams
from coremltools.models import datatypes

# 1. Architecture matching the Core ML model
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def train_mnist_model():
    print("--> Loading MNIST dataset from ./data (values in [0.0, 1.0])...")
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--> Training on device: {device}")

    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += len(labels)

        train_acc = (correct / total) * 100.0
        avg_loss = total_loss / total
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

    # Evaluate on test set
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_correct += (outputs.argmax(1) == labels).sum().item()
            test_total += len(labels)
    test_acc = (test_correct / test_total) * 100.0
    print(f"==> Final MNIST Test Accuracy: {test_acc:.2f}%")

    # Extract weights
    cpu_model = model.cpu()
    W1 = cpu_model.fc1.weight.detach().numpy().astype(np.float32)  # shape (128, 784)
    b1 = cpu_model.fc1.bias.detach().numpy().astype(np.float32)    # shape (128,)

    W2 = cpu_model.fc2.weight.detach().numpy().astype(np.float32)  # shape (64, 128)
    b2 = cpu_model.fc2.bias.detach().numpy().astype(np.float32)    # shape (64,)

    W3 = cpu_model.fc3.weight.detach().numpy().astype(np.float32)  # shape (10, 64)
    b3 = cpu_model.fc3.bias.detach().numpy().astype(np.float32)    # shape (10,)

    return W1, b1, W2, b2, W3, b3

def export_updatable_coreml(W1, b1, W2, b2, W3, b3):
    print("--> Exporting updatable Core ML model with pre-trained weights...")
    input_features = [('image', datatypes.Array(1, 28, 28))]
    output_features = [('probs', datatypes.Array(10))]

    builder = NeuralNetworkBuilder(
        input_features=input_features,
        output_features=output_features,
        mode=None
    )

    # Layer 1: Dense 784 -> 128 + ReLU
    builder.add_inner_product(
        name="dense_1",
        W=W1, b=b1,
        input_channels=784, output_channels=128,
        has_bias=True,
        input_name="image",
        output_name="dense_1_out"
    )
    builder.add_activation(
        name="relu_1",
        non_linearity="RELU",
        input_name="dense_1_out",
        output_name="relu_1_out"
    )

    # Layer 2: Dense 128 -> 64 + ReLU
    builder.add_inner_product(
        name="dense_2",
        W=W2, b=b2,
        input_channels=128, output_channels=64,
        has_bias=True,
        input_name="relu_1_out",
        output_name="dense_2_out"
    )
    builder.add_activation(
        name="relu_2",
        non_linearity="RELU",
        input_name="dense_2_out",
        output_name="relu_2_out"
    )

    # Layer 3: Dense 64 -> 10 (Updatable classification head)
    builder.add_inner_product(
        name="dense_3",
        W=W3, b=b3,
        input_channels=64, output_channels=10,
        has_bias=True,
        input_name="relu_2_out",
        output_name="logits"
    )

    # Softmax Layer
    builder.add_softmax(
        name="softmax_layer",
        input_name="logits",
        output_name="probs"
    )

    # Mark the final classification layer as updatable
    builder.make_updatable(["dense_3"])
    builder.set_categorical_cross_entropy_loss(name="lossLayer", input="probs")
    builder.set_sgd_optimizer(SgdParams(lr=0.01, batch=32))
    builder.set_epochs(1)

    builder.spec.isUpdatable = True
    builder.spec.description.trainingInput[1].shortDescription = "Target integer label for training"

    model = ct.models.MLModel(builder.spec)
    root_model_path = "MNISTUpdatable.mlmodel"
    ios_model_path = "EdgeFLClient/EdgeFLClient/MNISTUpdatable.mlmodel"

    model.save(root_model_path)
    shutil.copy(root_model_path, ios_model_path)
    print(f"--> Saved pre-trained updatable model to {root_model_path} and {ios_model_path} successfully!")

if __name__ == "__main__":
    W1, b1, W2, b2, W3, b3 = train_mnist_model()
    export_updatable_coreml(W1, b1, W2, b2, W3, b3)

