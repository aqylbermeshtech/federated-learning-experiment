import numpy as np
import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder, SgdParams
from coremltools.models import datatypes

# 1. Define model input and output features
# Input: Grayscale image tensor of shape [1, 28, 28] (784 floats)
input_features = [('image', datatypes.Array(1, 28, 28))]
output_features = [('probs', datatypes.Array(10))]

# 2. Build the Neural Network specification directly
builder = NeuralNetworkBuilder(
    input_features=input_features,
    output_features=output_features,
    mode=None
)

# Initialize small random weights matching our PyTorch architecture
np.random.seed(42)
W1 = np.random.randn(128, 784).astype(np.float32) * 0.05
b1 = np.zeros(128, dtype=np.float32)

W2 = np.random.randn(64, 128).astype(np.float32) * 0.05
b2 = np.zeros(64, dtype=np.float32)

W3 = np.random.randn(10, 64).astype(np.float32) * 0.05
b3 = np.zeros(10, dtype=np.float32)

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

# Softmax Layer (pure Core ML 'softmax', which satisfies the CCE loss checker)
builder.add_softmax(
    name="softmax_layer",
    input_name="logits",
    output_name="probs"
)

# 3. Mark the final classification layer as updatable
builder.make_updatable(["dense_3"])

# 4. Attach Categorical Cross-Entropy loss referencing the Softmax output
builder.set_categorical_cross_entropy_loss(
    name="lossLayer",
    input="probs"
)

# 5. Set SGD Optimizer and Epochs
builder.set_sgd_optimizer(SgdParams(lr=0.01, batch=32))
builder.set_epochs(1)

# 6. Save the model
builder.spec.isUpdatable = True
builder.spec.description.trainingInput[1].shortDescription = "Target integer label for training"

model = ct.models.MLModel(builder.spec)
model.save("MNISTUpdatable.mlmodel")

print("Created MNISTUpdatable.mlmodel successfully with on-device training enabled.")
