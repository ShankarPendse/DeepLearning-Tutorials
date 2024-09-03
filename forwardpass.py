import numpy as np


def numpy_relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1/(1+np.exp(-z))


def forward_pass(X, weights1, weights2, weights3, bias1, bias2, bias3):
    # dictionary to store the layer computations
    layer_computations = dict()

    # Hidden layer 1 computations:
    z1 = np.matmul(weights1, X) + bias1
    a1 = numpy_relu(z1)
    layer_computations['z1'] = z1
    layer_computations['a1'] = a1

    # Hidden layer2 computations:
    z2 = np.matmul(weights2, a1) + bias2
    a2 = numpy_relu(z2)
    layer_computations['z2'] = z2
    layer_computations['a2'] = a2

    # Output layer computations:
    z3 = np.matmul(weights3, a2) + bias3
    a3 = sigmoid(z3)
    layer_computations['z3'] = z3
    layer_computations['a3'] = a3

    return layer_computations


# Input features (considering only one data point with 4 features)
#X = [22, 21.8, 18.75, 20]

X = np.random.randn(4, 10)
print("input data points: ", X)
print("shape of X is: ", X.shape)

# Weights & bias associated with layer 1
weights1 = np.array(
    [
        [0.8, -0.4, 0.3, 0.6],
        [0.5, -1.4, 1.3, -0.6],
        [0.1, 0.4, -0.3, 0.75],
        [-0.29, -0.85, 0.3, 0.68],
        [0.75, -3.4, 1.3, -0.5]
    ]
)

bias1 = np.array([0, 3, 1, 5, -2]).reshape(5, 1)

# Weights & bias associated with layer 2
weights2 = weights1/2
weights2 = np.c_[weights2, np.array([1, 2, 3, 4, 5])]
bias2 = np.array([0.22, 0.36, -0.52, 1, -0.78]).reshape(5, 1)

# Weights & bias associated with layer 3
weights3 = np.array([0.22, 0.36, -0.52, 1, -0.78])
bias3 = -2

layer_computations = forward_pass(X, weights1, weights2, weights3, bias1, bias2, bias3)

print("Forward propagation outputs:")
for key, value in layer_computations.items():
    print(key, ":", layer_computations[key])
    print("shape: ", layer_computations[key].shape)