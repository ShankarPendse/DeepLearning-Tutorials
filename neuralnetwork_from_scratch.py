# Let's build entire neural network from scratch

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)


# relu activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def derivative_relu(x):
    return np.array(x > 0, dtype=np.float32)


# Function to predict classes from probabilities:
def predict_classes(predictions):
    predicted_classes = np.zeros(len(predictions[0]))
    predicted_classes[np.where(predictions[0] >= 0.5)] = 1
    return predicted_classes


# Function to calculate accuracy:
def calculate_accuracy(predictions, actual):
    accuracy = np.sum(predictions == actual)
    return accuracy/len(actual)


def forward_pass(X, weights1, weights2, weights3, bias1, bias2, bias3):
    layer_computations = dict()

    # print("weights1 shape: ", weights1.shape)
    # print("X shape:", X.shape)
    z1 = np.matmul(weights1, X) + bias1
    a1 = relu(z1)

    # print("weights2 shape: ", weights2.shape)
    # print("a1 shape:", a1.shape)
    z2 = np.matmul(weights2, a1) + bias2
    a2 = relu(z2)

    # print("weights3 shape: ", weights3.shape)
    # print("a2 shape:", a2.shape)
    z3 = np.matmul(weights3, a2) + bias3
    a3 = sigmoid(z3)

    layer_computations['z1'] = z1
    layer_computations['a1'] = a1
    layer_computations['z2'] = z2
    layer_computations['a2'] = a2
    layer_computations['z3'] = z3
    layer_computations['a3'] = a3

    return layer_computations


# Binary cross entropy function
def compute_cost(predictions, actual):
    m = len(actual)
    log_of_predictions = np.log(predictions)
    log_of_oneMinusPredictions = np.log(1-predictions)
    cost = -(1/m)*np.sum(((actual*log_of_predictions) + ((1-actual)*(log_of_oneMinusPredictions))))
    return cost


# Defining backward propagation function
def back_propagation(X, y, layer_computations, weights1, weights2, weights3, bias1, bias2, bias3):
    m = X.shape[1]

    a1 = layer_computations['a1']
    a2 = layer_computations['a2']
    a3 = layer_computations['a3']

    z1 = layer_computations['z1']
    z2 = layer_computations['z2']
    z3 = layer_computations['z3']

    dz3 = a3 - y
    dw3 = (1 / m) * (np.matmul(dz3, a2.T))
    db3 = (1 / m) * (np.sum(dz3, axis=1, keepdims=True))

    dz2 = np.matmul(weights3.T, dz3) * derivative_relu(z2)
    dw2 = (1 / m) * (np.matmul(dz2, a1.T))
    db2 = (1 / m) * (np.sum(dz2, axis=1, keepdims=True))

    dz1 = np.matmul(weights2.T, dz2) * derivative_relu(z1)
    dw1 = (1 / m) * (np.matmul(dz1, X.T))
    db1 = (1 / m) * (np.sum(dz1, axis=1, keepdims=True))

    gradients = {
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2,
        "dw3": dw3,
        "db3": db3
    }

    return gradients


# Function to update weights and biases:
def update_weights_biases(weights1, weights2, weights3, bias1, bias2, bias3, gradients, learning_rate):

    weights1 = weights1 - learning_rate * gradients['dw1']
    weights2 = weights2 - learning_rate * gradients['dw2']
    weights3 = weights3 - learning_rate * gradients['dw3']

    bias1 = bias1 - learning_rate * gradients['db1']
    bias2 = bias2 - learning_rate * gradients['db2']
    bias3 = bias3 - learning_rate * gradients['db3']

    return weights1, weights2, weights3, bias1, bias2, bias3


# Gradient descent function:
def gradient_descent(X, y, weights1, weights2, weights3, bias1, bias2, bias3):
    learning_rate = 0.5
    w1, w2, w3, b1, b2, b3 = weights1, weights2, weights3, bias1, bias2, bias3
    costs = []
    accuracies = []

    for i in range(1000):
        layer_computations = forward_pass(X, w1, w2, w3, b1, b2, b3)
        class_predictions = predict_classes(layer_computations['a3'])
        accuracies.append(calculate_accuracy(class_predictions, y))
        cost = compute_cost(layer_computations['a3'], y)
        costs.append(cost)
        gradients = back_propagation(X,y, layer_computations, w1, w2, w3, b1, b2, b3)
        w1, w2, w3, b1, b2, b3 = update_weights_biases(w1, w2, w3, b1, b2, b3, gradients, learning_rate)

    return w1, w2, w3, b1, b2, b3, costs, accuracies


# Let's create some dummy data using random.randn method of numpy
X = np.random.randn(15, 100)
y = np.random.randint(0, 2, (1, 100))
print("No of 1s in y :", y.sum())

# Let's create weights and biases of our neural network
# weights and bias for hidden layer 1
w1 = np.random.randn(10, 15) * 0.01
b1 = np.zeros((10, 1))

# weights and bias for hidden layer 2
w2 = np.random.randn(5, 10) * 0.01
b2 = np.zeros((5, 1))

# weights and bias for output layer
w3 = np.random.randn(1, 5) * 0.01
b3 = np.zeros((1, 1))

w1, w2, w3, b1, b2, b3, costs, accuracies = gradient_descent(X, y, w1, w2, w3, b1, b2, b3)

print("costs: \n", costs[0], costs[-1])
print("accuracies: \n", accuracies[0], accuracies[-1])

plt.plot(range(1000), costs)
plt.plot(accuracies)
plt.show()

#layer_computations = forward_pass(X, w1, w2, w3, b1, b2, b3)
'''print("shape of a1: ", layer_computations['a1'].shape)
print("shape of a2: ", layer_computations['a2'].shape)
print("shape of a3: ", layer_computations['a3'].shape)

cost = compute_cost(layer_computations['a3'], y)
print("cost after just one forward pass: ", cost)

print("shape of weights and biases :")
print("w3: ", w3.shape)
print("b3: ", b3.shape)
print("w2: ", w2.shape)
print("b2: ", b2.shape)
print("w1: ", w1.shape)
print("b1: ", b1.shape)

gradients = back_propagation(X, y, layer_computations, w1, w2, w3, b1, b2, b3)
for key, value in gradients.items():
    print("shape of {}: {}".format(key, gradients[key].shape))'''

