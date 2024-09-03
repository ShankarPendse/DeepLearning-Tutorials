# Simulating the working of a single neuron in a Artificial Neural Network.

# Rectified linear unit function
def relu(x):
    return max(0, x)


# Input features
inputs = [12, 14, 11.5, 15]

# Neuron components
weights = [0.2, 0.55, -0.1, -0.2]
bias = -10

# neuron calculations:

# Linear Transformation:
z = (weights[0]*inputs[0] + weights[1]*inputs[1] + weights[2]*inputs[2] + weights[3]*inputs[3]) + bias

# Non linear transformation: ReLu (Rectified Linear Unit)
a = relu(z)

print("value after linear transformation is {} and after applying relu activation, output of the neuron is {}".format(z, a))




