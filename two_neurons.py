import numpy as np

inputs = [22, 21.8, 18.75, 20]

weights1 = [0.8, -0.4, 0.3, 0.6]
weights2 = [0.5, -1.4, 1.3, -0.6]
bias1 = 0
bias2 = 3


# Rectified linear unit function
def relu(x):
    return max(0, x)


def numpy_relu(z):
    return np.maximum(0, z)


# Linear transformations for neuron 1 and 2:

# for neuron 1
z1 = (weights1[0]*inputs[0] + weights1[1]*inputs[1] + weights1[2]*inputs[2] + weights1[3]*inputs[3]) + bias1

# for neuron 2
z2 = (weights2[0]*inputs[0] + weights2[1]*inputs[1] + weights2[2]*inputs[2] + weights2[3]*inputs[3]) + bias2

# activation on z1 and z2:

a1 = relu(z1)
a2 = relu(z2)

print("linear transformations: ", z1, z2)
print("neuron outputs: ", a1, a2)


weights = np.array(
    [
        [0.8, -0.4, 0.3, 0.6],
        [0.5, -1.4, 1.3, -0.6]
    ]
)

bias = np.array([0, 3])

z = np.matmul(weights, inputs)
z = z+bias
a = numpy_relu(z)

print("\n using matrix operations, the outputs are:\n")
print("linear transformations: ", z)
print("neuron outputs: ", a)