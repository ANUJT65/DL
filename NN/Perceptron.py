# https://medium.com/codex/single-layer-perceptron-and-activation-function-b6b74b4aae66             medium article for reference
# https://medium.com/@abhishekjainindore24/perceptron-vs-neuron-single-layer-perceptron-and-multi-layer-perceptron-68ce4e8db5ea              medium article for reference

###############################################################

# import pandas as pd
# import numpy as np

# # Load the CSV file
# data = pd.read_csv('D:\\Deep Learning\\Datasets\\gates.csv')  # Replace with your CSV file path

# # Define the perceptron model
# class Perceptron:
#     def __init__(self, input_size, learning_rate=0.1, epochs=10):
#         self.weights = np.zeros(input_size + 1)  # +1 for bias
#         self.learning_rate = learning_rate
#         self.epochs = epochs

#     def activation_fn(self, x):
#         return 1 if x >= 0 else 0  # Step function

#     def predict(self, inputs):
#         summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
#         return self.activation_fn(summation)

#     def train(self, X, y):
#         for _ in range(self.epochs):
#             for inputs, label in zip(X, y):
#                 prediction = self.predict(inputs)
#                 self.weights[1:] += self.learning_rate * (label - prediction) * inputs
#                 self.weights[0] += self.learning_rate * (label - prediction)

# # Function to train and test on given logic gate
# def train_perceptron_for_gate(gate):
#     print(f"\nTraining perceptron for {gate} gate:")
#     X = data[['A', 'B']].values
#     y = data[gate].values

#     perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
#     perceptron.train(X, y)

#     print("Final Weights:", perceptron.weights)
#     print("Testing:")
#     for inputs in X:
#         print(f"Input: {inputs}, Predicted Output: {perceptron.predict(inputs)}")

# # Train on AND, OR, and XOR gates
# train_perceptron_for_gate('AND')
# train_perceptron_for_gate('OR')
# train_perceptron_for_gate('XOR')



##################################################################################



# import numpy as np

# def step_function(x):
#     return np.where(x >= 0, 1, 0)

# class SingleLayerPerceptron:
#     def __init__(self, input_size, learning_rate=0.1):
#         self.weights = np.zeros(input_size + 1)  # +1 for bias
#         self.learning_rate = learning_rate

#     def predict(self, x):
#         x = np.insert(x, 0, 1)  # Bias input
#         z = np.dot(self.weights, x)
#         return step_function(z)

#     def train(self, X, y, epochs=10):
#         for epoch in range(epochs):
#             total_correct = 0
#             for xi, target in zip(X, y):
#                 xi_bias = np.insert(xi, 0, 1)
#                 z = np.dot(self.weights, xi_bias)
#                 output = step_function(z)
#                 error = target - output
#                 self.weights += self.learning_rate * error * xi_bias
#                 if error == 0:
#                     total_correct += 1

#             accuracy = total_correct / len(y) * 100
#             print(f"Epoch {epoch+1}: Weights = {self.weights}, Accuracy = {accuracy:.2f}%")


# def test_perceptron(gate_name, X, y):
#     print(f"\nTraining Perceptron for {gate_name} Gate")
#     perceptron = SingleLayerPerceptron(input_size=2, learning_rate=0.1)
#     perceptron.train(X, y, epochs=10)

#     print(f"\nTesting {gate_name} Gate:")
#     for xi, target in zip(X, y):
#         pred = perceptron.predict(xi)
#         print(f"Input: {xi}, Predicted: {pred}, Expected: {target}")


# if __name__ == "__main__":
#     # Define input/output for logic gates
#     X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

#     # AND gate
#     y_and = np.array([0, 0, 0, 1])
#     test_perceptron("AND", X, y_and)

#     # OR gate
#     y_or = np.array([0, 1, 1, 1])
#     test_perceptron("OR", X, y_or)

#     # XOR gate (will not converge)
#     y_xor = np.array([0, 1, 1, 0])
#     test_perceptron("XOR", X, y_xor)



################################################################################################


import numpy as np

def step_function(x):
    """Step activation function."""
    return np.where(x >= 0, 1, 0)

class SingleLayerPerceptron:
    def __init__(self, input_size, learning_rate=0.1):
        """Initialize weights and learning rate."""
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate

    def predict(self, x):
        """Make a prediction based on input x."""
        x = np.insert(x, 0, 1)  # Bias input
        z = np.dot(self.weights, x)
        return step_function(z)

    def train(self, X, y, epochs=10):
        """Train the perceptron using the provided dataset."""
        for epoch in range(epochs):
            total_correct = 0
            for xi, target in zip(X, y):
                xi_bias = np.insert(xi, 0, 1)
                z = np.dot(self.weights, xi_bias)
                output = step_function(z)
                error = target - output
                self.weights += self.learning_rate * error * xi_bias
                if error == 0:
                    total_correct += 1

            accuracy = total_correct / len(y) * 100
            print(f"Epoch {epoch+1}: Weights = {self.weights}, Accuracy = {accuracy:.2f}%")

    def __str__(self):
        """Return a string representation of the perceptron's weights."""
        return f"Weights: {self.weights}"

def test_perceptron(gate_name, X, y, epochs=10):
    """Train and test the perceptron on a specific logic gate."""
    print(f"\nTraining Perceptron for {gate_name} Gate")
    perceptron = SingleLayerPerceptron(input_size=X.shape[1], learning_rate=0.1)
    perceptron.train(X, y, epochs)

    print(f"\nTesting {gate_name} Gate:")
    for xi, target in zip(X, y):
        pred = perceptron.predict(xi)
        print(f"Input: {xi}, Predicted: {pred}, Expected: {target}")

if __name__ == "__main__":
    # Define input/output for logic gates
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # AND gate
    y_and = np.array([0, 0, 0, 1])
    test_perceptron("AND", X, y_and)

    # OR gate
    y_or = np.array([0, 1, 1, 1])
    test_perceptron("OR", X, y_or)

    # XOR gate (will not converge)
    y_xor = np.array([0, 1, 1, 0])
    test_perceptron("XOR", X, y_xor)
