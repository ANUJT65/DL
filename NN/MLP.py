# import numpy as np

# def sigmoid(x: np.ndarray) -> np.ndarray:
#     """Sigmoid activation function."""
#     return 1 / (1 + np.exp(-x))

# def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
#     """Derivative of the sigmoid function."""
#     return x * (1 - x)

# class MLP:
#     def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float = 0.1):
#         # Initialize weights with small random values
#         self.weights_input_hidden = np.random.uniform(-0.5, 0.5, (input_size + 1, hidden_size))
#         self.weights_hidden_output = np.random.uniform(-0.5, 0.5, (hidden_size + 1, output_size))
#         self.learning_rate = learning_rate

#     def add_bias(self, X: np.ndarray) -> np.ndarray:
#         """Add bias unit to the input data."""
#         return np.insert(X, 0, 1, axis=1)

#     def forward(self, X: np.ndarray) -> np.ndarray:
#         """Perform forward propagation."""
#         X_bias = self.add_bias(X)
#         self.hidden_input = np.dot(X_bias, self.weights_input_hidden)
#         self.hidden_output = sigmoid(self.hidden_input)
        
#         hidden_output_bias = self.add_bias(self.hidden_output)
#         self.final_input = np.dot(hidden_output_bias, self.weights_hidden_output)
#         self.final_output = sigmoid(self.final_input)
        
#         return self.final_output

#     def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10000, print_interval: int = 1000):
#         """Train the MLP using backpropagation."""
#         n_samples = X.shape[0]
        
#         for epoch in range(1, epochs + 1):
#             # Forward pass
#             output = self.forward(X)
#             error = y.reshape(-1, 1) - output
#             loss = np.mean(np.square(error))
            
#             # Backpropagation
#             delta_output = error * sigmoid_derivative(output)
#             hidden_output_bias = self.add_bias(self.hidden_output)
#             d_weights_hidden_output = np.dot(hidden_output_bias.T, delta_output)

#             weights_hidden_output_no_bias = self.weights_hidden_output[1:, :]
#             delta_hidden = np.dot(delta_output, weights_hidden_output_no_bias.T) * sigmoid_derivative(self.hidden_output)
#             X_bias = self.add_bias(X)
#             d_weights_input_hidden = np.dot(X_bias.T, delta_hidden)

#             # Update weights
#             self.weights_hidden_output += self.learning_rate * d_weights_hidden_output
#             self.weights_input_hidden += self.learning_rate * d_weights_input_hidden

#             # Print loss and accuracy at specified intervals
#             if epoch % print_interval == 0 or epoch == 1:
#                 predictions = (output >= 0.5).astype(int)
#                 accuracy = np.mean(predictions == y.reshape(-1, 1)) * 100
#                 print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")

#     def predict(self, X: np.ndarray) -> np.ndarray:
#         """Predict output for given input data."""
#         output = self.forward(X)
#         return (output >= 0.5).astype(int)

# def test_mlp(gate_name: str, X: np.ndarray, y: np.ndarray, epochs: int = 10000):
#     """Test the MLP on a specific logic gate."""
#     print(f"\nTraining MLP for {gate_name} Gate")
#     mlp = MLP(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)
#     mlp.train(X, y, epochs=epochs, print_interval=epochs // 10)

#     print(f"\nTesting {gate_name} Gate:")
#     predictions = mlp.predict(X)
#     for xi, pred, target in zip(X, predictions.flatten(), y):
#         print(f"Input: {xi}, Predicted: {pred}, Expected: {target}")

# if __name__ == "__main__":
#     # Define input for logic gates
#     X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
#     # AND gate
#     y_and = np.array([0, 0, 0, 1])
#     test_mlp("AND", X, y_and)

#     # OR gate
#     y_or = np.array([0, 1, 1, 1])
#     test_mlp("OR", X, y_or)

#     # XOR gate
#     y_xor = np.array([0, 1, 1, 0])
#     test_mlp("XOR", X, y_xor)





import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid function."""
    return x * (1 - x)

class MLP:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float = 0.1):
        self.weights_input_hidden = np.random.uniform(-0.5, 0.5, (input_size + 1, hidden_size))
        self.weights_hidden_output = np.random.uniform(-0.5, 0.5, (hidden_size + 1, output_size))
        self.learning_rate = learning_rate

    def add_bias(self, X: np.ndarray) -> np.ndarray:
        """Add bias unit to the input data."""
        return np.insert(X, 0, 1, axis=1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Perform forward propagation."""
        X_bias = self.add_bias(X)
        hidden_output = sigmoid(np.dot(X_bias, self.weights_input_hidden))
        final_output = sigmoid(np.dot(self.add_bias(hidden_output), self.weights_hidden_output))
        return final_output

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10000, print_interval: int = 1000):
        """Train the MLP using backpropagation."""
        for epoch in range(1, epochs + 1):
            output = self.forward(X)
            error = y.reshape(-1, 1) - output
            loss = np.mean(np.square(error))

            # Backpropagation
            delta_output = error * sigmoid_derivative(output)
            delta_hidden = np.dot(delta_output, self.weights_hidden_output[1:, :].T) * sigmoid_derivative(sigmoid(np.dot(self.add_bias(X), self.weights_input_hidden)))

            # Update weights
            self.weights_hidden_output += self.learning_rate * np.dot(self.add_bias(sigmoid(np.dot(self.add_bias(X), self.weights_input_hidden))).T, delta_output)
            self.weights_input_hidden += self.learning_rate * np.dot(self.add_bias(X).T, delta_hidden)

            # Print loss and accuracy at specified intervals
            if epoch % print_interval == 0 or epoch == 1:
                predictions = (output >= 0.5).astype(int)
                accuracy = np.mean(predictions == y.reshape(-1, 1)) * 100
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict output for given input data."""
        return (self.forward(X) >= 0.5).astype(int)

def test_mlp(gate_name: str, X: np.ndarray, y: np.ndarray, epochs: int = 10000):
    """Test the MLP on a specific logic gate."""
    print(f"\nTraining MLP for {gate_name} Gate")
    mlp = MLP(input_size=2, hidden_size=2, output_size=1)
    mlp.train(X, y, epochs=epochs)

    print(f"\nTesting {gate_name} Gate:")
    predictions = mlp.predict(X)
    for xi, pred in zip(X.tolist(), predictions.flatten()):
        print(f"Input: {xi}, Predicted: {pred}, Expected: {y[np.where((X == xi).all(axis=1))[0][0]]}")

if __name__ == "__main__":
    # Define input for logic gates
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    # Test logic gates
    test_mlp("AND", X, np.array([0, 0, 0, 1]))
    test_mlp("OR", X, np.array([0, 1, 1, 1]))
    test_mlp("XOR", X, np.array([0, 1, 1, 0]))

