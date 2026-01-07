#Task 2.1 : Implement a simple Artificial Neural Network (ANN) from scratch using NumPy,
# including forward propagation, backpropagation, and weight updates.

import numpy as np

# A simple implementation of an Artificial Neural Network with Backpropagation
class NeuralNetwork:
    """
    A simple two-layer neural network with one hidden layer, implemented 
    from scratch using NumPy.
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # 1. Initialize Weights and Biases
        # Weights are initialized randomly to break symmetry and ensure different 
        # hidden units learn different things. We use a small range (-1 to 1).
        # Weights from Input (I) to Hidden (H) layer (W_ih)
        self.W_ih = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        # Bias for Hidden layer (B_h)
        self.B_h = np.random.uniform(-1, 1, (1, self.hidden_size))
        
        # Weights from Hidden (H) to Output (O) layer (W_ho)
        self.W_ho = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        # Bias for Output layer (B_o)
        self.B_o = np.random.uniform(-1, 1, (1, self.output_size))

    # --- Activation Functions ---
    def sigmoid(self, x):
        """The sigmoid activation function: f(x) = 1 / (1 + e^(-x))"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid function, required for backpropagation.
        f'(x) = f(x) * (1 - f(x))
        """
        return x * (1 - x)

    # --- Training Method ---
    def train(self, X, Y, epochs):
        """
        Trains the neural network using the backpropagation algorithm.
        X: Input data (features)
        Y: Target data (labels)
        epochs: Number of iterations to train
        """
        print(f"Training for {epochs} epochs...")

        for epoch in range(epochs):
            
            # --- PHASE 1: FORWARD PROPAGATION ---
            
            # 1. Calculate weighted sum (dot product) for Hidden layer
            # Z_h = X * W_ih + B_h
            self.Z_h = np.dot(X, self.W_ih) + self.B_h
            
            # 2. Apply activation function (Sigmoid) to get Hidden layer output
            # A_h (or H) is the activation of the hidden layer
            self.A_h = self.sigmoid(self.Z_h)
            
            # 3. Calculate weighted sum for Output layer
            # Z_o = A_h * W_ho + B_o
            self.Z_o = np.dot(self.A_h, self.W_ho) + self.B_o
            
            # 4. Apply activation function (Sigmoid) to get final prediction
            # Predicted output (A_o or P)
            self.A_o = self.sigmoid(self.Z_o)

            # --- PHASE 2: BACKPROPAGATION ---
            
            # 1. Calculate Output Error (Cost/Loss)
            # The error is the difference between the target (Y) and the prediction (A_o)
            output_error = Y - self.A_o
            
            # 2. Calculate Output Delta (Error gradient for the Output layer)
            # This is the derivative of the loss * the derivative of the output activation
            # Delta_o = Error * Sigmoid_derivative(A_o)
            output_delta = output_error * self.sigmoid_derivative(self.A_o)
            
            # 3. Calculate Hidden Layer Error 
            # This is the output delta propagated backward through the output weights
            # Error_h = Delta_o * Transpose(W_ho)
            hidden_error = output_delta.dot(self.W_ho.T)
            
            # 4. Calculate Hidden Layer Delta (Error gradient for the Hidden layer)
            # Delta_h = Hidden_Error * Sigmoid_derivative(A_h)
            hidden_delta = hidden_error * self.sigmoid_derivative(self.A_h)
            
            # --- PHASE 3: WEIGHT UPDATES (Gradient Descent) ---
            
            # 1. Update Weights and Biases for the Hidden-to-Output Layer
            # Change in W_ho = (A_h transpose) * Delta_o * Learning_Rate
            self.W_ho += self.A_h.T.dot(output_delta) * self.learning_rate
            self.B_o += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
            
            # 2. Update Weights and Biases for the Input-to-Hidden Layer
            # Change in W_ih = (X transpose) * Delta_h * Learning_Rate
            self.W_ih += X.T.dot(hidden_delta) * self.learning_rate
            self.B_h += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

            # --- Reporting Loss ---
            if epoch % 1000 == 0:
                # Mean Squared Error (MSE) is a common loss function
                mse = np.mean(np.square(output_error))
                print(f"Epoch {epoch}: Loss (MSE) = {mse:.6f}")

    # --- Testing Method ---
    def predict(self, X):
        """
        Performs a forward pass to generate predictions for new data X.
        """
        # Forward pass is the same as in training, but without saving Z_h, Z_o
        # Input -> Hidden
        Z_h = np.dot(X, self.W_ih) + self.B_h
        A_h = self.sigmoid(Z_h)
        
        # Hidden -> Output
        Z_o = np.dot(A_h, self.W_ho) + self.B_o
        A_o = self.sigmoid(Z_o)
        
        return A_o

# --- Data Preparation (XOR Gate Problem) ---
# The XOR problem is non-linearly separable, requiring a hidden layer to solve.
print("--- Initializing XOR Problem Data ---")

# X: Input (4 samples, 2 features each)
X = np.array([
    [0, 0],  # Input 1
    [0, 1],  # Input 2
    [1, 0],  # Input 3
    [1, 1]   # Input 4
])

# Y: Target (4 samples, 1 output each)
Y = np.array([
    [0],     # Expected output for 0 XOR 0
    [1],     # Expected output for 0 XOR 1
    [1],     # Expected output for 1 XOR 0
    [0]      # Expected output for 1 XOR 1
])

# --- Model Setup and Training ---

# Define the architecture: 2 inputs, 4 hidden units, 1 output
INPUT_NODES = 2
HIDDEN_NODES = 4
OUTPUT_NODES = 1
LEARNING_RATE = 0.1
EPOCHS = 10000

nn = NeuralNetwork(
    input_size=INPUT_NODES,
    hidden_size=HIDDEN_NODES,
    output_size=OUTPUT_NODES,
    learning_rate=LEARNING_RATE
)

# Train the network
nn.train(X, Y, epochs=EPOCHS)

# --- Testing and Evaluation ---
print("\n--- Testing Trained Network ---")
predictions = nn.predict(X)

print("\nInputs (X) and Expected Output (Y):")
print(np.hstack((X, Y))) # Combine input and target for easy viewing

print("\nNetwork Predictions (P) (Raw Output):")
print(predictions)

# Evaluate performance by rounding the prediction to 0 or 1
print("\nEvaluation (Rounded Prediction P_round):")
rounded_predictions = np.round(predictions)

results = np.hstack((X, Y, rounded_predictions))
print("Input | Target | Prediction")
for row in results:
    # Format the output for readability
    print(f"{int(row[0])} {int(row[1])} | {int(row[2])}      | {int(row[3])}")

# Calculate accuracy
correct_predictions = np.sum(rounded_predictions == Y)
accuracy = correct_predictions / len(Y)
print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")