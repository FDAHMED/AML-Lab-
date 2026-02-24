# Task 6: Implement the non-parametric Locally Weighted Regression algorithm in order to fit data points. Select appropriate data set for your experiment and draw graphs.

#LLM Prompt : Give me python code to Implement the non-parametric Locally Weighted Regression (LWR) algorithm in order to fit data points. 
# Use np.asmatrix and noisy sine wave for this experiment to draw graphs.

import numpy as np
import matplotlib.pyplot as plt

def get_weights(query_point, X, tau):
    """Calculate the diagonal weight matrix for a specific query point."""
    m = X.shape[0]
    W = np.eye(m)
    for i in range(m):
        diff = query_point - X[i]
        # Gaussian kernel calculation
        W[i, i] = np.exp(np.dot(diff, diff.T) / (-2.0 * tau**2))
    return np.asmatrix(W)

def lwr_predict(X, y, query_point, tau):
    """Predict the value at a single point using LWR."""
    # Add a bias term (1) to the query point
    q = np.asmatrix([1, query_point])
    
    # Add a bias term (column of 1s) to the dataset
    m = X.shape[0]
    X_bias = np.hstack([np.ones((m, 1)), X])
    X_mat = np.asmatrix(X_bias)
    y_mat = np.asmatrix(y).T
    
    # Calculate weights for this specific query point
    W = get_weights(q[0, 1], X_mat[:, 1], tau)
    
    # Normal Equation: theta = (X^T * W * X)^-1 * X^T * W * y
    try:
        theta = (X_mat.T * (W * X_mat)).I * (X_mat.T * (W * y_mat))
        return q * theta
    except np.linalg.LinAlgError:
        return None

# --- Experiment ---

# 1. Generate Noisy Sine Wave Data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, 100)

# 2. Generate predictions for a range of values
domain = np.linspace(0, 10, 100)
tau = 0.3  # The "smoothness" parameter
predictions = [lwr_predict(X, y, p, tau)[0, 0] for p in domain]

# 3. Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='lightgray', label='Noisy Data')
plt.plot(domain, predictions, color='red', lw=2, label=f'LWR Fit (tau={tau})')
plt.title('Locally Weighted Regression on Noisy Sine Wave')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
