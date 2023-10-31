import numpy as np
import matplotlib.pyplot as plt

# Define your custom function f(x)
def f(x):
    return 2 * x**2 - 3 * x + 1  # Example quadratic function

# Generate noisy data points for x and f(x)
np.random.seed(0)  # For reproducibility
num_points = 100
x = np.linspace(0, 10, num_points)  # Generate x values
f_true = f(x)
noise = np.random.normal(0, 2, num_points)  # Add random noise to f(x)
f_x = f_true + noise

# Perform linear least squares fit
Acol1 = x
Acol2 = np.ones(num_points)
A = np.column_stack((Acol1, Acol2))
x_hat = np.linalg.lstsq(A, f_x, rcond=None)[0]

# Extract the best estimate for the coefficients
m_best = x_hat[0]
b_best = x_hat[1]

# Calculate the fitted values based on the linear model
f_x_least_squares = m_best * x + b_best

# Plot the noisy data and the linear least squares regression
plt.figure(1)
plt.plot(x, f_x, 'o', label='Noisy Data')
plt.plot(x, f_x_least_squares, label='Linear Least Squares Fit')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()
