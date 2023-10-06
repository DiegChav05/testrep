

import numpy as np

# Define the function and its derivative for the ellipsoid
def f(x):
    return x[0]**2 + 4*x[1]**2 + 4*x[2]**2 - 16

def f_prime(x):
    return np.array([2*x[0], 8*x[1], 8*x[2]])

# Initial point
x0 = np.array([1, 1, 1])

# Convergence criteria
tolerance = 1e-10
max_iterations = 100

# Lists to store errors and error ratios
errors = []

# Perform the iteration (Newton's method)
x = x0

for iteration_count in range(1, max_iterations + 1):
    # Evaluate f and its derivative at x
    f_val = f(x)
    f_prime_val = f_prime(x)

    # Calculate the step using the pseudo-inverse of the Jacobian
    step = -np.linalg.pinv(np.atleast_2d(f_prime_val)) @ np.atleast_2d(f_val).T

    # Update x using the iteration formula
    x_next = x + step.flatten()

    # Calculate error
    error = np.linalg.norm(x_next - x)
    errors.append(error)

    # Check for convergence
    if error < tolerance:
        break

    # Update for the next iteration
    x = x_next

# Calculate error ratios and print
print("Converged to:")
print("x =", x)
print("Number of iterations:", iteration_count)

# Print error ratios
for i in range(1, len(errors)):
    error_ratio = errors[i] / (errors[i - 1]**2)  # Quadratic convergence
    print(error_ratio)
    
