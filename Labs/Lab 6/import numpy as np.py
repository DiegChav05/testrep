import numpy as np

# Function f(x) = cos(x)
f = lambda x: np.cos(x)
# Value of x at which to evaluate the finite differences
x_value = np.pi / 2

# Calculate step sizes h based on the given formula
h = 0.01 * 2.0 ** (-np.arange(0, 10))

# Compute forward finite differences
forward_differences = (f(x_value + h) - f(x_value)) / h

# Compute centered finite differences
centered_differences = (f(x_value + h) - f(x_value - h)) / (2 * h)

# Print the results
for i in range(len(h)):
    print(f"Step size (h): {h[i]:.10f}")
    print(f"Forward difference: {forward_differences[i]:.10f}")
    print(f"Centered difference: {centered_differences[i]:.10f}")
    print()

forward_errors = []
centered_errors = []

# Compute forward finite differences and errors
for i in range(len(h)):
    forward_difference = (f(x_value + h[i]) - f(x_value)) / h[i]
    forward_error = np.abs(forward_difference + np.sin(x_value)) 
    forward_errors.append(forward_error)


for i in range(len(h)):
    centered_difference = (f(x_value + h[i]) - f(x_value - h[i])) / (2 * h[i])
    centered_error = np.abs(centered_difference + np.sin(x_value)) 
    centered_errors.append(centered_error)

# Calculate the estimated order of accuracy for both forward and centered differences
forward_order = np.log(forward_errors[0] / forward_errors[1]) / np.log(h[0] / h[1])
centered_order = np.log(centered_errors[0] / centered_errors[1]) / np.log(h[0] / h[1])

# Print the results
print(f"Forward difference Order: {forward_order:.5f}")
print(f"Centered difference Order: {centered_order:.5f}")

##############################
##############################