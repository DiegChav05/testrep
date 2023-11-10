import numpy as np
import matplotlib.pyplot as plt

# Define your custom quadratic function
def f_quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def f_fourth(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

def least_squares_fit(A, y, lambda_reg=0.01):
    # Custom least squares solution with L2 regularization
    return np.linalg.inv(A.T @ A + lambda_reg * np.identity(A.shape[1])) @ A.T @ y

# Generate noisy data points for x and f(x)
np.random.seed(0)
num_points = 100
x = np.linspace(-5, 5, num_points)
f_true = f_fourth(x, 1, 0, 0, 0, 0)  # True fourth-order polynomial function
noise = np.random.normal(0, 100, num_points)
f_x = f_true + noise

# Quadratic Fit (Only a Coefficient)
A_quadratic_a = np.column_stack((x**2, x*0, np.zeros(num_points)))
coeffs_quadratic_a = least_squares_fit(A_quadratic_a, f_x)

# Extract the best estimate for the coefficients
a_best_a = coeffs_quadratic_a[0]
b_best_a = coeffs_quadratic_a[1]
c_best_a = coeffs_quadratic_a[2]

# Calculate the fitted values based on the quadratic model
f_x_quadratic_a = f_quadratic(x, a_best_a, b_best_a, c_best_a)

# Calculate residuals
residuals_quadratic_a = f_x - f_x_quadratic_a

# Plot the results
plt.figure(1)
plt.plot(x, f_x, 'o', label='Noisy Data')
plt.plot(x, f_true, label='True Fourth-Order Polynomial Function', linewidth=2, color='g')
plt.plot(x, f_x_quadratic_a, label='Quadratic Fit (Only a)', linestyle='dashed')

for i in range(num_points):
    plt.plot([x[i], x[i]], [f_x[i], f_x_quadratic_a[i]], 'r--')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

# Quadratic Fit (All Coefficients)
A_quadratic_all = np.column_stack((x**2, x, np.ones(num_points)))
coeffs_quadratic_all = least_squares_fit(A_quadratic_all, f_x)

# Extract the best estimate for the coefficients
a_best_all = coeffs_quadratic_all[0]
b_best_all = coeffs_quadratic_all[1]
c_best_all = coeffs_quadratic_all[2]

# Calculate the fitted values based on the quadratic model
f_x_quadratic_all = f_quadratic(x, a_best_all, b_best_all, c_best_all)

# Calculate residuals
residuals_quadratic_all = f_x - f_x_quadratic_all

# Plot the results
plt.figure(2)
plt.plot(x, f_x, 'o', label='Noisy Data')
plt.plot(x, f_true, label='True Fourth-Order Polynomial Function', linewidth=2, color='g')
plt.plot(x, f_x_quadratic_all, label='Quadratic Fit (All Coefficients)', linestyle='dashed')

for i in range(num_points):
    plt.plot([x[i], x[i]], [f_x[i], f_x_quadratic_all[i]], 'r--')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

# Fourth Order Fit
A_fourth = np.column_stack((x**4, x**3, x**2, x, np.ones(num_points)))
coeffs_4th = least_squares_fit(A_fourth, f_x)

# Extract the best estimate for the coefficients
a_best_4th = coeffs_4th[0]
b_best_4th = coeffs_4th[1]
c_best_4th = coeffs_4th[2]
d_best_4th = coeffs_4th[3]
e_best_4th = coeffs_4th[4]

# Calculate the fitted values based on the model
f_x_4th = f_fourth(x, a_best_4th, b_best_4th, c_best_4th, d_best_4th, e_best_4th)

# Calculate residuals
residuals_4th = f_x - f_x_4th

# Plot the results
plt.figure(3)
plt.plot(x, f_x, 'o', label='Noisy Data')
plt.plot(x, f_true, label='True Fourth-Order Polynomial Function', linewidth=2, color='g')
plt.plot(x, f_x_4th, label='Fourth Order Fit', linestyle='dashed')

for i in range(num_points):
    plt.plot([x[i], x[i]], [f_x[i], f_x_4th[i]], 'r--')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()
