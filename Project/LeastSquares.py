import numpy as np
import matplotlib.pyplot as plt

# Define quadratic function
def f_quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def f_fourth(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


def least_squares_fit(A, y):
    A_T = A.T
    
    # Calculate the dot product of the transpose of A and A
    A_T_A = np.dot(A_T, A)
    
    # Calculate the inverse of the dot product
    A_T_A_inv = np.linalg.pinv(A_T_A)
    
    # Calculate the dot product of the inverse and the transpose of A
    A_T_A_inv_A_T = np.dot(A_T_A_inv, A_T)
    
    # Calculate the final coefficients
    coefficients = np.dot(A_T_A_inv_A_T, y)
    
    return coefficients

# Generate noisy data points for x and f(x)
np.random.seed(0)
num_points = 100
x = np.linspace(-3, 3, num_points)
#noise = np.random.normal(0, .001, num_points) #(0,.1) for x^2
x = x
#f_true = f_fourth(x, 0, 0, 1, 0, 0)  # True fourth-order polynomial function
f_true = f_fourth(x, 1, -2, -2, 6, 5)
noise = np.random.normal(0, 20, num_points)
f_x = f_true + noise

# Quadratic Fit (Only a Coefficient)
A_quadratic_a = np.column_stack((x**2, x*0, np.zeros(num_points)))
coeffs_quadratic_a = least_squares_fit(A_quadratic_a, f_x)

# Extract the best estimate for the coefficients
a_best_a = coeffs_quadratic_a[0]
b_best_a = coeffs_quadratic_a[1]
c_best_a = coeffs_quadratic_a[2]


# Quadratic Fit (All Coefficients)
A_quadratic_all = np.column_stack((x**2, x, np.ones(num_points)))
coeffs_quadratic_all = least_squares_fit(A_quadratic_all, f_x)

# Extract the best estimate for the coefficients
a_best_all = coeffs_quadratic_all[0]
b_best_all = coeffs_quadratic_all[1]
c_best_all = coeffs_quadratic_all[2]

# Fourth Order Fit
A_fourth = np.column_stack((x**4, x**3, x**2, x, np.ones(num_points)))
coeffs_4th = least_squares_fit(A_fourth, f_x)

# Extract the best estimate for the coefficients
a_best_4th = coeffs_4th[0]
b_best_4th = coeffs_4th[1]
c_best_4th = coeffs_4th[2]
d_best_4th = coeffs_4th[3]
e_best_4th = coeffs_4th[4]



# Calculate the fitted values based on the quadratic model
f_x_quadratic_a = f_quadratic(x, a_best_a, b_best_a, c_best_a)

# Calculate the fitted values based on the quadratic model
f_x_quadratic_all = f_quadratic(x, a_best_all, b_best_all, c_best_all)

# Calculate the fitted values based on the model
f_x_4th = f_fourth(x, a_best_4th, b_best_4th, c_best_4th, d_best_4th, e_best_4th)


# Calculate residuals
residuals_quadratic_a = f_x - f_x_quadratic_a

# Plot the results
plt.figure(1)
plt.plot(x, f_x, 'o', label='Noisy Data')
plt.plot(x, f_true, label='Actual Function', linewidth=2, color='g')
plt.plot(x, f_x_quadratic_a, label='Quadratic Fit (Only a)', linestyle='dashed')

for i in range(num_points):
    plt.plot([x[i], x[i]], [f_x[i], f_x_quadratic_a[i]], 'r--')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()


plt.figure(2)
plt.plot(x, residuals_quadratic_a, 'o', label='Error')
plt.title('Residuals (Quadratic Fit only a)')


residuals_quadratic_all = f_x - f_x_quadratic_all


# Plot the results
plt.figure(3)
plt.plot(x, f_x, 'o', label='Noisy Data')
plt.plot(x, f_true, label='Actual Function', linewidth=2, color='g')
plt.plot(x, f_x_quadratic_all, label='Quadratic Fit (All Coefficients)', linestyle='dashed')

for i in range(num_points):
    plt.plot([x[i], x[i]], [f_x[i], f_x_quadratic_all[i]], 'r--')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

plt.figure(4)
plt.plot(x, residuals_quadratic_all, 'o', label='Error')
plt.title('Residuals (Quadratic Fit all coefficients)')


# Calculate residuals
residuals_4th = f_x - f_x_4th

# Plot the results
plt.figure(5)
plt.plot(x, f_x, 'o', label='Noisy Data')
plt.plot(x, f_true, label='True Fourth-Order Polynomial Function', linewidth=2, color='g')
plt.plot(x, f_x_4th, label='Fourth Order Fit', linestyle='dashed')

for i in range(num_points):
    plt.plot([x[i], x[i]], [f_x[i], f_x_4th[i]], 'r--')


plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

plt.figure(6)
plt.plot(x, residuals_4th, 'o', label='Error')
plt.title('Residuals (Fourth Order Fit)')


plt.show()

