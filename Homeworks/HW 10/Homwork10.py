import numpy as np
import matplotlib.pyplot as plt

def maclaurin_polynomial(x, degree):
    return np.sin(x) + (-1)**(degree//2) * x**(degree+1) / np.math.factorial(degree+1)

def pade_approximation(x, deg_numerator, deg_denominator, a_coeffs, b_coeffs):
    numerator_coeffs = [a_coeffs[i] if i < len(a_coeffs) else 0 for i in range(deg_numerator + 1)]
    denominator_coeffs = [b_coeffs[i] if i < len(b_coeffs) else 0 for i in range(deg_denominator + 1)]

    numerator = sum(numerator_coeffs[i] * x**i for i in range(deg_numerator + 1))
    denominator = 1 + sum(denominator_coeffs[i] * x**i for i in range(1, deg_denominator + 1))

    return numerator / denominator

# Define parameters
deg_numerator = 4
deg_denominator = 2
a_coeffs = [0, 1, 0,-14/120]
b_coeffs = [1, 0, 1/20]
interval = np.linspace(0, 5, 100)

# Calculate errors
maclaurin_errors = [np.abs(np.sin(x) - maclaurin_polynomial(x, 6)) for x in interval]
pade_errors = [np.abs(np.sin(x) - pade_approximation(x, deg_numerator, deg_denominator, a_coeffs, b_coeffs)) for x in interval]

# Plotting the errors on a log scale
plt.plot(interval, maclaurin_errors, label='Maclaurin Polynomial (6th order)')
plt.plot(interval, pade_errors, label='Padé Approximation')
plt.xlabel('x')
plt.ylabel('Error (log)')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.legend()
plt.show()
