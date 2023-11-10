import numpy as np
from scipy.integrate import quad

def f(x):
    return 1 / (1 + x**2)

def composite_trapezoidal_rule(a, b, n):
    h = (b - a) / n
    x_values = np.linspace(a, b, n+1)
    integral = h * (f(x_values[0]) + 2 * np.sum(f(x_values[1:n])) + f(x_values[n])) / 2
    return integral

def composite_simpsons_rule(a, b, n):
    h = (b - a) / n
    x_values = np.linspace(a, b, n+1)
    integral = h/3 * (f(x_values[0]) + 4 * np.sum(f(x_values[1:n:2])) + 2 * np.sum(f(x_values[2:n-1:2])) + f(x_values[n]))
    return integral

def integrate(a, b, n, method='trapezoidal'):
    if method == 'trapezoidal':
        return composite_trapezoidal_rule(a, b, n)
    elif method == 'simpsons':
        # Ensure n is even for Simpson's rule
        if n % 2 != 0:
            n += 1
        return composite_simpsons_rule(a, b, n)
    else:
        raise ValueError("Invalid method. Choose 'trapezoidal' or 'simpsons'.")

# Example usage:
a, b = -5, 5
n = 100  
method = 'simpsons'  # ('trapezoidal''simpsons')

# Your implementation
result_sn = integrate(a, b, n, method)
print(f"Approximate integral using {method} rule: {result_sn}")


# Using scipy quad with default tolerance
result_quad_default, err, info = quad(f, a, b,full_output=1)
print(f"Approximate integral using quad (default tolerance): {result_quad_default}")
# Compare the number of function evaluations
print("It used {} points for default tolerance".format(info['neval']))


# Using scipy quad with set tolerance of 1e-4
result_quad_set_tol, err, info = quad(f, a, b,full_output=1, epsabs=1e-4)
print(f"Approximate integral using quad (set tolerance): {result_quad_set_tol}")

print("It used {} points for 10^-4 tolerance".format(info['neval']))