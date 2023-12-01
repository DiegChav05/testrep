import numpy as np
from scipy.integrate import quad
from scipy.special import gamma
from numpy.polynomial.laguerre import laggauss

# Function to integrate
def func_trapezoidal(t, x):
    return t**(x-1) * np.exp(-t)

def func_laguerre(t, x):
    return t**(x-1) * np.exp(-t)

# Composite Trapezoidal Rule
def composite_trapezoidal_rule(func, a, b, n, x):
    h = (b - a) / n
    result = 0.5 * (func(a, x) + func(b, x))
    for i in range(1, n):
        result += func(a + i * h, x)
    result *= h
    return result

def gauss_laguerre_approx(x, n):
    nodes, weights = laggauss(n)
    integral_func = lambda t: func_laguerre(t, x)
    integral_value = np.sum(weights * integral_func(nodes))*gamma(x)
    return integral_value

# Interval and values of x
a = 0
b = 10
x_values = [2,4,6,8,10]

for x in x_values:
    # Composite Trapezoidal Rule
    trapezoidal_result = composite_trapezoidal_rule(func_trapezoidal, a, b, 1000, x)

    # Gauss-Legendre Quadrature
    
    gauss_legendre_result = gauss_laguerre_approx(x, 100)  

    # Adaptive Quadrature
    quad_result, quad_err, quad_output = quad(func_laguerre, a, b, args=(x,), full_output=True)

    print(f"For x = {x}:")
    print(f"Composite Trapezoidal Rule Result: {trapezoidal_result}")
    print(f"Gauss-Legendre Quadrature Result: {gauss_legendre_result}")
    print(f"Adaptive Quadrature Result: {quad_result}")
    print(f"Adaptive Quadrature error: {quad_err}")
    print(f"Number of Function Evaluations (Adaptive Quadrature): {quad_output['neval']}")
    print("\n")
