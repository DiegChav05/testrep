import numpy as np

def bisection(f, a, b, tol, Nmax):
    iterations = 0
    while iterations < Nmax:
        c = (a + b) / 2
        if f(c) == 0 or abs(b - a) < tol:
            return c, iterations
        elif f(c) * f(a) < 0:
            b = c
        else:
            a = c
        iterations += 1
    return None, iterations


def newton(f, f_prime, x0, tol, Nmax):
    x = x0
    iterations = 0
    while iterations < Nmax:
        if abs(f(x)) < tol:
            return x, iterations
        if f_prime(x) == 0:
            return None, iterations
        x = x - f(x) / f_prime(x)
        iterations += 1
    return None, iterations




def bisection_to_newton(tol, a, b, f, f_prime):
    # Bisection method
    num_iterations = 0
    while (b - a) / 2 >= tol:
        midpoint = (a + b) / 2

        if f(midpoint) == 0 or f_prime(midpoint) == 0:
            return midpoint, num_iterations  # Return if midpoint is a root or division by zero

        if f(midpoint) * f(a) < 0:
            b = midpoint
        else:
            a = midpoint

        num_iterations += 1

    root = midpoint - f(midpoint) / f_prime(midpoint)
    return root,num_iterations
   
tol = 1e-10

# Define the function and its derivative using lambda functions
f = lambda x: np.exp(x**2+7*x-30) -1
f_prime = lambda x: (2*x+7) * np.exp(x**2+7*x-30)


a1= 2
b2 = 4.5
Nmax = 100000

xstar,iterations = bisection(f,a1,b2,tol,Nmax)
print("Root found using Bisection method:", xstar)
print("Number of iterations for Bisection method:", np.ceil(iterations))


x0 = 4.5

xstar,iterations = newton(f, f_prime, x0, tol, Nmax)
print("Root found using Newton's method:", xstar)
print("Number of iterations for Newton's method:", np.ceil(iterations))

a = 2
b = 4.5


# Perform bisection and check for basin of convergence for Newton's method
root, num_iterations = bisection_to_newton(tol, a, b, f, f_prime)
print("Root found using Hybrid method:", root)
print("Number of iterations for Hybrid method:", num_iterations)










