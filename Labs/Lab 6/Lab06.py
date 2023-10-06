import numpy as np
import math
from numpy.linalg import inv
from numpy.linalg import norm

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




def evalF(x):
    F  = np.zeros(3)
    F[0] = 3*x[0]-math.cos(x[1]*x[2])-1/2
    F[1] = x[0]-81*(x[1]+0.1)**2+math.sin(x[2])+1.06
    F[2] = np.exp(-x[0]*x[1])+20*x[2]+(10*math.pi-3)/3
    return F

def evalJ(x):
    J = np.array([[8*x[0], 2*x[1]],
    [1 - np.cos(x[0]-x[1]), 1+np.cos(x[0] - x[1])]])

    return J


def LazyNewton(x0,tol,Nmax):
#''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
#''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
#''' Outputs: xstar= approx root, ier = error message, its = num its'''
    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
        if (norm(x1-x0) < tol):
            xstar = x1
            ier =0
            return[xstar, ier,its]
        x0  = x1
    xstar = x1
    ier = 1
    return[xstar,ier,its]

def SlackerNewton(x0, tol, Nmax, recalculate_interval=3):
    '''
    Lazy Newton: use the inverse of the Jacobian for initial guess.
    
    Inputs:
    x0 = initial guess
    tol = tolerance
    Nmax = max iterations
    recalculate_interval = how often to recalculate the Jacobian
    
    Outputs:
    xstar = approximate root
    ier = error message
    its = number of iterations
    jacobian_count = number of times Jacobian is recalculated
    '''
    jacobian_count = 0

    for its in range(Nmax):
        if its % recalculate_interval == 0:
            J = evalJ(x0)
            Jinv = inv(J)
            jacobian_count += 1
        
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
        
        if norm(x1 - x0) < tol:
            xstar = x1
            ier = 0
            return [xstar, ier, its, jacobian_count]
        
        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its, jacobian_count]

def evalF(x):
    F  = np.zeros(2)
    F[0] = 4*x[0]**2-x[1]**2-4
    F[1] = x[0]+x[1]-np.sin(x[0]-x[1])
    return F

x0 = np.array([1,0])
tol = 1e-10
Nmax = 1000

[xstar, ier, its, jacobian_count] = SlackerNewton(x0, tol, Nmax, recalculate_interval=2)

print(xstar)
print("Jacobian count",jacobian_count)
print("Iteration count", its)
print("My lab partners was better as it converged in 4 iterations rather than 7")

def evalJ(x, h=1e-5):
    return finiteJ(h, x, evalF)

def finiteJ(h, x, f):

    par1 = (f([x[0] + h, x[1]]) - f(x)) / h
    par2 = (f([x[0], x[1] + h]) - f(x)) / h

    J = np.array([[par1[0], par2[0]],
                  [par1[1], par2[1]]])

    return J

def Newton(x0, tol, Nmax, evalJ_function):
    for its in range(Nmax):
        J = evalJ_function(x0)  # Use the provided evalJ function
        Jinv = inv(J)
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
        if norm(x1 - x0) < tol:
            xstar = x1
            ier = 0
            return [xstar, ier, its]
        x0 = x1
    xstar = x1
    ier = 1
    return [xstar, ier, its]

# Test Newton's method with the finiteJ function
x0 = np.array([1.0, 1.0])
tolerance = 1e-8
max_iterations = 100
[root,ier,its] = Newton(x0, tolerance, max_iterations, evalJ)
print("Newton's method modified jacobian result:", root)
print("Newton's modified jacobian result:", its)