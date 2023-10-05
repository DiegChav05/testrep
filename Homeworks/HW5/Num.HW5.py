import numpy as np
import math
import time
from numpy.linalg import inv
from numpy.linalg import norm


def driver():
    ###################
    #Problem 1
    f = lambda x,y: 3*x**2 - y**2
    g = lambda x,y: 3*x*y - x**3 - 1

    M = np.array([[1/6,1/18] , [0,1/6]])

    tol = 1e-6

    x0 = 1
    y0 = 1

    count = 0
    iterations = 100

    sol,iter = iterate(f,g,x0, y0, iterations,tol)
    print("Problem 1a)")
    print("It converged to",sol[:,-1],"")
    print("It converged in ",iter,"iterations")

    tol = 1e-10
    x0 = np.array([1,1])

    t = time.time()
    for j in range(50):
        [xstar,ier,its] = Newton(x0,tol,iterations)
    elapsed = time.time()-t
    print(xstar)
    print('Newton: the error message reads:',ier)
    print('Newton: took this many seconds:',elapsed/50)
    print('Netwon: number of iterations is:',its)


def iterate(f,g,x0, y0, num_iterations,tol):
    X = np.zeros((2, num_iterations+1))
    X[:, 0] = np.array([x0, y0])

    
    for n in range(num_iterations):
        fn = f(X[0, n], X[1, n])
        gn = g(X[0, n], X[1, n])
        
        # Update X_{n+1} using the iteration formula
        X[:, n+1] = X[:, n] - np.matmul(
            np.array([[1/6, 1/18], [0, 1/6]]),
            np.array([fn, gn])
        )
          # Check for convergence based on the change in x and y
        change_in_x = np.abs(X[0, n + 1] - X[0, n])
        change_in_y = np.abs(X[1, n + 1] - X[1, n])
        if change_in_x < tol and change_in_y < tol:
            return X[:, :n + 2], n + 1

    return X, num_iterations


def evalF(x):
    F = np.zeros(2)
    F[0] = 3*x[0]**2 - x[1]**2
    F[1] = 3*x[0]*x[1]**2 - x[0]**3 - 1

    return F

def evalJ(x):
    J = np.array([[6*x[0],-2*x[1]] , [3*x[1]**2 - 3*x[0]**2,6*x[0]*x[1]]])

    return J

def Newton(x0,tol,Nmax):
    for its in range(Nmax):
        J = evalJ(x0)
        Jinv = inv(J)
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
        if (norm(x1-x0) < tol):
            xstar = x1
            ier =0
            return[xstar, ier, its]
        x0 = x1
    xstar = x1
    ier = 1
    return[xstar,ier,its]

        
  
driver()
    
    



