import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv
from numpy.linalg import norm
def driver():
    f = lambda x: (1) / (1 + x**2)
    a = -5
    b = 5

    Nint_vec = [5, 10, 15, 20]  # Vector of Number of intervals
    Neval = 100  # Number of evaluation points
    xeval = np.linspace(a, b, Neval + 1)

    for Nint in Nint_vec:
        # Chebyshev nodes on the interval [a, b]
        #k = np.arange(0, Nint+1)
        #xint = 0.5 * (a + b) - 0.5 * (b - a) * np.cos((2 * k + 1) * np.pi / (2 * (Nint+1)))
        xint = np.linspace(a, b, Nint + 1)
        yint = f(xint)

        (M, C, D) = create_natural_spline(yint, xint, Nint)
        
        print(f'M for Nint={Nint}:\n', M)

        yeval = eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D)

        fex = f(xeval)
        nerr = norm(fex - yeval)
        print(f'nerr for Nint={Nint}:', nerr)

        plt.figure()
        plt.plot(xeval, fex, 'ro-', label='exact function')
        plt.plot(xeval, yeval, 'bs--', label=f'natural spline (Nint={Nint})')
        plt.legend()
        plt.show()

        err = abs(yeval - fex)
        plt.figure()
        plt.semilogy(xeval, err, 'ro--', label=f'absolute error (Nint={Nint})')
        plt.legend()
        plt.show()

def create_natural_spline(yint,xint,N):
# create the right hand side for the linear system
    b = np.zeros(N+1)
# vector values
    h = np.zeros(N+1)

    for i in range(1,N):
        hi = xint[i]-xint[i-1]
        hip = xint[i+1] - xint[i]
        b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
        h[i-1] = hi
        h[i] = hip
# create matrix so you can solve for the M values
# This is made by filling one row at a time
    A = np.zeros((N+1,N+1))
    A[0][0] = 1.0
    for j in range(1,N):
        A[j][j-1] = h[j-1]/6
        A[j][j] = (h[j]+h[j-1])/3
        A[j][j+1] = h[j]/6
    A[N][N] = 1
    Ainv = inv(A)
    M = Ainv.dot(b)
# Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
        C[j] = yint[j]/h[j]-h[j]*M[j]/6
        D[j] = yint[j+1]/h[j]-h[j]*M[j+1]/6
    return(M,C,D)
def eval_local_spline(xeval,xi,xip,Mi,Mip,C,D):
# Evaluates the local spline as defined in class
# xip = x_{i+1}; xi = x_i
# Mip = M_{i+1}; Mi = M_i
    hi = xip-xi
    yeval = (Mi*(xip-xeval)**3 +(xeval-xi)**3*Mip)/(6*hi) \
    + C*(xip-xeval) + D*(xeval-xi)
    return yeval



def create_clamped_spline(yint, xint, N):
    b = np.zeros(N + 1)
    h = np.zeros(N + 1)

    for i in range(1, N):
        hi = xint[i] - xint[i - 1]
        hip = xint[i + 1] - xint[i]
        b[i] = (yint[i + 1] - yint[i]) / hip - (yint[i] - yint[i - 1]) / hi
        h[i - 1] = hi
        h[i] = hip

    A = np.zeros((N + 1, N + 1))
    A[0][0] = 2 * (xint[1] - xint[0])
    A[0][1] = h[0]
    b[0] = 6 * ((yint[1] - yint[0]) / (xint[1] - xint[0]) - 0)  # Clamped derivative at x = 0
    
    for j in range(1, N):
        A[j][j - 1] = h[j - 1]
        A[j][j] = 2 * (h[j - 1] + h[j])
        A[j][j + 1] = h[j]
    
    A[N][N - 1] = h[N - 1]
    A[N][N] = 2 * (xint[N] - xint[N - 1])
    b[N] = 6 * (0 - (yint[N] - yint[N - 1]) / (xint[N] - xint[N - 1]))  # Clamped derivative at x = 1
    
    Ainv = inv(A)
    M = Ainv.dot(b)
    
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
        C[j] = yint[j] / h[j] - h[j] * M[j] / 6
        D[j] = yint[j + 1] / h[j] - h[j] * M[j + 1] / 6
    
    return M, C, D

def eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D):
    yeval = np.zeros(Neval + 1)
    
    for j in range(Nint):
        atmp = xint[j]
        btmp = xint[j + 1]
        ind = np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]
        yloc = eval_local_spline(xloc, atmp, btmp, M[j], M[j + 1], C[j], D[j])
        yeval[ind] = yloc
    
    return yeval

def eval_local_spline(xeval, xi, xip, Mi, Mip, Ci, Di):
    hi = xip - xi
    yeval = (Mi * (xip - xeval) ** 3 + (xeval - xi) ** 3 * Mip) / (6 * hi) + Ci * (xip - xeval) + Di * (xeval - xi)
    return yeval
def create_naturally_periodic_spline(yint, xint, N):
    M, C, D = create_natural_spline(yint, xint, N)
    
    # Ensure the second derivative at both ends is equal
    M[0] = M[N]
    
    return M, C, D
driver()