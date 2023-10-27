import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm

def driver():
    a = 0
    b = 2 * np.pi
    Nint = 100  # Number of intervals
    xint = np.linspace(a, b, Nint + 1)
    yint = np.sin(10 * xint)  

    (M, C, D) = create_naturally_periodic_spline(yint, xint, Nint)

    Neval = 100  # Number of evaluation points
    xeval = np.linspace(a, b, Neval + 1)
    yeval = eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D)

    fex = np.sin(10 * xeval)
    nerr = norm(fex - yeval)
    print('nerr =', nerr)

    plt.figure()
    plt.plot(xeval, fex, 'ro-', label='exact function')
    plt.plot(xeval, yeval, 'bs--', label='natural cubic spline')
    plt.legend()
    plt.show()

    err = abs(yeval - fex)
    plt.figure()
    plt.semilogy(xeval, err, 'ro--', label='absolute error')
    plt.legend()
    plt.show()

def create_naturally_periodic_spline(yint, xint, N):
    M, C, D = create_natural_spline(yint, xint, N)

    # Ensure the second derivative at both ends is equal
    M[N] = M[0]

    return M, C, D
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
driver()
