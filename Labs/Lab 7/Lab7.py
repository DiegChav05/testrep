import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
def driver():
    f = lambda x: 1 / (1 + (10 * x) ** 2)
    N = 10

    h = 2 / (N - 1)
    xint = np.array([-1 + (j - 1) * h for j in range(1, N + 1)])
    yint = f(xint)
    Neval = 1000
    xeval = np.linspace(-1, 1, Neval + 1)

    a = -1
    b = 1

    yeval_l = np.zeros(Neval + 1)
    yeval_dd = np.zeros(Neval + 1)
    yeval_van = np.zeros(Neval + 1)
    y = np.zeros((N, N + 1))

    for j in range(N):
        y[j][0] = yint[j]

    y = dividedDiffTable(xint, y, N)




    x1 = np.linspace(-1, 1, 3)
    y1 = 1 / (1 + (10 * x1) ** 2)



    coefficients, evaluation = Vandermond(x1, y1, xeval)






    #coefficients, evaluation = Vandermond(x, y, xeval1)

    
    fex = f(xeval)

    # Calculate the absolute error at each point


    plt.figure()
    plt.plot(xeval, yeval_l, 'bs--', label='Lagrange')
    plt.plot(xeval, yeval_dd, 'c.--', label='Newton DD')
    plt.plot(xeval, evaluation, 'g.-', label='Vandermonde')
    plt.legend()
    plt.title('Interpolation Results differnet nodes')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    
    plt.figure()
    err_l = abs(yeval_l - fex)
    err_dd = abs(yeval_dd - fex)
    err_vand = abs(yeval_van - fex)
    plt.semilogy(xeval, err_l, 'ro--', label='Lagrange')
    plt.semilogy(xeval, err_dd, 'bs--', label='Newton DD')
    plt.semilogy(xeval, err_vand, 'g.--', label='Vandermonde')
    plt.legend()
    plt.title('Error Plot different nodes')
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.show() 

    


def vandermonde_matrix(x):
    n = len(x)
    vander_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            vander_matrix[i][j] = x[i] ** j

    return vander_matrix

def Vandermond(x, y, xeval):
    n = len(x)
    vander_matrix = vandermonde_matrix(x)

    coefficients = np.linalg.solve(vander_matrix, y)  # Solve the linear system to obtain coefficients

    # Evaluate the polynomial using Horner's method
    evaluation = np.zeros(len(xeval))
    for i in range(len(xeval)):
        for j in range(n):
            evaluation[i] += coefficients[j] * (xeval[i] ** j)

    return coefficients, evaluation


def eval_lagrange(xeval,xint,yint,N):
    lj = np.ones(N+1)
    for count in range(N+1):
        for jj in range(N+1):
            if (jj != count):
                lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])
    yeval = 0.
    for jj in range(N+1):
        yeval = yeval + yint[jj]*lj[jj]
    return(yeval)
#''' create divided difference matrix'''
def dividedDiffTable(x, y, n):
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                                     (x[j] - x[i + j]));
    return y;

def evalDDpoly(xval, xint,y,N):
#''' evaluate the polynomial terms'''
    ptmp = np.zeros(N+1)
    ptmp[0] = 1.
    for j in range(N):
        ptmp[j+1] = ptmp[j]*(xval-xint[j])
#'''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N+1):
        yeval = yeval + y[0][j]*ptmp[j]
    return yeval

driver()

