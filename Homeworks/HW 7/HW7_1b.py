import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
def driver():
    f = lambda x: 1 / (1 + (10 * x) ** 2)
    N = 10

    h = 2 / (N - 1)
    xint = np.array([-1 + (j - 1) * h for j in range(1, N + 1)])
    yint = f(xint)
    Neval = 1001
    xeval = np.linspace(-1, 1, Neval + 1)

    a = -1
    b = 1

    yeval_van = np.zeros(Neval + 1)
    y = np.zeros((N, N + 1))

    for j in range(N):
        y[j][0] = yint[j]


    x1 = np.linspace(-1, 1, N)
    y1 = 1 / (1 + (10 * x1) ** 2)



    coefficients, evaluation = Vandermond(x1, y1, xeval)


    #coefficients, evaluation = Vandermond(x, y, xeval1)

    

    # Calculate the absolute error at each point

# Plot data points as circles
    plt.figure()
    # Generate a finer grid for plotting the polynomial and f(x)
  
    # Plot the polynomial and f(x) on the finer grid
    plt.plot(xeval, evaluation, 'o', label='Interpolated Polynomial')
    plt.plot(xeval, f(xeval), 'r-', label='f(x)')
    plt.plot(xint, yint, 'o', label='Data Points')


    plt.legend()
    plt.title('Interpolation Results with Data Points')
    plt.xlabel('x')
    plt.ylabel('f(x)')

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

driver()