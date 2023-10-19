import numpy as np
import matplotlib.pyplot as plt

def driver():
    f = lambda x: 1 / (1 + (10 * x) ** 2)
    a = -1
    b = 1
    Neval = 1000
    xeval = np.linspace(a, b, Neval)
    Nint = 6
    yeval = eval_lin_spline(xeval, a, b, f, Nint)

    M = evaluateM(xeval, yeval, Neval)

    yeval_cubic = eval_cubic_spline(xeval, a, b, f, Nint, M)
    fex = np.zeros(Neval)
    for j in range(Neval):
        fex[j] = f(xeval[j])
    plt.figure()
    plt.plot(xeval, fex, label="Actual")
    plt.plot(xeval, yeval_cubic, label="Cubic Approximation")
    plt.legend()
    plt.show()

def eval_lin_spline(xeval, a, b, f, Nint):
    xint = np.linspace(a, b, Nint + 1)
    yeval = np.zeros(len(xeval))

    for jint in range(Nint):
        a1 = xint[jint]
        fa1 = f(a1)
        b1 = xint[jint + 1]
        fb1 = f(b1)

        condition = (xeval >= a1) & (xeval <= b1)
        ind = np.where(condition)[0]

        n = len(ind)

        for kk in range(n):
            x0 = a1
            f_x0 = fa1
            x1 = b1
            f_x1 = fb1
            yeval[ind[kk]] = evaluate_line(x0, f_x0, x1, f_x1, xeval[ind[kk]])

    return yeval

def evaluate_line(x0, f_x0, x1, f_x1, x):
    m = (f_x1 - f_x0) / (x1 - x0)
    b = f_x0 - m * x0
    y = m * x + b
    return y

def eval_cubic_spline(xeval, a, b, f, Nint, M):
    xint = np.linspace(a, b, Nint + 1)
    yeval = np.zeros(len(xeval))

    for jint in range(Nint):
        a1 = xint[jint]
        fa1 = f(a1)
        b1 = xint[jint + 1]
        fb1 = f(b1)

        condition = (xeval >= a1) & (xeval <= b1)
        ind = np.where(condition)[0]

        for kk in ind:
            x0 = a1
            f_x0 = fa1
            x1 = b1
            f_x1 = fb1
            yeval[kk] = evaluate_cub(x0, f_x0, x1, f_x1, xeval[kk], M, jint)

    return yeval

def evaluate_cub(x0, f_x0, x1, f_x1, x, M, jint):
    h = x1 - x0
    hi = h
    C = f_x0 / hi - hi * M[jint] / 6
    D = f_x1 / hi - hi * M[jint + 1] / 6
    s = ((x1 - x)**3 * M[jint]) / (6 * hi) + ((x - x0)**3 * M[jint + 1]) / (6 * hi) + C * (x1 - x) + D * (x - x0)
    return s

def evaluateM(xeval, yeval, Nint):
    h = (xeval[-1] - xeval[0])
    matrix = np.zeros([Nint + 1, Nint + 1])

    for i in range(Nint + 1):
        matrix[i, i] = 1 / 3
        if i > 0:
            matrix[i, i - 1] = 1 / 12
        if i < Nint:
            matrix[i, i + 1] = 1 / 12

    y = np.zeros(Nint + 1)
    for k in range(Nint-1):  # Corrected the loop range
        if k == 1:
            y[k] = (yeval[k - 1] - 2 * yeval[k] + yeval[k + 1]) / (h ** 2)
        else:
            y[k] = (yeval[k - 1] - 2 * yeval[k] + yeval[k + 1]) / (h ** 2)

    M = np.linalg.solve(matrix[1:Nint, 1:Nint], y[1:Nint])
    return M

driver()
