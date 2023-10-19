import numpy as np
import matplotlib.pyplot as plt
import math

def driver():
    f = lambda x: 1/(1+(10*x)**2)
    a = -1
    b = 1
    Neval = 1000
    xeval = np.linspace(a, b, Neval)
    Nint = 10
    yeval = eval_lin_spline(xeval, Neval, a, b, f, Nint)
    #yeval_cubic = eval_cubic_spline(xeval, a, b, f, Nint,M)
    fex = np.zeros(Neval)
    for j in range(Neval):
        fex[j] = f(xeval[j])

    plt.figure()
    plt.plot(xeval, fex)
    plt.title("Actual")
    plt.plot(xeval, yeval)
    plt.title("Linear Approximation")
    #plt.plot(xeval, yeval_cubic)
    #plt.title("Cubic Approximation")
    plt.legend()
    plt.show()

    err = abs(yeval - fex)
    #err_cubic = abs(yeval_cubic - fex)
    plt.figure()
    plt.plot(xeval, err)
    plt.title("Linear Error")
    #plt.plot(xeval, err_cubic)
    #plt.title("Cubic Error")
    plt.show()

def eval_lin_spline(xeval, Neval, a, b, f, Nint):
    xint = np.linspace(a, b, Nint + 1)
    yeval = np.zeros(Neval)

    for jint in range(Nint):
        a1 = xint[jint]
        fa1 = f(a1)
        b1 = xint[jint + 1]
        fb1 = f(b1)

        # Find indices of xeval in the interval (xint(jint), xint(jint+1))
        condition = (xeval >= a1) & (xeval <= b1)
        ind = np.where(condition)[0]

        # Find the length of ind
        n = len(ind)

        for kk in range(n):
            x0 = a1
            f_x0 = fa1
            x1 = b1
            f_x1 = fb1
            # Use your line evaluator to evaluate the lines at each of the points in the interval
            yeval[ind[kk]] = evaluate_line(x0, f_x0, x1, f_x1, xeval[ind[kk]])

    return yeval


def find_points_in_subintervals(xeval, xint):
    indices = []
    for i in range(len(xint) - 1):
        condition = (xeval >= xint[i]) & (xeval < xint[i + 1])
        interval_indices = np.where(condition)[0]
        indices.append(interval_indices)
    return indices

def evaluate_line(x0, f_x0, x1, f_x1, x):
    # Construct the line passing through (x0, f_x0) and (x1, f_x1)
    m = (f_x1 - f_x0) / (x1 - x0)  # Slope of the line
    b = f_x0 - m * x0  # Intercept of the line
    y = m * x + b  # Line equation
    return y







driver()