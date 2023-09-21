import numpy as np

def driver():
    # test functions 
    f1 = lambda x: 1 + 0.5 * np.sin(x)  # fixed point is alpha1 = 1.4987
    f2 = lambda x: 3 + 2 * np.sin(x)  # fixed point is alpha2 = 3.09
    Nmax = 100
    tol = 1e-6

    # test f1
    x0 = 0.0
    xstar_approximations, ier = fixedpt(f1, x0, tol, Nmax)
    print('Approximate fixed point iterations for f1:', xstar_approximations)
    print('f1(xstar):', f1(xstar_approximations[-1]))
    print('Error message reads:', ier)

    # test f2
    x0 = 0.0
    xstar_approximations, ier = fixedpt(f2, x0, tol, Nmax)
    print('Approximate fixed point iterations for f2:', xstar_approximations)
    print('f2(xstar):', f2(xstar_approximations[-1]))
    print('Error message reads:', ier)

# Modified the fixedpt function to return a list of approximations
def fixedpt(f, x0, tol, Nmax):
    '''
    x0 = initial guess
    Nmax = max number of iterations
    tol = stopping tolerance
    '''

    xstar_approximations = [x0]  # Initialize with the initial guess
    count = 0

    while count < Nmax:
        count += 1
        x1 = f(x0)
        xstar_approximations.append(x1)  # Append the current approximation to the list

        if abs(x1 - x0) < tol:
            ier = 0
            return xstar_approximations, ier

        x0 = x1

    ier = 1
    return xstar_approximations, ier

driver()