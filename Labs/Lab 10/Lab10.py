import matplotlib.pyplot as plt
import numpy as np
import math
import numpy.linalg as la
from scipy.integrate import quad



def driver():
    # Function you want to approximate
    f = lambda x: math.exp(x)
    
    
    # Interval of interest
    a = -1
    b = 1
    
    # Weight function
    w = lambda x: 1.0
    #w = lambda x: 1 / np.sqrt(1 - x**2)
    
    # Order of approximation
    n = 2
    
    # Number of points you want to sample in [a, b]
    N = 1000
    xeval = np.linspace(a, b, N + 1)
    pval = np.zeros(N + 1)
    
    for kk in range(N + 1):
        pval[kk] = eval_legendre_expansion(f, a, b, w, n, xeval[kk])

    # Create a vector with exact values
    fex = np.zeros(N + 1)

    print(pval)
    
    for kk in range(N + 1):
        fex[kk] = f(xeval[kk])

    plt.figure()
    plt.plot(xeval, fex, 'ro-', label='f(x)')
    plt.plot(xeval, pval, 'bs--', label='Expansion')
    plt.legend()
    plt.show()
    err = abs(pval - fex)
    plt.semilogy(xeval, err, 'ro--', label='error')
    plt.legend()
    plt.show()

def eval_legendre(n, x):
    if n == 0:
        return [1]
    elif n == 1:
        return [1, x]
    else:
        p = [1, x]
        for i in range(2, n + 1):
            phi_n = ((2 * i - 1) * x * p[i - 1] - (i - 1) * p[i - 2]) / i
            p.append(phi_n)
        return p


def eval_legendre_expansion(f, a, b, w, n, x):
    # Evaluate all the Legendre polynomials at x that are needed
    p = eval_legendre(n, x)
    
    # Initialize the sum to 0
    pval = 0.0
    
    for j in range(0, n + 1):
        # Make a function handle for evaluating phi_j(x)
        phi_j = lambda x: eval_legendre(n,x)[j]
        
        # Make a function handle for evaluating phi_j^2(x) * w(x)
        phi_j_sq = lambda x: eval_legendre(n,x)[j] ** 2 * w(x) 
        
        # Use the quad function from scipy to evaluate normalizations
        norm_fac, err = quad(phi_j_sq, a, b)
        
        # Make a function handle for phi_j(x) * f(x) * w(x) / norm_fac
        func_j = lambda x: phi_j(x) * f(x) * w(x) / norm_fac
        
        # Use the quad function from scipy to evaluate coeffs
        aj, err = quad(func_j, a, b)
        
        # Accumulate into pval
        pval = pval + aj * p[j]
    
    return pval

if __name__ == '__main__':
    # Run the drivers only if this is called from the command line
    driver()
