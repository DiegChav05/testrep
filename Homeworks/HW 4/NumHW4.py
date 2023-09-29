import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import special

def driver():
# Given constants
    Ts = -15
    Ti = 20
    alpha = 0.138e-6
    t = 60 * 24 * 3600  # 60 days in seconds

# Define the function f(x)
    f = lambda x: ((Ti - Ts) * special.erf(x / (2 * math.sqrt(alpha * t)))) + Ts


    x = special.erfinv(15/35)*2*np.sqrt(alpha*t)
    print("The distance is",x,"m")

    x_bar = 10  
    a = 0
    tol = 10e-13
    [astar, ier] = bisection(f,a,x_bar,tol)
    print("Bisection approximation is ",astar)

    p0 = .01
    fp =  lambda x: ((Ti - Ts) / (np.sqrt(np.pi * alpha * t))) * np.exp(-(x / (2 * np.sqrt(alpha * t)))**2)
    Nmax = 10000

    [p,pstar,info,it] = newton(f,fp,p0,tol,Nmax)
    print("Newtons approximation is ",pstar)

    p0 = x_bar
    [p,pstar,info,it] = newton(f,fp,p0,tol,Nmax)
    print("Newtons approximation with xbar is ",pstar)




# Plot f(x) over the range [0, x_bar]

    x_values = np.linspace(0, x_bar, 1000)
    f_values = f(x_values)


    plt.plot(x_values, f_values)
    plt.xlabel('x (meters)')
    plt.ylabel('f(x)')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.title('Plot of f(x)')
    plt.grid(True)
    plt.show()

    f = lambda x: x**6-x-1
    fp = lambda x: 6*x**5-1
    p0 = 2

    [x_next, errors_newton] = newton_method_er(p0,f,fp, tol, Nmax)

    x0 = 2
    x1 = 1

    [x_next, errors_secant] = secant_method(f,x0, x1, tol, Nmax)

    print("Newton's Method:")
    print("Step | Error")
    print("-----------------")
    for i, error in enumerate(errors_newton):
        print(f"{i+1:4d} | {error:.10f}")

    print("\nSecant Method:")
    print("Step | Error")
    print("-----------------")
    for i, error in enumerate(errors_secant):
        print(f"{i+1:4d} | {error:.10f}")

    errors_newton_shifted = np.array(errors_newton[:-1])
    errors_secant_shifted = np.array(errors_secant[:-1])

    errors_newton_shifted_next = np.array(errors_newton[1:])
    errors_secant_shifted_next = np.array(errors_secant[1:])

    # Plot on log-log axes
    plt.figure(figsize=(10, 6))
    plt.loglog(errors_newton_shifted, errors_newton_shifted_next, label="Newton's Method")
    plt.loglog(errors_secant_shifted, errors_secant_shifted_next, label="Secant Method")

# Add labels and legend
    plt.xlabel(r'$|x_{k} - \alpha|$')
    plt.ylabel(r'$|x_{k+1} - \alpha|$')
    plt.legend()
    plt.title('Convergence of Newton\'s Method and Secant Method')
    plt.grid(True)
    plt.show()


def bisection(f,a,b,tol):
# Inputs:
# f,a,b - function and endpoints of initial interval
# tol - bisection stops when interval length < tol
# Returns:
# astar - approximation of root
# ier - error message
# - ier = 1 => Failed
# - ier = 0 == success
# first verify there is a root we can find in the interval
    fa = f(a)
    fb = f(b)
    if (fa*fb>0):
        ier = 1
        astar = a
        return [astar, ier]
    # verify end points are not a root
    if (fa == 0):
        astar = a
        ier =0
        return [astar, ier]
    if (fb ==0):
        astar = b
        ier = 0
        return [astar, ier]
    count = 0
    d = .5*(a+b)

    while (abs(d-a)> tol):
        fd = f(d)
        if (fd ==0):
            astar = d
            ier = 0
            return [astar, ier]
        if (fa*fd<0):
            b = d
        else:
            a = d
            fa = fd
        d = 0.5*(a+b)
        count = count +1
# print('abs(d-a) = ', abs(d-a))
    astar = d
    ier = 0
    return [astar, ier]

def newton(f,fp,p0,tol,Nmax):

    p = np.zeros(Nmax+1);
    p[0] = p0
    for it in range(Nmax):
        p1 = p0-f(p0)/fp(p0)
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it]

def newton_method_er(x0,f,fp, tol, max_iter):
    errors = []
    x = x0
    
    for i in range(max_iter):
        x_next = x - f(x) / fp(x)
        error = abs(x_next - x)
        errors.append(error)
        
        if error < tol:
            break
        
        x = x_next
    
    return x_next, errors

def secant_method(f,x0, x1, tol, max_iter):
    errors = []
    x_prev = x0
    x = x1
    
    for i in range(max_iter):
        x_next = x - f(x) * (x - x_prev) / (f(x) - f(x_prev))
        error = abs(x_next - x)
        errors.append(error)
        
        if error < tol:
            break
        
        x_prev = x
        x = x_next
    
    return x_next, errors

driver()




    
   