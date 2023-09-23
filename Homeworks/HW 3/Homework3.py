import numpy as np
from matplotlib import pyplot as plt 

def driver():
    # use routines    
    f = lambda x: np.sin(x) -2*x +1
    a = 0
    b = np.pi

    tol = 1e-7

    [astar, ier, num_iterations] = bisection(f, a, b, tol)
    print('Question 1c)')
    print('the approximate root is', '{:.8f}'.format(astar))
    #print('the error message reads:', ier)
    #print('f(astar) =', f(astar))
    print('Number of iterations:', '{:.8f}'.format(num_iterations))

    a = 4.82
    b = 5.2

    f = lambda x: (x-5)**9

    tol = 1e-4

    [astar, ier, num_iterations] = bisection(f, a, b, tol)
    print('Question 2a)')
    print('the approximate root is', astar)
    print('the error message reads:', ier)
    #print('f(astar) =', f(astar))
    print('Number of iterations:', '{:.8f}'.format(num_iterations))

    #Expanded version of (x-5)^9

    f = lambda x: x**9 - 45*x**8 + 900*x**7 - 12600*x**6 + 126000*x**5 - 907200*x**4 + 4536000*x**3 - 15120000*x**2 + 30240000*x - 1953125


    [astar, ier, num_iterations] = bisection(f, a, b, tol)
    print('Question 2b)')
    print('the approximate root is', astar)
    print('the error message reads:', ier)
    #print('f(astar) =', f(astar))
    print('Number of iterations:', '{:.8f}'.format(num_iterations))

    print('Question 5a)')
    print('There are 5 zero crossings')

    ####
    #Question 5
    x = np.arange(-1*np.pi, 3 * np.pi, 0.1)  
    y = x - (4 * np.sin(2*x)) - 3

  
    plt.title("Question 5a") 
    plt.xlabel("x axis") 
    plt.ylabel("y axis") 
    plt.axhline(y = 0, color = 'k', linestyle = '-')
    plt.plot(x, y) 
    plt.show()

    #f1 = lambda x: - np.sin(2*x) + (5*x)/4 - 3/4  

    g = lambda x: -np.sin(2 * x) + 5 * x / 4 - 3 / 4

    initial_guess = 1.0  # You can choose a different initial guess
    tolerance = 1e-8
    max_iterations = 1000

    root, iterations = fixed_point(g,initial_guess, tolerance, max_iterations)

    if root is not None:
        print('the approximate root is', '{:.10f}'.format(-root))
    else:
        print("Root not found within the maximum number of iterations.")

        g = lambda x: -np.sin(2 * x) + 5 * x / 4 - 3 / 4

    initial_guess = .9  # You can choose a different initial guess
    tolerance = 1e-12
    max_iterations = 1000

    root, iterations = fixed_point(g,initial_guess, tolerance, max_iterations)

    if root is not None:
        print('the approximate root is', '{:.10f}'.format(-root))
    else:
        print("Root not found within the maximum number of iterations.")






# define routines
def bisection(f, a, b, tol):
    """
    Inputs:
        f, a, b   - function and endpoints of the initial interval
        tol       - bisection stops when the interval length < tol

    Returns:
        astar     - approximation of the root
        ier       - error message
                    - ier = 1 => Failed
                    - ier = 0 == success
        num_iterations - number of iterations taken to find the root
    """

    # first verify there is a root we can find in the interval
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        ier = 1
        astar = a
        num_iterations = 0
        return [astar, ier, num_iterations]

    # verify endpoints are not a root
    if fa == 0:
        astar = a
        ier = 0
        num_iterations = 0
        return [astar, ier, num_iterations]

    if fb == 0:
        astar = b
        ier = 0
        num_iterations = 0
        return [astar, ier, num_iterations]

    count = 0
    d = 0.5 * (a + b)
    while abs(d - a) > tol:
        fd = f(d)
        if fd == 0:
            astar = d
            ier = 0
            num_iterations = count
            return [astar, ier, num_iterations]
        
        if fa * fd < 0:
            b = d
        else:
            a = d
            fa = fd
        d = 0.5 * (a + b)
        count += 1

    astar = d
    ier = 0
    num_iterations = count
    return [astar, ier, num_iterations]

def fixed_point(g,initial_guess, tolerance, max_iterations):
    x = initial_guess
    for i in range(max_iterations):
        x_next = g(x)
        if abs(x_next - x) < tolerance:
            return x_next, i+1  # Return the root and the number of iterations
        x = x_next

    return None, max_iterations  # Return None if max_iterations reached




driver()
