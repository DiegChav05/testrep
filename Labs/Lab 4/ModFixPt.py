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
    print("Fixed point converges in:", len(xstar_approximations))
    print('Error message reads:', ier)

    accelerated_sequence = aitkens_method(xstar_approximations, tol, Nmax)
    print("Accelerated sequence:", accelerated_sequence)
    print("Accelerated sequence converges in:", len(accelerated_sequence))

def aitkens_method(approximations, tol, Nmax):
   
    n = len(approximations)
    
    accelerated_sequence = np.zeros(n-2)  # Store the accelerated sequence
    
    for i in range(n-2):
        a_i = approximations[i]
        a_ip1 = approximations[i+1]
        a_ip2 = approximations[i+2]
        
        # Aitken's formula to calculate the accelerated sequence
        accelerated_sequence[i] = a_i - ((a_ip1 - a_i)**2) / (a_ip2 - 2*a_ip1 + a_i)
    
    # Check for convergence using tolerance
    for i in range(Nmax):
        if np.abs(accelerated_sequence[-1] - accelerated_sequence[-2]) < tol:
            break
        
        a_i = accelerated_sequence[-2]
        a_ip1 = accelerated_sequence[-1]
        a_ip2 = accelerated_sequence[-2]
        
        # Update the accelerated sequence
        accelerated_sequence = np.append(accelerated_sequence, a_i - ((a_ip1 - a_i)**2) / (a_ip2 - 2*a_ip1 + a_i))
    
    return accelerated_sequence



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