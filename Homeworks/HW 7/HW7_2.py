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

    # Evaluate using Barycentric Lagrange
    yeval_barycentric = eval_barycentric_lagrange_p(xeval, xint, yint)


    plt.figure()
    plt.plot(xeval, yeval_barycentric, 'm.-', label='Barycentric Lagrange')
    plt.plot(xeval, f(xeval), '-', label='Actual')
    plt.legend()
    plt.title('Barycentric Lagrange Interpolation')
    plt.xlabel('x')
    plt.ylabel('f(x)')

    plt.show()

def eval_barycentric_lagrange_p(xeval, xint, yint):
    p_x = np.zeros_like(xeval)
    for i in range(len(xeval)):
        x = xeval[i]
        w_num = 0.0
        w_denom = 0.0
        for j in range(len(xint)):
            w = 1.0
            for k in range(len(xint)):
                if k != j:
                    w *= (x - xint[k])
                    w /= (xint[j] - xint[k])
            w_num += w * yint[j]
            w_denom += w
        p_x[i] = w_num / w_denom
    return p_x


# Call the driver function
driver()


    

    