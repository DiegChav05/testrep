import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math
def driver():
    f = lambda x: 1./(1.+x**2)
    fp = lambda x: -2*x/(1.+x**2)**2
    N = 20

    a = -5
    b = 5
    
    # Generate Chebyshev nodes
    k = np.arange(0, N+1)
    xint = 0.5 * (a + b) - 0.5 * (b - a) * np.cos((2 * k + 1) * np.pi / (2 * (N+1)))

    #''' create interpolation data'''
    yint = np.zeros(N+1)
    ypint = np.zeros(N+1)
    for jj in range(N+1):
        yint[jj] = f(xint[jj])
        ypint[jj] = fp(xint[jj])
    #''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 100
    xeval = np.linspace(a,b,Neval+1)
    yevalL = np.zeros(Neval+1)
    yevalH = np.zeros(Neval+1)
    for kk in range(Neval+1):
        yevalL[kk] = eval_lagrange(xeval[kk],xint,yint,N)
        yevalH[kk] = eval_hermite(xeval[kk],xint,yint,ypint,N)
#''' create vector with exact values'''
    fex = np.zeros(Neval+1)
    for kk in range(Neval+1):
        fex[kk] = f(xeval[kk])
    plt.figure()
    plt.plot(xeval,fex,'ro-',label='Actual')
    plt.plot(xeval,yevalL,'bs--',label='Lagrange')
    plt.plot(xeval,yevalH,'c.--',label='Hermite')
    plt.semilogy()
    plt.legend(loc='upper right')
    plt.show()
    errL = abs(yevalL-fex)
    errH = abs(yevalH-fex)
    plt.figure()
    plt.semilogy(xeval,errL,'bs--',label='Lagrange')
    plt.semilogy(xeval,errH,'c.--',label='Hermite')
    plt.show()

def eval_hermite(xeval,xint,yint,ypint,N):
    #''' Evaluate all Lagrange polynomials'''
    lj = np.ones(N+1)
    for count in range(N+1):
        for jj in range(N+1):
            if (jj != count):
                lpj2 = np.ones(N+1) 
                lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])
    #''' Construct the l_j'(x_j)'''
    lpj = np.zeros(N+1)
    for count in range(N+1):
        for jj in range(N+1):
            if (jj != count):
# lpj2[count] = lpj2[count]*(xint[count] - xint[jj])
                lpj[count] = lpj[count]+ 1./(xint[count] - xint[jj])
    yeval = 0.

    for jj in range(N+1):
        Qj = (1.-2.*(xeval-xint[jj])*lpj[jj])*lj[jj]**2
        Rj = (xeval-xint[jj])*lj[jj]**2
# if (jj == 0):
# print(Qj)
# print(Rj)
# print(Qj)
# print(xeval)
# return
        yeval = yeval + yint[jj]*Qj+ypint[jj]*Rj
    return(yeval)
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


if __name__ == '__main__':
# run the drivers only if this is called from the command line
    driver()
