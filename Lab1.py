import numpy as np
import numpy.linalg as la
import math
import time



def driver():

    n = 100
    x = np.linspace(0,np.pi,n)
# this is a function handle. You can use it to define
# instead of using a subroutine like you
# have to in a true low level language.
    f = lambda x: np.sin(x)
    g = lambda x: np.cos(x)
    y = f(x)
    w = g(x)
# evaluate the dot product of y and w
    tic = time.perf_counter()
    dp = dotProduct(y,w,n)
    toc = time.perf_counter()
# print the output
    print('the dot product is : ', dp)
#return

    A = [[1,2],[1,2]]
    B = [[1,2],[1,2]]

    result = matrix_mult(A,B)

    tic2 = time.perf_counter()
    print("Numpy dot product is",np.dot(y,w))
    toc2 = time.perf_counter()

    print(np.matmul(A,B))

    print("Homemade dot product time is",toc-tic,'s')
    print("Numpy dot product time is", toc2-tic2, 's')
    print("Numpy is faster")

 

    


def matrix_mult(A, B):
    #Initializing and creating the result matrix
    result = []
    for row in range(len(A)):
        row_it = []
        for col in range(len(B[0])):
            #append adds last row_it to the end of the matrix
            row_it.append(0)
        result.append(row_it)
    #This loop iterates of the rows of A
    for i in range(len(A)):
        #This loop iterates of the colomns of B
        for j in range(len(B[0])):
            curr_val = 0
            #This loop iterates of the inner dimention 
            for k in range(len(A[0])):
                # += adds the past value 
                curr_val += A[i][k]*B[k][j]
            result[i][j] = curr_val
    print(result)

def dotProduct(x,y,n):
    dp = 0
    for j in range(n):
        dp = dp + x[j]*y[j]
    return dp

driver()
