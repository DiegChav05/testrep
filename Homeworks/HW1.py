import numpy as np
import matplotlib.pyplot as plt

#problem1
#x = np.arange(1.920,2.081,.001) 
#y = x**9 - 18*x**8 + 144*x**7 - 672*x**6 + 2016*x**5 - 4032*x**4 + 5376*x**3 - 4608*x**2 +2304*x - 512
#plt.title("Evaluation via Coefficients") 
#plt.xlabel("x") 
#plt.ylabel("y") 
#plt.plot(x,y) 
#plt.show()

x3 = np.arange(1.920,2.081,.001) 
y3 = (x3-2)**9
plt.xlabel("x") 
plt.ylabel("y") 
plt.plot(x3,y3) 
plt.show()

######5b




x = np.pi

exponents = np.linspace(-16, 0, 16)  # Create a vector of exponents
delta = 10.0 ** exponents  # Compute the corresponding values

# Redefining the difference expression
f = np.cos(x) * np.cos(delta) - np.sin(x) * np.sin(delta) - np.cos(x)

plt.figure()
plt.plot(exponents, f)
plt.title("Error when varying delta (x=pi)")
plt.xlabel("-exponent of delta")
plt.ylabel("Difference between expression")


##Redefining x as a large number
x2 = 10.0 ** 6

exponents2 = np.linspace(-16, 0, 16)  # Create a vector of exponents
delta2 = 10.0 ** exponents2  # Compute the corresponding values

f2 = np.cos(x2) * np.cos(delta2) - np.sin(x2) * np.sin(delta2) - np.cos(x2)

plt.figure()
plt.plot(exponents2, f2)
plt.title("Error when varying delta (x=10^6)")
plt.xlabel("-exponent of delta")
plt.ylabel("Difference between expression")

plt.show()

###### 5c 

#Taylor expansion
ft = delta*(np.sin(x)) + ((delta**2)/2) * -(np.cos(x))

plt.figure()
plt.plot(exponents,ft)
plt.title("New Algorithm difference error)")
plt.xlabel("-exponent of delta")
plt.ylabel("Difference between expression")

plt.show()
