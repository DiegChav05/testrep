import numpy as np
import matplotlib.pyplot as plt
import random


A = .5*np.array([[1,1],
        [1+10**-10,1-10**-10]])

A_inv = np.array([[1-10**10,10**10],
         [1+10**10,-10**10]])

A_norm = np.linalg.norm(A)
A_inv_norm = np.linalg.norm(A_inv)


K = A_inv_norm*A_norm

#print(K)

#######
## Q3
x = 9.999999995000000E-10

# Calculate e^(-x)
result = np.exp(-x)

# Print the result with 16 decimal places
#print(f"e^(-x) ≈ {result:.16f}")


def taylor_approximation(x, n):
    approximation = 0.0

    for i in range(n):
        term = ((-1) ** i) * (x ** i) / np.math.factorial(i)
        approximation += term

    return approximation

x = 9.999999995000000e-10
n = 5  # You can increase n for more accuracy if needed

result = taylor_approximation(x, n)
#print("Approximation:", result)



#####
##Q4a


# Create the t vector starting at 0, incrementing by pi/30, up to pi
t = np.arange(0, np.pi + np.pi/30, np.pi/30)

# Calculate the y vector as cos(t)
y = np.cos(t)

# Calculate the sum S using dot product
#dot product of the two vectors will multipy each component and 
#add them together accomplishing the sum

S = np.dot(t, y)

# Print the result
print(f"The sum is: {S}")




########
##Q4b

# Constants
R = 1.2
delta_r = 0.1
f = 15
p = 0

# Define the parameter range
theta = np.linspace(0, 2 * np.pi, 1000)

# Parametric equations
x = R * (1 + delta_r * np.sin(f * theta + p)) * np.cos(theta)
y = R * (1 + delta_r * np.sin(f * theta + p)) * np.sin(theta)

# Create the plot
plt.figure(figsize=(6, 6))
plt.plot(x, y)
plt.axis('equal')
plt.title("Parametric Curve")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()



# Number of curves
num_curves = 10

# Create a range of values for R, delta_r, f, and p
R_values = np.linspace(1, num_curves, num_curves)
delta_r_values = 0.05
f_values = 2 + R_values
p_values = np.random.uniform(0, 2, num_curves)

# Create subplots
fig, ax = plt.subplots(figsize=(8, 8))

# Plot each curve
for i in range(num_curves):
    theta = np.linspace(0, 2 * np.pi, 1000)
    x = R_values[i] * (1 + delta_r_values * np.sin(f_values[i] * theta + p_values[i])) * np.cos(theta)
    y = R_values[i] * (1 + delta_r_values * np.sin(f_values[i] * theta + p_values[i])) * np.sin(theta)
    
    ax.plot(x, y, label=f"Curve {i+1}")

# Set equal aspect ratio and add legend
ax.axis('equal')
ax.legend()
plt.title("Parametric Curves")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()