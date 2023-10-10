import numpy as np
import matplotlib.pyplot as plt

def newton_systems():
    # Apply Newton's method to find the roots of f(z) = z^3-1

    col = 100  # Number of Newton iterations
    m = 2000  # Plot resolution
    cx = 0  # Center of plot (real)
    cy = 0  # Center of plot (imaginary)
    l = 1  # box size
    x = np.linspace(cx - l, cx + l, m)
    y = np.linspace(cy - l, cy + l, m)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    c = -0.5 * (1 + 1j * np.sqrt(3))

    for k in range(col):  # Loop over the Newton iterations
        Z = 2/3 * Z + 1/3 * 1 / (np.finfo(float).eps + Z**2)

    W = np.abs(Z - c)
    A = np.angle(Z)
    plt.colormaps()
    plt.pcolor(W - A, cmap='prism')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    newton_systems()
