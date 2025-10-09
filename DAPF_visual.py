import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# 2D Gaussian function parameters
A = 1.0  # peak amplitude
x0, y0 = -2.0, -2.0  # center of the Gaussian
sigma_x, sigma_y = 2.0, 2.0  # standard deviations

# Create a grid of (x, y) values
x = np.linspace(-6, 6, 100)
y = np.linspace(-6, 6, 100)
x, y = np.meshgrid(x, y)


# Function to apply directional scaling along [cos(theta), sin(theta)]
def rotated_and_scaled_gaussian_2d(x, y, theta, lambda_factor, x0, y0, sigma_x, sigma_y, A):
    # Rotation matrix components
    ct = np.cos(theta)
    st = np.sin(theta)
    # cg = np.cos(gamma)
    # sg = np.sin(gamma)
    # Rotate the coordinates (x, y) -> (x', y')
    x_prime = ct * x + st * y
    y_prime = -st * x + ct * y

    # x_prime_ = cg * x + sg * y
    # y_prime_ = -sg * x + cg * y

    # Apply scaling in the x' direction (for positive x')
    x_prime_scaled = np.where(x_prime > 0, lambda_factor * x_prime, x_prime)
    # x_prime_scaled_ = np.where(x_prime_ > 0, lambda_factor * x_prime_, x_prime_)

    # Calculate the Gaussian in the rotated and scaled space
    z1 = A * np.exp(-(x_prime_scaled ** 2 / (2 * sigma_x ** 2) + y_prime ** 2 / (2 * sigma_y ** 2)))
    # z2 = A * np.exp(-((x_prime_scaled_ - x0) ** 2 / (2 * sigma_x ** 2) + (y_prime_ - y0) ** 2 / (2 * sigma_y ** 2)))
    return z1


# Parameters
theta = 5 * np.pi / 6  # 45 degree angle (direction along [cos(45°), sin(45°)])
gamma = 2 * np.pi / 3
lambda_factor = 0.4  # Stretching factor

# Compute z values for the rotated and scaled Gaussian function
za = rotated_and_scaled_gaussian_2d(x, y, theta, lambda_factor, x0, y0, sigma_x, sigma_y, A)
zb = rotated_and_scaled_gaussian_2d(x, y, gamma, lambda_factor, x0, y0, sigma_x, sigma_y, A)

# Plotting the surface of the rotated and scaled Gaussian function
fig = plt.figure(figsize=(16, 6))
ax2 = fig.add_subplot(122, projection='3d')
ax1 = fig.add_subplot(121)
ax1: plt.Axes = sns.heatmap((za + zb) / 2, cmap="plasma", ax=ax1, robust=True)
ax2.plot_surface(x, y, (za + zb) / 2, cmap='plasma', edgecolor='none')
ax1.set_xticks([])
ax1.set_yticks([])

# Labels and title
ax2.set_title('2D Gaussian Function with Stretching Along Direction [cos(theta), sin(theta)]')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

plt.show()
