import numpy as np
import matplotlib.pyplot as plt

# Define constants
hbar = 1.0       # Planck's constant / 2pi
m = 1.0          # Electron mass
e = 1.0          # Elementary charge
epsilon_0 = 1.0  # Vacuum permittivity

# Define grid parameters
num_points = 1000  # Number of grid points
grid_spacing = 0.01  # Spacing between grid points

# Create grid
grid = np.linspace(-5, 5, num_points)

# Initialize electron density
rho = np.zeros(num_points)

# Initialize potential (assuming no external potential)
V_ext = np.zeros(num_points)

# Define exchange-correlation functional (simplified form)
def exchange_correlation(rho):
    return - (3 / np.pi)**(1/3) * rho**(1/3)

# Initialize total potential
V_total = V_ext

# Initial guess for the Kohn-Sham orbitals
orbitals = np.zeros((num_points, 1))

# Create figure for visualization
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# Iterative procedure for self-consistency
max_iterations = 100
tolerance = 1e-6
for iteration in range(max_iterations):
    # Update effective potential
    V_eff = V_total + np.dot(rho, exchange_correlation(rho))
    
    # Solve Kohn-Sham equation for orbitals
    kinetic_energy = -hbar**2 / (2 * m) * (1 / grid_spacing**2) * np.diag(np.ones(num_points))
    H = kinetic_energy + np.diag(V_eff)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    orbitals_new = eigenvectors[:, :1]
    
    # Update electron density
    rho_new = np.sum(orbitals_new**2, axis=1) * (1 / grid_spacing)
    
    # Check convergence
    if np.linalg.norm(rho_new - rho) < tolerance:
        print("Convergence reached after", iteration + 1, "iterations.")
        break
    
    # Update variables for next iteration
    orbitals = orbitals_new
    rho = rho_new
    
    # Visualize electron density and potential
    axs[0].cla()
    axs[0].plot(grid, rho, color='blue')
    axs[0].set_xlabel('Position')
    axs[0].set_ylabel('Electron Density')
    axs[0].set_title('Electron Density (Iteration {})'.format(iteration+1))
    
    axs[1].cla()
    axs[1].plot(grid, V_eff, color='red')
    axs[1].set_xlabel('Position')
    axs[1].set_ylabel('Effective Potential')
    axs[1].set_title('Effective Potential (Iteration {})'.format(iteration+1))
    
    plt.pause(0.1)

# Show the final electron density and potential
plt.show()
