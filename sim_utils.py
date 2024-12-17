
from simulation.Simulation import Simulation
from simulation.SparseSimulation import SparseSimulation
from visualizer.ImageParticleVisualizer import ImageParticleVisualizer 
from visualizer.VorticityVisualizer import VorticityVisualizer
import numpy as np 
import torch


def rescale_timestep(dt, steps_per_frame, dx, max_velocity):
    CFL_NUMBER = 1
    max_dt = dx/(CFL_NUMBER * max_velocity)
    while (dt < 0.4 * max_dt) and steps_per_frame > 1:
        dt = dt * 2
        steps_per_frame = steps_per_frame // 2
    while (dt > max_dt):
        dt = dt / 2
        steps_per_frame *= 2
    
    return dt, steps_per_frame


def run_simulation(sim, visualizers=[], duration=5, framerate=30):
    # Determine the timestep

    # Sample the velocity at all grid points
    max_vel = sim.estimate_max_velocity()
    dx = np.pi / sim.resolution
    dt, steps_per_frame = 1 / framerate, 1
    dt, steps_per_frame = rescale_timestep(dt, steps_per_frame, dx, max_vel)

    # Simulation loop
    for i in range(int(duration * framerate)):
        for j in range(steps_per_frame):
            print(f"simulating frame={i}/{duration * framerate}, step={j}/{steps_per_frame}, id={sim.id}")

            sim.forward(dt)
            for vis in visualizers:
                vis.update(dt)
            
        for vis in visualizers:
            vis.visualize_frame()
        
        max_vel = sim.estimate_max_velocity()
        # Rescale dt if needed to speed up simulation, or to ensure stability
        # dt, steps_per_frame = rescale_timestep(dt, steps_per_frame, dx, max_vel)

    # Render
    for vis in visualizers:
        vis.render()


def generate_grid(resolution):
    """
    Constructs and returns the grid coordinates X and Y.

    Parameters:
    - resolution (int): The number of points along each axis in the grid.

    Returns:
    - X (torch.Tensor): Flattened tensor of X-coordinates, shape (resolution^2,).
    - Y (torch.Tensor): Flattened tensor of Y-coordinates, shape (resolution^2,).
    """
    # Create a meshgrid with values ranging from 0 to pi
    X, Y = torch.meshgrid(
        torch.linspace(0, torch.pi, resolution),
        torch.linspace(0, torch.pi, resolution),
        indexing='ij'
    )
    
    # Center the grid by shifting each point
    shift = torch.pi / (2 * resolution)
    X = X + shift
    Y = Y + shift
    
    # Flatten the grid for easier computations
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    
    return X, Y

def project_onto_eigenfunctions(u_init, domain, n):
    """
    Computes the w_init coefficients based on the Laplacian eigenfunction series.
    Finds the coefficients via orthogonal projection of the initial condition u_init.

    Parameters:
    - X (torch.Tensor): Flattened tensor of X-coordinates, shape (resolution^2,).
    - Y (torch.Tensor): Flattened tensor of Y-coordinates, shape (resolution^2,).
    - n (int): The maximum index for eigenfunctions in each dimension.

    Returns:
    - w_init (torch.Tensor): A tensor of shape (n^2,) containing the coefficients.
    """
    X, Y = domain
    resolution_squared = X.shape[0]
    resolution = int(resolution_squared ** 0.5)
    
    def find_coefficient(u_init, m, n):
        # Find the (m, n)th coefficient of the Laplacian eigenfunction series of u_init
        # u_init: resolution x resolution x 2 tensor
        # The (m, n)th eigenfunction is (1/lambda) * (m * torch.sin(m * x) * torch.cos(n * y), -m * torch.cos(m * x) * torch.sin(n * y))
        # where lambda = - (m^2 + n^2) 
        # We want to integrate this eigenfunction with u_init to find the coefficient

        # Compute the eigenfunction values
        lambda_val = - (m ** 2 + n ** 2)
        eigenfunction = torch.zeros(resolution * resolution, 2)
        eigenfunction[:, 0] = m * torch.sin(m * X) * torch.cos(n * Y)
        eigenfunction[:, 1] = -m * torch.cos(m * X) * torch.sin(n * Y)
        eigenfunction = eigenfunction / lambda_val

        # Find the dot product at each point
        dot_product = torch.sum(u_init * eigenfunction, dim=-1)

        # Integrate the dot product
        integral = torch.sum(dot_product) * (torch.pi / resolution) ** 2
        return integral
    
    # Initialize w_init tensor
    w_init = torch.zeros(n ** 2)
    
    # Compute coefficients for each (m, n) pair
    for i in range(n):
        for j in range(n):
            k = i * n + j
            print(f"Computing coefficient {k}/{n**2}")
            w_init[k] = find_coefficient(u_init, i, j)
    
    # Ensure the first coefficient is zero
    w_init[0] = 0.0
    
    return w_init