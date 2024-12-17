from sim_utils import run_simulation, generate_grid, project_onto_eigenfunctions
from visualizer.VorticityVisualizer import VorticityVisualizer
from visualizer.ImageParticleVisualizer import ImageParticleVisualizer
from simulation.SparseSimulation import SparseSimulation
import torch

# This runs in 14 seconds on my Macbook.
viscosity = 0.001
n = 7 # for 49 eigenfunctions
duration = 5 # seconds
framerate = 60
resolution = 128
sim_id = f'res_{resolution}_N_{n**2}'

num_particles = 2000 * (resolution // 128) ** 2 # For visualization  

# Setup the initial field that is moving right in the top half and left in the bottom half
X, Y = generate_grid(resolution)
u_init = torch.zeros((resolution**2, 2))
u_init[:, 0] = torch.tensor(Y > torch.pi / 2, dtype=torch.float32) - torch.tensor(Y < torch.pi / 2, dtype=torch.float32)

# Compute the eigenfunction coeffficients
w_init = project_onto_eigenfunctions(u_init, (X, Y), n)

# Simulation code 

sim = SparseSimulation(w_init, sim_id, n=n, viscosity=viscosity, resolution=resolution)

visualizers = [
    # VorticityVisualizer(sim, width=resolution, height=resolution, border_size=max(resolution//64, 1), framerate=framerate),
    ImageParticleVisualizer(sim, num_particles=num_particles, width=resolution, height=resolution, framerate=framerate, border_size=max(resolution//64, 1), gamma=0.5)
]

run_simulation(sim, visualizers, duration=duration, framerate=framerate)