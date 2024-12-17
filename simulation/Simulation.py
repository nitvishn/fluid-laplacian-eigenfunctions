import numpy as np
import torch

class Simulation():
    def __init__(self, w_init, id, n, viscosity, device='cpu'):
        self.id = id
        self.n = n 
        self.N = n ** 2
        self.device = device

        self.viscosity = viscosity

        self.w = w_init
        self.t = 0

        # Compute eigenvalues
        self.eigenvalues = torch.tensor([- (k_1 **2 + k_2 **2) for k_1 in range(n) for k_2 in range(n)], device=self.device)

        self.C = torch.zeros((self.N, self.N, self.N), device=self.device) # Structure coefficients
        self.set_coefficients()

        self.X = torch.linspace(0, np.pi, steps=200, device=self.device)
        self.Y = torch.linspace(0, np.pi, steps=200, device=self.device)
        X, Y = torch.meshgrid(self.X, self.Y, indexing='ij')
        self.domain = torch.stack([X, Y], dim=-1)

    def get_wave_numbers(self, k):
        if k == 0:
            raise ValueError("k must be greater than 0")
        n1 = k % self.n
        n2 = k // self.n
        return n1, n2
    
    def estimate_max_velocity(self):
        velocity_field = self.sample_velocity_field(self.domain.view(-1, 2))
        return torch.max(torch.norm(velocity_field, dim=-1))

    def set_coefficients(self):
        # Fill in the structure coefficients
        print(f"Setting structure coefficients...")
        self.C = self.C.to('cpu')
        for i in range(1, self.N):
            for j in range(1, self.N):
                i_1, i_2 = i // self.n, i % self.n
                j_1, j_2 = j // self.n, j % self.n

                k_11, k_12 = i_1 + j_1, i_2 + j_2
                k_1 = k_11 * self.n + k_12
                if 0 <= k_1 < self.N:
                    self.C[i, j, k_1] = - (i_1 * j_2 - i_2 * j_1) / (4 * (i_1 ** 2 + i_2 ** 2))

                k_21, k_22 = i_1 + j_1, i_2 - j_2
                k_2 = k_21 * self.n + k_22
                if 0 <= k_2 < self.N:
                    self.C[i, j, k_2] = (i_1 * j_2 + i_2 * j_1) / (4 * (i_1 ** 2 + i_2 ** 2))

                k_31, k_32 = i_1 - j_1, i_2 + j_2
                k_3 = k_31 * self.n + k_32
                if 0 <= k_3 < self.N:
                    self.C[i, j, k_3] = - (i_1 * j_2 + i_2 * j_1) / (4 * (i_1 ** 2 + i_2 ** 2))

                k_41, k_42 = i_1 - j_1, i_2 - j_2
                k_4 = k_41 * self.n + k_42
                if 0 <= k_4 < self.N:
                    self.C[i, j, k_4] = (i_1 * j_2 - i_2 * j_1) / (4 * (i_1 ** 2 + i_2 ** 2)) # In the paper, there is a minus sign in the denominator, but I think it is a typo
        self.C = self.C.to(self.device)

    def forward(self, dt: float):
        e1 = torch.sum(self.w ** 2) # Kinetic energy, computed using Parseval's equality!

        w_dot = torch.einsum('ijk,i,j->k', self.C, self.w, self.w)
        
        self.w += dt * w_dot

        e2 = torch.sum(self.w ** 2) # New kinetic energy 

        self.w *= torch.sqrt(e1 / e2)
        
        # Viscosity
        self.w *= torch.exp(self.viscosity * self.eigenvalues * dt)

        self.t += dt


    def sample_velocity_field(self, particle_locations: torch.Tensor):
        x = particle_locations[:, 0]
        y = particle_locations[:, 1]

        u = torch.zeros_like(x).to(self.device)
        v = torch.zeros_like(y).to(self.device)

        for k in range(1, self.N):
            k1, k2 = self.get_wave_numbers(k)
            lambda_k = np.sqrt(k1 ** 2 + k2 ** 2)
            u += self.w[k] * k2 * torch.sin(k1 * x) * torch.cos(k2 * y) / lambda_k
            v += - self.w[k] * k1 * torch.cos(k1 * x) * torch.sin(k2 * y) / lambda_k
        
        return torch.stack([u, v], dim=-1)
    

    def sample_vorticity_field(self, particle_locations: torch.Tensor):
        x = particle_locations[:, 0]
        y = particle_locations[:, 1]

        omega = torch.zeros_like(x).to(self.device)

        for k in range(1, self.N):
            k1, k2 = self.get_wave_numbers(k)
            omega += self.w[k] * torch.sin(k1 * x) * torch.sin(k2 * y)
        
        return omega