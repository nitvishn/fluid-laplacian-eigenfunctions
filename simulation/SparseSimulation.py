import numpy as np
import torch

class SparseSimulation():
    def __init__(self, w_init, id, n, viscosity, resolution, device='cpu'):
        self.id = id
        self.n = n 
        self.N = n ** 2
        self.device = device

        self.resolution = resolution

        self.viscosity = viscosity

        self.w = w_init
        self.t = 0

        # Compute eigenvalues
        self.eigenvalues = torch.tensor([- (k_1 **2 + k_2 **2) for k_1 in range(n) for k_2 in range(n)], device=self.device)

        # Instead of a dense NxNxN tensor, we will store a list of sparse NxN matrices for each k.
        # self.C[k] will be a sparse NxN matrix.
        self.C = [None] * self.N
        self.set_coefficients()

        self.X = torch.linspace(0, np.pi, steps=resolution, device=self.device)
        self.Y = torch.linspace(0, np.pi, steps=resolution, device=self.device)
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
        print(f"Setting structure coefficients...")
        # We'll accumulate entries for each k in a coordinate list before making them sparse.
        # For each k, we'll store (i_indices, j_indices, values).
        C_entries = [ [[], [], []] for _ in range(self.N) ]  # For each k: [[i_coords], [j_coords], [vals]]

        for i in range(1, self.N):
            for j in range(1, self.N):
                i_1, i_2 = i // self.n, i % self.n
                j_1, j_2 = j // self.n, j % self.n

                denom = 4 * (i_1 ** 2 + i_2 ** 2)
                if denom == 0:
                    # If i_1 = i_2 = 0 (shouldn't happen since i>=1), skip
                    continue

                # k_1
                k_11, k_12 = i_1 + j_1, i_2 + j_2
                k_1 = k_11 * self.n + k_12
                if 0 <= k_1 < self.N:
                    val = - (i_1 * j_2 - i_2 * j_1) / denom
                    C_entries[k_1][0].append(i)
                    C_entries[k_1][1].append(j)
                    C_entries[k_1][2].append(val)

                # k_2
                k_21, k_22 = i_1 + j_1, i_2 - j_2
                k_2 = k_21 * self.n + k_22
                if 0 <= k_2 < self.N:
                    val = (i_1 * j_2 + i_2 * j_1) / denom
                    C_entries[k_2][0].append(i)
                    C_entries[k_2][1].append(j)
                    C_entries[k_2][2].append(val)

                # k_3
                k_31, k_32 = i_1 - j_1, i_2 + j_2
                k_3 = k_31 * self.n + k_32
                if 0 <= k_3 < self.N:
                    val = - (i_1 * j_2 + i_2 * j_1) / denom
                    C_entries[k_3][0].append(i)
                    C_entries[k_3][1].append(j)
                    C_entries[k_3][2].append(val)

                # k_4
                k_41, k_42 = i_1 - j_1, i_2 - j_2
                k_4 = k_41 * self.n + k_42
                if 0 <= k_4 < self.N:
                    val = (i_1 * j_2 - i_2 * j_1) / denom
                    C_entries[k_4][0].append(i)
                    C_entries[k_4][1].append(j)
                    C_entries[k_4][2].append(val)

        # Now convert the accumulated entries into sparse matrices
        for k in range(self.N):
            if len(C_entries[k][0]) == 0:
                # If no entries, just store a zero sparse matrix
                indices = torch.empty((2,0), dtype=torch.long, device=self.device)
                values = torch.empty(0, device=self.device)
            else:
                i_coords = torch.tensor(C_entries[k][0], device=self.device, dtype=torch.long)
                j_coords = torch.tensor(C_entries[k][1], device=self.device, dtype=torch.long)
                vals = torch.tensor(C_entries[k][2], device=self.device, dtype=torch.float)
                indices = torch.stack([i_coords, j_coords], dim=0)
                values = vals

            self.C[k] = torch.sparse_coo_tensor(indices, values, (self.N, self.N), device=self.device).coalesce()

    def forward(self, dt: float):
        e1 = torch.sum(self.w ** 2)

        # Now we must compute w_dot = torch.einsum('ijk,i,j->k', self.C, self.w, self.w)
        # with sparse C. This is equivalent to w_dot(k) = w^T C_k w for each k.
        w_dot = torch.zeros(self.N, device=self.device)
        w_vec = self.w.unsqueeze(1)  # N x 1
        for k in range(self.N):
            # M_k = C[k]
            # M_k w = (N x N) * (N x 1) = (N x 1)
            if self.C[k]._nnz() > 0:
                M_k_w = torch.sparse.mm(self.C[k], w_vec)  # N x 1
                # w_dot[k] = w^T (M_k w)
                w_dot[k] = torch.dot(self.w, M_k_w.squeeze(1))
            # If nnz=0, w_dot[k] remains 0.

        self.w += dt * w_dot

        e2 = torch.sum(self.w ** 2)

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
            if lambda_k == 0:
                continue
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
    

    def get_domain_vorticity_field(self):
        return self.sample_vorticity_field(self.domain.view(-1, 2)).view(self.domain.shape[:-1])