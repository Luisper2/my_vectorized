import pickle

import numpy as np
from tqdm import tqdm


class LBM():
    def __init__(self, nx, ny, tau=1, u0=0.1):
        self.u0 = u0
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.x, self.y = np.meshgrid(np.arange(nx + 2) - 0.5, np.arange(ny + 2) - 0.5, indexing='ij')
        self.number_velocities = 9
        self.wt = np.zeros(self.number_velocities)
        self.ex = np.zeros(self.number_velocities, dtype=int)
        self.ey = np.zeros(self.number_velocities, dtype=int)
        self.bounce_back = np.zeros(self.number_velocities, dtype=int)
        self.lattice = np.zeros((nx + 2, ny + 2))  # one ghost node per side
        self.u = np.zeros((nx + 2, ny + 2))
        self.v = np.zeros((nx + 2, ny + 2))
        self.v = np.zeros((nx + 2, ny + 2))
        self.density = np.ones((nx + 2, ny + 2))
        self.feq = np.zeros((self.number_velocities, nx + 2, ny + 2))
        self.f = np.zeros((self.number_velocities, nx + 2, ny + 2))
        self.ferr = np.zeros((self.number_velocities, nx + 2, ny + 2))
        self.ftemp = self.f.copy()

        self.__post_init__()

    def __post_init__(self):
        self.wt[0] = 4 / 9
        self.wt[1:5] = 1 / 9
        self.wt[5:] = 1 / 36

        self.ex[[0, 2, 4]] = 0
        self.ex[[1, 5, 8]] = 1
        self.ex[[3, 6, 7]] = -1
        self.ey[[0, 1, 3]] = 0
        self.ey[[2, 5, 6]] = 1
        self.ey[[4, 7, 8]] = -1

        self.bounce_back[0] = 0
        self.bounce_back[1] = 3
        self.bounce_back[2] = 4
        self.bounce_back[3] = 1
        self.bounce_back[4] = 2
        self.bounce_back[5] = 7
        self.bounce_back[6] = 8
        self.bounce_back[7] = 5
        self.bounce_back[8] = 6

        # set bcs
        self.lattice[:, 0] = 1
        self.lattice[:, -1] = 1
        self.lattice[0, 0:-1] = 1
        self.lattice[-1, 0:-1] = 1

    import numpy as np

    def update_equilibrium_distribution(self):
        usq = self.u ** 2 + self.v ** 2  # Element-wise square of u and v
        # Pre-compute the velocity components in x and y directions

        # 102, 102

        ue = self.u[:, :, np.newaxis] * self.ex[np.newaxis, np.newaxis, :]  # Broadcasting for u * ex
        ve = self.v[:, :, np.newaxis] * self.ey[np.newaxis, np.newaxis, :]  # Broadcasting for v * ey

        # 102, 102, 9
        # 102, 102, 9

        # Calculate equilibrium distribution function using vectorized operations
        feq = (self.density[:, :, np.newaxis] * self.wt[np.newaxis, np.newaxis, :]) * (
                1.0 + 3.0 * (ue + ve) + 4.5 * (ue + ve) ** 2 - 1.5 * usq[:, :, np.newaxis]
        )
        
        # 102, 102, 9

        # Assign the result to self.feq
        self.feq = feq.transpose(2,0,1)

        # 9, 102, 102

    def compute_collisions(self):
        self.f = self.f - (self.f - self.feq) / self.tau

    def compute_streaming(self):
        self.ftemp = self.f.copy()
        for k in range(self.number_velocities):
            # Create 2D meshgrid of indices
            i, j = np.meshgrid(np.arange(1, self.nx + 1), np.arange(1, self.ny + 1))
            # Compute shifted indices
            ii = i + self.ex[k]
            jj = j + self.ey[k]
            # Perform the assignment
            self.f[k, ii, jj] = self.ftemp[k, i, j]

    def compute_bounceback_bc(self, direction: str):
        dict_directions_bounceback = {
            'left': [3, 6, 7],
            'right': [1, 5, 8],
            'up': [2, 5, 6],
            'down': [4, 7, 8],
        }
        dict_indices_interior_points = {
            'down': (slice(1, -2), 1),
            'up': (slice(1, -2), -2),
            'left': (1, slice(1, -2)),
            'right': (-2, slice(1, -2)),
        }

        dir_bounceback = dict_directions_bounceback[direction]
        indx_ij = dict_indices_interior_points[direction]
        indexing_pre = [dir_bounceback, *indx_ij]
        indexing_post = [self.bounce_back[dir_bounceback], *indx_ij]
        self.f[*indexing_post] = self.ftemp[*indexing_pre]

    def compute_neumann_bc(self):
        # this is applied at the top surface
        j = self.ny
        k_inx = [2,5,6]
        for i in range(1, self.nx + 1):
                utop = self.u0 / 2. * (1. + np.sin(2. * np.pi / self.nx * (i - 0.5) - np.pi / 2.))
                # utop = self.u0
                self.f[self.bounce_back[k_inx], i, j] = self.ftemp[k_inx, i, j] - 6 * self.wt[k_inx] * self.density[i, j] * self.ex[
                    k_inx] * utop

    def compute_distribution_error(self):
        self.ferr = np.abs(self.f - self.ftemp)

    def compute_macroscopic_variables(self):
        # Initialize density, u, v arrays to zeros
        self.density[:] = np.sum(self.f, axis=0)

        # Avoid division by zero by checking for small densities
        non_zero_density = self.density > 1e-8

        # Compute velocities only for non-zero density cells
        self.u[:] = np.sum(self.ex[:, None, None] * self.f, axis=0)
        self.v[:] = np.sum(self.ey[:, None, None] * self.f, axis=0)

        # Normalize velocities by density
        self.u /= np.where(non_zero_density, self.density, 1)  # Avoid division by zero
        self.v /= np.where(non_zero_density, self.density, 1)  # Avoid division by zero

        # Set velocities to zero where density is zero
        self.u[~non_zero_density] = 0
        self.v[~non_zero_density] = 0

        # BCs treatment
        # left
        indices_lhs = (0, slice(1, self.ny))
        indices_rhs1 = (1, slice(1, self.ny))
        indices_rhs2 = (2, slice(1, self.ny))
        self.compute_macroscopic_variables_boundaries(indices_lhs, indices_rhs1, indices_rhs2)
        # right
        indices_lhs = (-1, slice(1, self.ny))  # nx+1
        indices_rhs1 = (-2, slice(1, self.ny))  # nx
        indices_rhs2 = (-3, slice(1, self.ny))  # nx-1
        self.compute_macroscopic_variables_boundaries(indices_lhs, indices_rhs1, indices_rhs2)

        # bottom
        indices_lhs = (slice(None), 0)
        indices_rhs1 = (slice(None), 1)
        indices_rhs2 = (slice(None), 2)
        self.compute_macroscopic_variables_boundaries(indices_lhs, indices_rhs1, indices_rhs2)
        # top
        indices_lhs = (slice(None), -1)
        indices_rhs1 = (slice(None), -2)
        indices_rhs2 = (slice(None), -3)
        self.compute_macroscopic_variables_boundaries(indices_lhs, indices_rhs1, indices_rhs2)

    def compute_macroscopic_variables_boundaries(self, indices_lhs, indices_rhs1, indices_rhs2):
        self.u[indices_lhs] = 2 * self.u[indices_rhs1] - self.u[indices_rhs2]
        self.v[indices_lhs] = 2 * self.v[indices_rhs1] - self.v[indices_rhs2]
        self.density[indices_lhs] = 2 * self.density[indices_rhs1] - self.density[indices_rhs2]

    def write_output(self, iteration):
        # Generate filename with leading zeros (7-digit format)
        filename = f"{iteration:07d}.pkl"
        # Save data
        with open(filename, "wb") as f:
            pickle.dump((self.x, self.y, self.u, self.v, self.density), f)

    def write_output_dat(self, iteration):
        # Check if ist is a multiple of 1000
        if iteration % 100 == 0:
            tchar = f"{iteration // 100:05d}"  # Format step number as a five-digit string
            filename = f"py{tchar}.dat"  # Construct filename

            # Stack all data efficiently into a single 2D array
            data = np.column_stack(
                (self.x.ravel(), self.y.ravel(), self.u.ravel(), self.v.ravel(), self.density.ravel()))

            # Write header and data in bulk
            header = 'variables = "x" "y" "u" "v" "density"\n'
            header += f'zone f=point i={self.nx + 2} j={self.ny + 2}\n'

            np.savetxt(filename, data, fmt="%.6f", delimiter=" ", header=header, comments='')

    def write_output_f(self, iteration):
        if iteration % 100 == 0:
            tchar = f"{iteration // 100:05d}"  # Format step number as a five-digit string
            filename = f"f-{tchar}.dat"  # Construct filename

            # Stack all data efficiently into a single 2D array
            data = np.column_stack((self.x[1:-1, 1:-1].ravel(), self.y[1:-1, 1:-1].ravel(),
                                    self.f[0, 1:-1, 1:-1].ravel(), self.f[1, 1:-1, 1:-1].ravel(),
                                    self.f[2, 1:-1, 1:-1].ravel(), self.f[3, 1:-1, 1:-1].ravel(),
                                    self.f[4, 1:-1, 1:-1].ravel(),
                                    self.f[5, 1:-1, 1:-1].ravel(), self.f[6, 1:-1, 1:-1].ravel(),
                                    self.f[7, 1:-1, 1:-1].ravel(),
                                    self.f[8, 1:-1, 1:-1].ravel()))

            # Write header and data in bulk
            header = 'variables = "x" "y" "f0" "f1" "f2" "f3" "f4" "f5" "f6" "f7" "f8"\n'
            header += f'zone f=point i={self.nx} j={self.ny}\n'

            np.savetxt(filename, data, fmt="%.6f", delimiter=" ", header=header, comments='')

    def write_output_f_eq(self, iteration):
        if iteration % 100 == 0:
            tchar = f"{iteration // 100:05d}"  # Format step number as a five-digit string
            filename = f"feq-{tchar}.dat"  # Construct filename

            # Stack all data efficiently into a single 2D array
            data = np.column_stack((self.x[1:-1, 1:-1].ravel(), self.y[1:-1, 1:-1].ravel(),
                                    self.feq[0, 1:-1, 1:-1].ravel(), self.feq[1, 1:-1, 1:-1].ravel(),
                                    self.feq[2, 1:-1, 1:-1].ravel(), self.feq[3, 1:-1, 1:-1].ravel(),
                                    self.feq[4, 1:-1, 1:-1].ravel(),
                                    self.feq[5, 1:-1, 1:-1].ravel(), self.feq[6, 1:-1, 1:-1].ravel(),
                                    self.feq[7, 1:-1, 1:-1].ravel(),
                                    self.feq[8, 1:-1, 1:-1].ravel()))

            # Write header and data in bulk
            header = 'variables = "x" "y" "f0" "f1" "f2" "f3" "f4" "f5" "f6" "f7" "f8"\n'
            header += f'zone f=point i={self.nx} j={self.ny}\n'

            np.savetxt(filename, data, fmt="%.6f", delimiter=" ", header=header, comments='')

    def run_num_steps(self, num_steps, save_step):
        for iter in tqdm(range(num_steps)):
            lbm.update_equilibrium_distribution()
            lbm.compute_collisions()
            lbm.compute_streaming()
            lbm.compute_bounceback_bc('down')
            lbm.compute_bounceback_bc('left')
            lbm.compute_bounceback_bc('right')
            lbm.compute_neumann_bc()
            lbm.compute_distribution_error()
            lbm.compute_macroscopic_variables()
            if iter % save_step == 0:
                lbm.write_output_dat(iter)
                # lbm.write_output_f(iter)
                # lbm.write_output_f_eq(iter)
                lbm.write_output(iter)


if __name__ == '__main__':
    lbm = LBM(100, 100)
    lbm.run_num_steps(1, 10)