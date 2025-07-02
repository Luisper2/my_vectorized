import os
import sys
import jax
import pickle
from tqdm import tqdm
import jax.numpy as jnp
from functools import partial

class lbm():
    def __init__(self, nx: int = 100, ny: int = 100):
        self.nx = nx
        self.ny = ny
        
        self.tau = 1

        self.walls = []

        self.D2Q9()
        
    def D2Q9(self):
        self.x, self.y = jnp.meshgrid(jnp.arange(self.nx + 2) - 0.5, jnp.arange(self.ny + 2) - 0.5, indexing='ij')

        self.number_velocities = 9

        self.weight = jnp.array([[4/9] + 4*[1/9] + 4*[1/36]])
        self.ex = jnp.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
        self.ey = jnp.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
        self.bounce = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
        
        shape = (self.nx + 2, self.ny + 2)
        self.lattice = jnp.zeros(shape)
        self.u = jnp.zeros(shape)
        self.v = jnp.zeros(shape)
        self.v = jnp.zeros(shape)
        self.rho = jnp.ones(shape)
        
        shape = (self.number_velocities, self.nx + 2, self.ny + 2)
        self.feq = jnp.zeros(shape)
        self.f = jnp.zeros(shape)
        self.ferr = jnp.zeros(shape)
        self.ftemp = self.f.copy()

    def _prepare_bounceback_indices(self):
        directions = {
            'left':  [3, 6, 7],
            'right': [1, 5, 8],
            'up':    [2, 5, 6],
            'down':  [4, 7, 8],
        }

        index = {
            'down':  (slice(1, -2),  1),
            'up':    (slice(1, -2), -2),
            'left':  (1, slice(1, -2)),
            'right': (-2, slice(1, -2)),
        }

        self._bb_idx = {}
        
        for pos, ks in directions.items():
            ks_arr = jnp.array(ks, dtype=jnp.int32)
            kbs   = self.bounce[ks_arr]

            i_idx, j_idx = index[pos]
            i_vals = (jnp.arange(i_idx.start, i_idx.stop) if isinstance(i_idx, slice) else jnp.array([i_idx]))
            j_vals = (jnp.arange(j_idx.start, j_idx.stop) if isinstance(j_idx, slice) else jnp.array([j_idx]))

            I, J = jnp.meshgrid(i_vals, j_vals, indexing='ij')
            
            shape = (ks_arr.shape[0],) + I.shape

            K  = jnp.broadcast_to(ks_arr[:, None, None], shape)
            KB = jnp.broadcast_to(kbs[:,    None, None], shape)
            I3 = jnp.broadcast_to(I[None, ...], shape)
            J3 = jnp.broadcast_to(J[None, ...],  shape)

            self._bb_idx[pos] = (K, KB, I3, J3)

    def define_tau(self, reynolds_number: float, velocity: float, lenght: float):
        self.tau = velocity * lenght / reynolds_number

    def wall(self, position: str, fixed: bool = True):
        direction = {
            'top': (-1, slice(None)),
            'left': (slice(None), 0),
            'right': (slice(None), -1),
            'bottom': (0, slice(None)),
        }

        side = direction.get(position)

        if side is None:
            print(f'Error setting wall at {position}')
            sys.exit()

        self.lattice = self.lattice.at[side].set(1)

        if fixed:
            self.walls.append(position)

    def update_equilibrium_distribution(self):
        usq = self.u ** 2 + self.v ** 2

        ue = self.u[:, :, None] * self.ex[None, None, :]
        ve = self.v[:, :, None] * self.ey[None, None, :]
        
        feq = (self.rho[:, :, None] * self.weight[None, None, :]) * (1.0 + 3.0 * (ue + ve) + 4.5 * (ue + ve) ** 2 - 1.5 * usq[:, :, None])

        self.feq = feq.squeeze(0).transpose(2,0,1)

    def compute_collisions(self):
        self.f = self.f - (self.f - self.feq) / self.tau

    def compute_streaming(self):
        self.ftemp = self.f.copy()
        
        for k in range(self.number_velocities):
            i, j = jnp.meshgrid(jnp.arange(1, self.nx + 1), jnp.arange(1, self.ny + 1))
        
            ii = i + self.ex[k]
            jj = j + self.ey[k]
        
            self.f = self.f.at[k, ii, jj].set(self.ftemp[k, i, j])

    def compute_bounceback_bc(self, position: str):
        K, KB, I3, J3 = self._bb_idx[position]
        
        vals = self.ftemp[K, I3, J3]
        
        self.f = self.f.at[KB, I3, J3].set(vals)

    def compute_neumann_bc(self):
        # this is applied at the top surface
        j = self.ny
        k_inx = [2,5,6]
        for i in range(1, self.nx + 1):
                utop = self.u0 / 2. * (1. + jnp.sin(2. * jnp.pi / self.nx * (i - 0.5) - jnp.pi / 2.))
                # utop = self.u0
                self.f[self.bounce[k_inx], i, j] = self.ftemp[k_inx, i, j] - 6 * self.weight[k_inx] * self.rho[i, j] * self.ex[
                    k_inx] * utop

    def compute_distribution_error(self):
        self.ferr = jnp.abs(self.f - self.ftemp)

    def compute_macroscopic_variables(self):
        # Initialize density, u, v arrays to zeros
        self.rho[:] = jnp.sum(self.f, axis=0)

        # Avoid division by zero by checking for small densities
        non_zero_density = self.rho > 1e-8

        # Compute velocities only for non-zero density cells
        self.u[:] = jnp.sum(self.ex[:, None, None] * self.f, axis=0)
        self.v[:] = jnp.sum(self.ey[:, None, None] * self.f, axis=0)

        # Normalize velocities by density
        self.u /= jnp.where(non_zero_density, self.rho, 1)  # Avoid division by zero
        self.v /= jnp.where(non_zero_density, self.rho, 1)  # Avoid division by zero

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
        self.rho[indices_lhs] = 2 * self.rho[indices_rhs1] - self.rho[indices_rhs2]

    def write_output(self, iteration):
        # Generate filename with leading zeros (7-digit format)
        filename = f"{iteration:07d}.pkl"
        # Save data
        with open(filename, "wb") as f:
            pickle.dump((self.x, self.y, self.u, self.v, self.rho), f)

    def write_output_dat(self, iteration):
        # Check if ist is a multiple of 1000
        if iteration % 100 == 0:
            tchar = f"{iteration // 100:05d}"  # Format step number as a five-digit string
            filename = f"py{tchar}.dat"  # Construct filename

            # Stack all data efficiently into a single 2D array
            data = jnp.column_stack(
                (self.x.ravel(), self.y.ravel(), self.u.ravel(), self.v.ravel(), self.rho.ravel()))

            # Write header and data in bulk
            header = 'variables = "x" "y" "u" "v" "density"\n'
            header += f'zone f=point i={self.nx + 2} j={self.ny + 2}\n'

            jnp.savetxt(filename, data, fmt="%.6f", delimiter=" ", header=header, comments='')

    def write_output_f(self, iteration):
        if iteration % 100 == 0:
            tchar = f"{iteration // 100:05d}"  # Format step number as a five-digit string
            filename = f"f-{tchar}.dat"  # Construct filename

            # Stack all data efficiently into a single 2D array
            data = jnp.column_stack((self.x[1:-1, 1:-1].ravel(), self.y[1:-1, 1:-1].ravel(),
                                    self.f[0, 1:-1, 1:-1].ravel(), self.f[1, 1:-1, 1:-1].ravel(),
                                    self.f[2, 1:-1, 1:-1].ravel(), self.f[3, 1:-1, 1:-1].ravel(),
                                    self.f[4, 1:-1, 1:-1].ravel(),
                                    self.f[5, 1:-1, 1:-1].ravel(), self.f[6, 1:-1, 1:-1].ravel(),
                                    self.f[7, 1:-1, 1:-1].ravel(),
                                    self.f[8, 1:-1, 1:-1].ravel()))

            # Write header and data in bulk
            header = 'variables = "x" "y" "f0" "f1" "f2" "f3" "f4" "f5" "f6" "f7" "f8"\n'
            header += f'zone f=point i={self.nx} j={self.ny}\n'

            jnp.savetxt(filename, data, fmt="%.6f", delimiter=" ", header=header, comments='')

    def write_output_f_eq(self, iteration):
        if iteration % 100 == 0:
            tchar = f"{iteration // 100:05d}"  # Format step number as a five-digit string
            filename = f"feq-{tchar}.dat"  # Construct filename

            # Stack all data efficiently into a single 2D array
            data = jnp.column_stack((self.x[1:-1, 1:-1].ravel(), self.y[1:-1, 1:-1].ravel(),
                                    self.feq[0, 1:-1, 1:-1].ravel(), self.feq[1, 1:-1, 1:-1].ravel(),
                                    self.feq[2, 1:-1, 1:-1].ravel(), self.feq[3, 1:-1, 1:-1].ravel(),
                                    self.feq[4, 1:-1, 1:-1].ravel(),
                                    self.feq[5, 1:-1, 1:-1].ravel(), self.feq[6, 1:-1, 1:-1].ravel(),
                                    self.feq[7, 1:-1, 1:-1].ravel(),
                                    self.feq[8, 1:-1, 1:-1].ravel()))

            # Write header and data in bulk
            header = 'variables = "x" "y" "f0" "f1" "f2" "f3" "f4" "f5" "f6" "f7" "f8"\n'
            header += f'zone f=point i={self.nx} j={self.ny}\n'

            jnp.savetxt(filename, data, fmt="%.6f", delimiter=" ", header=header, comments='')

    def run(self, steps: int = 10000, save: int = 100, dir: str = './data'):
        os.system('cls')
        os.makedirs(dir, exist_ok = True)

        for iter in tqdm(range(steps)):
            simulation.update_equilibrium_distribution()
            simulation.compute_collisions()
            simulation.compute_streaming()
            simulation.compute_bounceback_bc('down')
            simulation.compute_bounceback_bc('left')
            simulation.compute_bounceback_bc('right')
            simulation.compute_neumann_bc()
            simulation.compute_distribution_error()
            simulation.compute_macroscopic_variables()

if __name__ == '__main__':
    simulation = lbm(100, 100)
    simulation.define_tau(80, 0.04, 50 // 9)
    simulation.wall('top', False)
    simulation.wall('left')
    simulation.wall('right')
    simulation.wall('bottom')
    simulation.run(1, 10, './data4')