import os
import jax
import pickle
import jax.numpy as np
import jax.random as jr
from tqdm import tqdm

os.system('cls' if os.name == 'nt' else 'clear')

class LBM():
    def __init__(self, nx, ny, tau = 1.225, u0 = 0, v0 = 0, perturbations = False):
        self.nx = nx # Width
        self.ny = ny # Height

        self.tau = tau  # Kinematic Viscosity (Collision)
        
        self.u0 = u0 # Initial X Velocity
        self.v0 = v0 # Initial Y Velocity

        self.n_velocities = 9 # Number of Velocities

        self.x, self.y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij') # Grid

        self.weight = np.zeros(self.n_velocities)  # Weight
        # self.wt

        self.lattice = np.zeros((self.nx, self.ny)) # Lattice
        
        self.lattice_u = np.zeros(self.n_velocities, dtype=int) # Lattice X Velocities
        self.lattice_v = np.zeros(self.n_velocities, dtype=int) # Lattice Y Velocities
        # self.ex, self.ey

        self.u = self.lattice.copy() # X Velocities
        self.v = self.lattice.copy() # Y Velocities

        self.bounce = np.zeros(self.n_velocities, dtype=int) # Bounce
        # self.bounce_back

        self.density = np.ones((self.nx, self.ny))

        if perturbations:
            self.F = np.ones((self.ny, self.nx, self.n_velocities)) + 0.01 * jr.normal(jr.PRNGKey(0), (self.ny, self.nx, self.n_velocities))
        else:
            self.F = np.zeros((self.n_velocities, self.nx, self.ny))

        self.Feq = np.zeros((self.n_velocities, self.nx, self.ny))
        self.Ferr = self.Feq.copy()

        self.cache = self.Feq.copy()

        self.boundary = np.zeros((self.nx, self.ny))

        self.__init_D2Q9__()

    def __init_D2Q9__(self):
        # 6   2   5
        #   \ | /
        # 3 - 0 - 1
        #   / | \
        # 7   4   8

        self.weight.at[0].set(4 / 9)
        self.weight.at[1:5].set(1 / 9)
        self.weight.at[5:].set(1 / 36)

        self.lattice_u = (
            self.lattice_u
                .at[np.array([0,2,4])].set(0)
                .at[np.array([1,5,8])].set(1)
                .at[np.array([3,6,7])].set(-1)
        )

        self.lattice_v = (
            self.lattice_v
                .at[np.array([0,1,3])].set(0)
                .at[np.array([2,5,6])].set(1)
                .at[np.array([4,7,8])].set(-1)
        )

        self.bounce = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

        self.lattice = (
            self.lattice
            .at[:,  0].set(True)   # Left
            .at[:, -1].set(True)   # Right
            .at[0,  :].set(True)   # Down
            .at[-1, :].set(True)   # Up
        )

    def update_equilibrium_distribution(self):
        unique = self.u ** 2 + self.v ** 2

        u_e = self.u[:, :, np.newaxis] * self.lattice_u[np.newaxis, np.newaxis, :]
        v_e = self.v[:, :, np.newaxis] * self.lattice_v[np.newaxis, np.newaxis, :]

        feq = (self.density[:, :, np.newaxis] * self.weight[np.newaxis, np.newaxis, :]) * (
                1.0 + 3.0 * (u_e + v_e) + 4.5 * (u_e + v_e) ** 2 - 1.5 * unique[:, :, np.newaxis]
        )

        self.feq = feq.transpose(2,0,1)

    def compute_collisions(self):
        self.F = self.F - (self.F - self.feq) / self.tau

    def compute_streaming(self):
        def shift_channel(fk, shift):
            return np.roll(np.roll(fk, shift[0], axis=0), shift[1], axis=1)

        shifts = np.stack([self.lattice_u, self.lattice_v], axis=1)

        self.F = jax.vmap(shift_channel, in_axes=(0, 0))(self.F, shifts)

    def compute_bounceback_bc(self, direction: str):
        dict_pre = {
            'left':  np.array([3, 6, 7]),
            'right': np.array([1, 5, 8]),
            'up':    np.array([2, 5, 6]),
            'down':  np.array([4, 7, 8]),
        }

        pre_ks = dict_pre[direction]
        post_ks = np.array(self.bounce)[pre_ks]

        if direction in ('down', 'up'):
            i_idx = np.arange(1, self.nx + 1)
            j_idx = 1 if direction == 'down' else self.ny
        
            ks = pre_ks[:, None]
            xs = i_idx[None, :]
            ys = j_idx
        else:
            i_idx = 1 if direction == 'left' else self.nx
            j_idx = np.arange(1, self.ny + 1)
            
            ks = pre_ks[:, None]
            xs = i_idx
            ys = j_idx[None, :]

        vals = self.cache[ks, xs, ys]
        
        self.F = self.F.at[(post_ks[:, None], xs, ys)].set(vals)

    def compute_neumann_bc(self):
        k_inx = np.array([2, 5, 6])

        i_idx = np.arange(1, self.nx + 1)
        j = self.ny

        utop = (self.u0 / 2.0 * (1.0 + np.sin(2.0 * np.pi / self.nx * (i_idx - 0.5) - np.pi / 2.0)))
        vtop = (self.v0 / 2.0 * (1.0 + np.cos(2.0 * np.pi / self.nx * (i_idx - 0.5))))

        ftemp_sel = self.cache[k_inx[:, None], i_idx[None, :], j]
        rho_sel = self.density[i_idx, j]
        ex_sel = self.lattice_u[k_inx][:, None]
        ey_sel = self.lattice_v[k_inx][:, None]
        wt_sel = self.weight[k_inx][:, None]

        correction = 6.0 * wt_sel * rho_sel * (ex_sel * utop + ey_sel * vtop)
        delta = ftemp_sel - correction

        self.F = self.F.at[k_inx[:, None], i_idx[None, :], j].set(delta)

    def compute_distribution_error(self):
        self.Ferr = np.abs(self.F - self.cache)

    def compute_macroscopic_variables(self):
        rho = np.sum(self.F, axis=0)
        u_raw = np.tensordot(self.lattice_u, self.F, axes=(0, 0))
        v_raw = np.tensordot(self.lattice_v, self.F, axes=(0, 0))

        mask = rho > 1e-8
        
        u = np.where(mask, u_raw / rho, 0.0)
        v = np.where(mask, v_raw / rho, 0.0)

        def neumann_extrap(arr):
            arr = arr.at[0, :].set(2*arr[1, :] - arr[2, :])
            arr = arr.at[-1, :].set(2*arr[-2, :] - arr[-3, :])
            arr = arr.at[:, 0].set(2*arr[:, 1] - arr[:, 2])
            arr = arr.at[:, -1].set(2*arr[:, -2] - arr[:, -3])
        
            return arr

        self.density = neumann_extrap(rho)
        self.u = neumann_extrap(u)
        self.v = neumann_extrap(v)

    def save(self, iteration):
        filename = f'./data/{iteration:07d}.pkl'
        
        with open(filename, "wb") as f:
            pickle.dump((self.x, self.y, self.u, self.v, self.density), f)

    def run(self, steps = 1000, save = 10):
        jax.config.update("jax_enable_x64", True)

        os.makedirs(os.path.dirname('./data/'), exist_ok=True)
        
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

            if iter % save == 0:
                simulation.save(iter)

if __name__ == '__main__':
    simulation = LBM(400, 100, u0 = 1)
    simulation.run(10000, 10)

    cylinder_cx = 400 // 5
    cylinder_cy = 100 // 5
    cylinder_r = 100 // 9
    max_inflox_v = 0.04

    print('done')