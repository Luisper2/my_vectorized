import os

import jax
import jax.numpy as np
import jax.random as jr
from tqdm import tqdm

os.system('cls' if os.name == 'nt' else 'clear')

class LBM():
    def __init__(self, nx, ny, tau = 1, u0 = 0.1, perturbations = False):
        self.nx = nx # Width
        self.ny = ny # Height

        self.tau = tau  # Kinematic Viscosity (Collision)
        self.u0 = u0 # Initial Velocity

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
        def shift_layer(f_k, du, dv):
            return np.roll(np.roll(f_k, du, axis=0), dv, axis=1)
        
        self.F = jax.vmap(shift_layer, in_axes=(0, 0, 0))(
            self.F,
            self.lattice_u,
            self.lattice_v
        )

    def run(self, steps = 1000, save = 10):
        for iter in tqdm(range(steps)):
            simulation.update_equilibrium_distribution()
            simulation.compute_collisions()
            simulation.compute_streaming()

if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)

    simulation = LBM(400, 100)
    simulation.run(10000, 10)

    cylinder_cx = 400 // 5
    cylinder_cy = 100 // 5
    cylinder_r = 100 // 9
    max_inflox_v = 0.04

    print('done')