import os

import jax.numpy as np
from tqdm import tqdm

os.system('cls' if os.name == 'nt' else 'clear')

class LBM():
    def __init__(self, nx, ny, tau = 1, reynolds = 80, perturbations = False):
        self.nx = nx # Width
        self.ny = ny # Height

        self.tau = tau  # Cinematic Viscosity (Collision)
        self.u0 = reynolds # Initial Velocity

        # self.nt = 4000 # Number of Iterations

        self.n_velocities = 9 # Number of Velocities

        self.x, self.y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij') # Grid

        self.weight = np.zeros(self.n_velocities)  # Weight

        self.lattive_u = np.zeros(self.n_velocities, dtype=int) # Lattive X Velocities
        self.lattive_v = np.zeros(self.n_velocities, dtype=int) # Lattive Y Velocities

        self.bounce = np.zeros(self.n_velocities, dtype=int) # Bounce

        if perturbations:
            self.F = np.ones((self.ny, self.nx, self.n_velocities)) + 0.01 * jr.normal(jr.PRNGKey(0), (self.ny, self.nx, self.n_velocities))
        else:
            self.F = np.zeros((self.n_velocities, self.nx, self.ny))

        self.Feq = np.zeros((self.n_velocities, self.nx, self.ny))
        self.Ferr = self.Feq.copy()

        self.boundary = np.zeros((self.nx, self.ny))

        self.__init_D2Q9__()

    def __init_D2Q9__(self):
        self.weight.at[0].set(4 / 9)
        self.weight.at[1:5].set(1 / 9)
        self.weight.at[5:].set(1 / 36)

        self.lattive_u = (
            self.lattive_u
                .at[np.array([0,2,4])].set(0)
                .at[np.array([1,5,8])].set(1)
                .at[np.array([3,6,7])].set(-1)
        )

        self.lattive_v = (
            self.lattive_v
                .at[np.array([0,1,3])].set(0)
                .at[np.array([2,5,6])].set(1)
                .at[np.array([4,7,8])].set(-1)
        )

        self.bounce = [0, 3, 4, 1, 2, 7, 8, 5, 6]

if __name__ == '__main__':
    simulation = LBM(400, 100)

    cylinder_cx = 400 // 5
    cylinder_cy = 100 // 5
    cylinder_r = 100 // 9
    max_inflox_v = 0.04

    print('done')