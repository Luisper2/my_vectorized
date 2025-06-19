import os

import jax.numpy as np
import jax.random as jr
import tqdm

os.system('cls' if os.name == 'nt' else 'clear')

class LBM():
    def __init__(self):
        self.nx = 400 # Width
        self.ny = 100 # Height

        self.tau = 1  # Cinematic Viscosity (Collision)
        self.u = 0.1 # Initial Velocity

        self.nt = 4000 # Number of Iterations

        self.nv = 9 # Number of Velocities

        self.x, self.y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij') # Grid

        self.weight = np.zeros(self.nv)  # Weight
        self.bounce = np.zeros(self.nv, dtype=int) # Bounce

        self.F = np.ones((self.ny, self.nx, self.nv)) + 0.01 * jr.normal(jr.PRNGKey(0), (self.ny, self.nx, self.nv))

        self.D2()

    def D2(self):
        self.weight = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

        # self.bounce = np.array([9, 5, 4, 1, 2, ])

if __name__ == '__main__':
    simulation = LBM()