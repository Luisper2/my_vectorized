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

        self.vx = np.zeros(self.nv, dtype=int)
        self.vy = np.zeros(self.nv, dtype=int)

        self.F = np.zeros((self.nv, self.nx, self.ny))
        self.Feq = self.F.copy()
        self.Ferr = self.F.copy()

        self.boundary = np.zeros((self.nx, self.ny))
        
        # self.F = np.ones((self.ny, self.nx, self.nv)) + 0.01 * jr.normal(jr.PRNGKey(0), (self.ny, self.nx, self.nv))

        self.__init_D2Q9__()

    def __init_D2Q9__(self):
        self.weight[0] = 4 / 9
        self.weight[1:5] = 1 / 9
        self.weight[5:] = 1 / 36

        self.vx[[0, 2, 4]] = 0
        self.vx[[1, 5, 8]] = 1
        self.vx[[3, 6, 7]] = -1

        self.vy[[0, 1, 3]] = 0
        self.vy[[2, 5, 6]] = 1
        self.vy[[4, 7, 8]] = -1

        self.bounce = [0, 3, 4, 1, 2, 7, 8, 5, 6]

if __name__ == '__main__':
    simulation = LBM()