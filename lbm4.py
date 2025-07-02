import os
import jax
import pickle
from tqdm import tqdm
import jax.numpy as jnp
from functools import partial

class lbm:
    def __init__(self, nx: int, ny: int, density: float):
        self.nx = nx
        self.ny = ny

        self.u0 = 0.04

        self.tau = (0.04 * (50 // 9)) / 80

        self.D2Q9(density)

    def D2Q9(self, density = 1.225):
        self.x, self.y = jnp.meshgrid(jnp.arange(self.nx), jnp.arange(self.ny), indexing = 'ij')

        self.number_velocities = 9

        self.ex = jnp.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
        self.ey = jnp.array([0, 0, 1, 0, -1, 1, 1, -1, -1])

        self.index = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.oppositive = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

        self.weight = jnp.array([4/9] + 4*[1/9] + 4*[1/36])

        shape = (self.nx, self.ny)
        self.rho = density * jnp.ones(shape)
        self.u = jnp.zeros(shape)
        self.v = jnp.zeros(shape)
        self.lattice = jnp.zeros(shape)
        self.f = jnp.ones((self.number_velocities, self.nx, self.ny))

        self.up_velocity = jnp.array([2, 5, 6])
        self.left_velocity = jnp.array([3, 6, 7])
        self.right_velocity = jnp.array([1, 5, 8])
        self.down_velocity = jnp.array([4, 7,8])
        self.horizontal_velocity = jnp.array([0, 2, 4])
        self.vertical_velocity = jnp.array([0, 1, 3])

    def step(self, f, rho, u, v):
        u = u.at[0, :].set(0.04)

        return f, rho, u, v

    def run(self, steps: int, save: int, folder: str):
        jax.config.update('jax_enable_x64', True)

        os.system('cls')
        os.makedirs(folder, exist_ok = True)

        self.u = self.u.at[0, :].set(0.04)

        f, rho, u, v = self.f, self.rho, self.u, self.v

        for i in tqdm(range(steps)):
            f, rho, u, v = self.step(f, rho, u, v)
            
            if i % save == 0:
                filename = f'{folder}/{i:07d}.dat'
                
                with open(filename, 'wb') as fp:
                    pickle.dump((jnp.array(self.x), jnp.array(self.y), jnp.array(u), jnp.array(v), jnp.array(rho)), fp)

        self.f, self.rho, self.u, self.v = f, rho, u, v

if __name__ == '__main__':
    simulation = lbm(300, 50, 1.225)
    simulation.run(4000, 10, './data3')