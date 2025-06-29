import os
import jax
import pickle
from tqdm import tqdm
import jax.numpy as jnp
from functools import partial

class lbm:
    def __init__(self, nx = 100, ny = 100):
        self.nx = nx
        self.ny = ny
        self.u0 = 0.1

        self.tau = 1

        self.D2Q9()

    def D2Q9(self):
        self.dirs = 9

        self.x, self.y = jnp.meshgrid(jnp.arange(self.nx), jnp.arange(self.ny))
        
        self.weight = jnp.array([4/9] + [1/9]*4 + [1/36]*4)
        self.ex = jnp.array([0,1,0,-1,0,1,-1,-1,1], dtype=jnp.int32)
        self.ey = jnp.array([0,0,1,0,-1,1,1,-1,-1], dtype=jnp.int32)
        self.bounce_back = jnp.array([0,3,4,1,2,7,8,5,6], dtype=jnp.int32)

        shape= (self.nx, self.ny)

        self.u = jnp.zeros(shape)
        self.v = jnp.zeros(shape)
        self.rho = jnp.ones(shape)
        self.lattice = jnp.zeros(shape)
        self.f = jnp.zeros((self.nx, self.ny, self.dirs))

    def step(self, f, rho, u, v):
        
        
        return f, rho, u, v

    def run(self, num_steps = 1000, save_step = 1000, output_dir = './data'):
        os.makedirs(output_dir, exist_ok=True)

        f, rho, u, v = self.f, self.rho, self.u, self.v
        
        for iter in tqdm(range(num_steps)):
            f, rho, u, v = self.step(f, rho, u, v)

            if iter % save_step == 0:
                with open(f'{output_dir}/{iter:07d}.dat', 'wb') as file:
                    pickle.dump((jnp.array(self.x), jnp.array(self.y), jnp.array(u), jnp.array(v), jnp.array(rho)), file)

            self.f, self.rho, self.u, self.v = f, rho, u, v

if __name__ == '__main__':
    os.system('cls')

    simulation = lbm(400, 100)
    simulation.run(4000, 10, output_dir='./data3')

    print('Done')