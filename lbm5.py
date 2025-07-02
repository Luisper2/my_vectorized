import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from tqdm import tqdm

class LBM:
    def __init__(self, nx=100, ny=100):
        self.nx, self.ny = nx, ny
        self.tau = 1.0
        # D2Q9 constants
        self.Q = 9
        self.ex = jnp.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=jnp.int32)
        self.ey = jnp.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=jnp.int32)
        self.wt = jnp.array([4/9] + 4*[1/9] + 4*[1/36], dtype=jnp.float32)
        self.bounce_map = jnp.array([0,3,4,1,2,7,8,5,6], dtype=jnp.int32)
        # grid
        self.x, self.y = jnp.meshgrid(jnp.arange(nx+2)-0.5, jnp.arange(ny+2)-0.5, indexing='ij')
        # allocate state arrays
        shape2 = (nx+2, ny+2)
        shape3 = (self.Q, nx+2, ny+2)
        self.f = jnp.zeros(shape3)
        self.feq = jnp.zeros(shape3)
        # Predefine walls list
        self.walls = []

    def define_tau(self, Re, U, L):
        self.tau = U * L / Re

    def set_wall(self, position, fixed=True):
        # record wall positions; apply at streaming step
        if fixed:
            self.walls.append(position)

    def write_output(self, directory, iteration):
        os.makedirs(f'{directory}/dat', exist_ok=True)
        # host arrays
        u, v, rho = self.state_to_macroscopic(self.f)
        data = np.column_stack([np.asarray(a).ravel() for a in (self.x, self.y, u, v, rho)])
        header = 'variables = "x" "y" "u" "v" "rho"\n'
        header += f'zone f=point i={self.nx+2} j={self.ny+2}\n'
        np.savetxt(f'{directory}/dat/{iteration:07d}.dat', data, fmt='%.6f', delimiter=' ', header=header, comments='')

    def save_pkl(self, directory, iteration):
        os.makedirs(f'{directory}/pkl', exist_ok=True)
        u, v, rho = self.state_to_macroscopic(self.f)
        with open(f'{directory}/pkl/{iteration:07d}.pkl','wb') as f:
            pickle.dump((self.x, self.y, u, v, rho), f)

    def state_to_macroscopic(self, f):
        # f: (Q, nx+2, ny+2)
        rho = jnp.sum(f, axis=0)
        u = jnp.sum(f * self.ex[:,None,None], axis=0) / rho
        v = jnp.sum(f * self.ey[:,None,None], axis=0) / rho
        return u, v, rho

    @partial(jax.jit, static_argnums=0)
    def _compute_equilibrium(self, f, u, v, rho):
        # compute feq: (Q, nx+2, ny+2)
        usq = u**2 + v**2
        ue = u[None,...] * self.ex[:,None,None]
        ve = v[None,...] * self.ey[:,None,None]
        feq = rho[None,...] * self.wt[:,None,None] * (1 + 3*(ue+ve) + 4.5*(ue+ve)**2 - 1.5*usq[None,...])
        return feq

    @partial(jax.jit, static_argnums=0)
    def _collide(self, f, feq, tau):
        return f - (f - feq) / tau

    @partial(jax.jit, static_argnums=0)
    def _stream(self, f):
        # vectorized roll for streaming
        roll_fn = lambda fi, dx, dy: jnp.roll(fi, shift=(dx, dy), axis=(0,1))
        return jax.vmap(roll_fn, in_axes=(0,0,0))(f, self.ex, self.ey)

    @partial(jax.jit, static_argnums=0)
    def _apply_bounce(self, f, ftemp, walls):
        # bounce-back for each wall position
        for pos in walls:
            # determine k indices to reflect per pos
            if pos == 'left': ks = jnp.array([3,6,7]); axes=(slice(None),0)
            elif pos=='right': ks=jnp.array([1,5,8]); axes=(slice(None),-1)
            elif pos=='up': ks=jnp.array([2,5,6]); axes=(-1,slice(None))
            elif pos=='down': ks=jnp.array([4,7,8]); axes=(0,slice(None))
            kbs = self.bounce_map[ks]
            # use advanced indexing
            f = f.at[kbs, axes[0], axes[1]].set(ftemp[ks, axes[0], axes[1]])
        return f

    def step(self, f, walls):
        # pure functional step
        u, v, rho = self.state_to_macroscopic(f)
        feq = self._compute_equilibrium(f, u, v, rho)
        f_coll = self._collide(f, feq, self.tau)
        ftemp = f_coll
        f_strm = self._stream(ftemp)
        f_bounce = self._apply_bounce(f_strm, ftemp, walls)
        return f_bounce

    def run(self, steps=10000, save_every=100, directory='./data'):
        os.makedirs(directory, exist_ok=True)
        f = self.f
        walls = self.walls
        for i in tqdm(range(steps)):
            f = self.step(f, walls)
            if i % save_every == 0:
                self.save_pkl(directory, i)
                self.write_output(directory, i)

if __name__ == '__main__':
    sim = LBM(100,100)
    sim.define_tau(80, 0.04, 50/9)
    sim.set_wall('up', fixed=False)
    sim.set_wall('left')
    sim.set_wall('right')
    sim.set_wall('down')
    sim.run(10000, 10, './data4')
