import os
import jax
import pickle
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
from functools import partial

class LMB:
    def __init__(self, nx, ny, tau=1.0, u0=0.1):
        self.nx, self.ny = nx, ny
        self.tau = tau
        self.u0 = u0
        self.num_vel = 9

        # create meshgrid for coordinates (ghost nodes included)
        xv, yv = jnp.meshgrid(jnp.arange(nx+2) - 0.5, jnp.arange(ny+2) - 0.5, indexing='ij')
        self.x = xv
        self.y = yv

        # lattice weights and directions
        self.wt = jnp.array([4/9] + [1/9]*4 + [1/36]*4)
        self.ex = jnp.array([0,1,0,-1,0,1,-1,-1,1], dtype=jnp.int32)
        self.ey = jnp.array([0,0,1,0,-1,1,1,-1,-1], dtype=jnp.int32)
        self.bounce_back = jnp.array([0,3,4,1,2,7,8,5,6], dtype=jnp.int32)

        # fields
        shape = (nx+2, ny+2)
        self.u = jnp.zeros(shape)
        self.v = jnp.zeros(shape)
        self.rho = jnp.ones(shape)
        self.lattice = jnp.zeros(shape)
        self.f = jnp.zeros((self.num_vel, nx+2, ny+2))

        # set solid boundaries (ghost nodes)
        self.lattice = self.lattice.at[:,0].set(1)
        self.lattice = self.lattice.at[:,-1].set(1)
        self.lattice = self.lattice.at[0,:-1].set(1)
        self.lattice = self.lattice.at[-1,:-1].set(1)

    @partial(jax.jit, static_argnums=0)
    def compute_equilibrium(self, rho, u, v):
        # calculate equilibrium distribution and transpose to (num_vel, nx+2, ny+2)
        cu = (u[..., None] * self.ex) + (v[..., None] * self.ey)
        usq = u**2 + v**2
        feq = (rho[..., None] * self.wt[None, None, :]) * (
            1 + 3 * cu + 4.5 * cu**2 - 1.5 * usq[..., None]
        )
        
        return jnp.transpose(feq, (2, 0, 1))

    @partial(jax.jit, static_argnums=0)
    def collide(self, f, feq):
        return f - (f - feq) / self.tau

    @partial(jax.jit, static_argnums=0)
    def stream(self, f):
        def shift(fk, ex, ey):
            return jnp.roll(jnp.roll(fk, ex, axis=0), ey, axis=1)
        
        return jnp.stack([shift(f[k], self.ex[k], self.ey[k]) for k in range(self.num_vel)])

    @partial(jax.jit, static_argnums=0)
    def apply_bounce_back(self, f):
        def bb(fk, bk):
            return jnp.where(self.lattice > 0, f[bk], fk)
        
        return jnp.stack([bb(f[k], self.bounce_back[k]) for k in range(self.num_vel)])

    @partial(jax.jit, static_argnums=0)
    def apply_neumann_bc(self, f):
        j = self.ny
        k_idx = jnp.array([2, 5, 6], dtype=jnp.int32)
        x_idx = jnp.arange(1, self.nx + 1)
        utop = self.u0 / 2 * (1 + jnp.sin(2 * jnp.pi / self.nx * (x_idx - 0.5) - jnp.pi / 2))
        bb_idx = self.bounce_back[k_idx]
        rho_slice = f.sum(axis=0)[1:self.nx+1, j]
        term = 6 * self.wt[k_idx][:, None] * rho_slice[None, :] * self.ex[k_idx][:, None] * utop[None, :]
        f = f.at[k_idx, 1:self.nx+1, j].set(
            f[bb_idx, 1:self.nx+1, j] - term
        )
        return f

    @partial(jax.jit, static_argnums=0)
    def macroscopic(self, f):
        rho = jnp.sum(f, axis=0)
        u = jnp.sum(f * self.ex[:, None, None], axis=0) / rho
        v = jnp.sum(f * self.ey[:, None, None], axis=0) / rho
        
        u = u.at[0, :].set(2 * u[1, :] - u[2, :]).at[-1, :].set(2 * u[-2, :] - u[-3, :])
        u = u.at[:, 0].set(2 * u[:, 1] - u[:, 2]).at[:, -1].set(2 * u[:, -2] - u[:, -3])
        v = v.at[0, :].set(2 * v[1, :] - v[2, :]).at[-1, :].set(2 * v[-2, :] - v[-3, :])
        v = v.at[:, 0].set(2 * v[:, 1] - v[:, 2]).at[:, -1].set(2 * v[:, -2] - v[:, -3])
        return rho, u, v

    def step(self, f, rho, u, v):
        feq = self.compute_equilibrium(rho, u, v)
        f = self.collide(f, feq)
        f = self.stream(f)
        f = self.apply_bounce_back(f)
        f = self.apply_neumann_bc(f)
        rho, u, v = self.macroscopic(f)
        
        return f, rho, u, v

    def run(self, num_steps, save_step, output_dir='./data'):
        os.system('cls')

        os.makedirs(output_dir, exist_ok=True)

        f, rho, u, v = self.f, self.rho, self.u, self.v
        
        for it in tqdm(range(num_steps)):
            f, rho, u, v = self.step(f, rho, u, v)
        
            if it % save_step == 0:
                filename = f'{output_dir}/{it:07d}.pkl'
                
                with open(filename, 'wb') as fp:
                    pickle.dump((np.array(self.x), np.array(self.y), np.array(u), np.array(v), np.array(rho)), fp)
        
        self.f, self.rho, self.u, self.v = f, rho, u, v
        return rho, u, v

if __name__ == '__main__':
    lbm = LMB(nx=100, ny=100, tau=1.0, u0=0.1)
    lbm.run(num_steps=5000, save_step=10, output_dir='./data')
    
    print('Done')
