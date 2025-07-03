import os
import sys
import jax
import pickle
import numpy as np
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

        self.prepare_bounce()
        
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
        self.vorticity = jnp.zeros(shape)
        
        shape = (self.number_velocities, self.nx + 2, self.ny + 2)
        self.feq = jnp.zeros(shape)
        self.f = jnp.zeros(shape)
        self.ferr = jnp.zeros(shape)
        self.ftemp = self.f.copy()

    def prepare_bounce(self):
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
            ks = jnp.array(ks, dtype=jnp.int32)
            kbs = self.bounce[ks]

            i_idx, j_idx = index[pos]
            i_vals = (jnp.arange(i_idx.start, i_idx.stop)
                      if isinstance(i_idx, slice)
                      else jnp.array([i_idx]))
            j_vals = (jnp.arange(j_idx.start, j_idx.stop)
                      if isinstance(j_idx, slice)
                      else jnp.array([j_idx]))

            I, J = jnp.meshgrid(i_vals, j_vals, indexing='ij')
            shape = (ks.shape[0],) + I.shape

            K  = jnp.broadcast_to(ks[:,None,None],   shape)
            KB = jnp.broadcast_to(kbs[:,None,None],  shape)
            I3 = jnp.broadcast_to(I[None,...],       shape)
            J3 = jnp.broadcast_to(J[None,...],       shape)

            self._bb_idx[pos] = (K, KB, I3, J3)
    
    def define_tau(self, reynolds_number: float, velocity: float, lenght: float):
        self.tau = velocity * lenght / reynolds_number

    def wall(self, position: str, fixed: bool = True):
        direction = {
            'up': (-1, slice(None)),
            'left': (slice(None), 0),
            'right': (slice(None), -1),
            'down': (0, slice(None)),
        }

        side = direction.get(position)

        if side is None:
            print(f'Error setting wall at {position}')
            sys.exit()

        self.lattice = self.lattice.at[side].set(1)

        if fixed:
            self.walls.append(position)

    def compute_equilibrium(self):
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
        vals =  self.ftemp[K, I3, J3]

        self.f = self.f.at[KB, I3, J3].set(vals)

    def compute_neumann_bc(self):
        j = self.ny
        k_inx = [2,5,6]
        
        for i in range(1, self.nx + 1):
            utop = 0.04 / 2. * (1. + jnp.sin(2. * jnp.pi / self.nx * (i - 0.5) - jnp.pi / 2.))
            
            self.f[self.bounce[k_inx], i, j] = self.ftemp[k_inx, i, j] - 6 * self.weight[k_inx] * self.rho[i, j] * self.ex[k_inx] * utop

    def compute_distribution(self):
        self.ferr = jnp.abs(self.f - self.ftemp)

    def compute_macroscopic(self):
        rho = jnp.sum(self.f, axis=0)  # shape (nx,ny)

        nz = rho > 1e-8

        u = jnp.sum(self.ex[:, None, None] * self.f, axis=0)
        v = jnp.sum(self.ey[:, None, None] * self.f, axis=0)

        denom = jnp.where(nz, rho, 1.0)
        
        u = u / denom
        v = v / denom
        
        u = jnp.where(nz, u, 0.0)
        v = jnp.where(nz, v, 0.0)

        j_slice = slice(1, self.ny)
        u = u.at[0,    j_slice].set(  2*u[1,    j_slice] - u[2,    j_slice])
        u = u.at[-1,   j_slice].set(  2*u[-2,   j_slice] - u[-3,   j_slice])
        v = v.at[0,    j_slice].set(  2*v[1,    j_slice] - v[2,    j_slice])
        v = v.at[-1,   j_slice].set(  2*v[-2,   j_slice] - v[-3,   j_slice])
        rho = rho.at[0,  j_slice].set(2*rho[1,  j_slice] - rho[2,  j_slice])
        rho = rho.at[-1, j_slice].set(2*rho[-2, j_slice] - rho[-3, j_slice])

        i_slice = slice(None)
        u = u.at[i_slice, 0].set(  2*u[i_slice, 1] - u[i_slice, 2])
        u = u.at[i_slice,-1].set(  2*u[i_slice,-2] - u[i_slice,-3])
        v = v.at[i_slice, 0].set(  2*v[i_slice, 1] - v[i_slice, 2])
        v = v.at[i_slice,-1].set(  2*v[i_slice,-2] - v[i_slice,-3])
        rho = rho.at[i_slice, 0].set(2*rho[i_slice, 1] - rho[i_slice, 2])
        rho = rho.at[i_slice,-1].set(2*rho[i_slice,-2] - rho[i_slice,-3])

        self.rho = rho
        self.u = u
        self.v = v

    def compute_vorticity(self):
        du_dx, du_dy = jnp.gradient(self.u)
        dv_dx, dv_dy = jnp.gradient(self.v)

        self.vorticity = du_dy - dv_dx

    def save_pkl(self, directory: str, iteration: str):
        os.makedirs(f'{directory}/pkl', exist_ok = True)
        
        with open(f'{directory}/pkl/{iteration:07d}.pkl', 'wb') as file:
            pickle.dump((self. x, self.y, self.u, self.v, self.vorticity, self.rho), file)

    def save_dat(self, directory: str, iteration: str):
        os.makedirs(f'{directory}/dat', exist_ok = True)

        data = jnp.column_stack((
            jnp.asarray(self.x).ravel(),
            jnp.asarray(self.y).ravel(),
            jnp.asarray(self.u).ravel(),
            jnp.asarray(self.v).ravel(),
            jnp.asarray(self.vorticity).ravel(),
            jnp.asarray(self.rho).ravel()
        ))

        np.savetxt(f'{directory}/dat/{iteration}.dat', data, fmt='%.6f', delimiter=' ', comments='')

    def run(self, steps: int = 10000, save: int = 100, dir: str = './data'):
        os.system('cls')
        os.makedirs(dir, exist_ok = True)

        for i in tqdm(range(steps)):
            simulation.compute_equilibrium()
            simulation.compute_collisions()
            simulation.compute_streaming()

            for side in self.walls:
                simulation.compute_bounceback_bc(side)
            
            simulation.compute_neumann_bc()

            simulation.compute_distribution()
            simulation.compute_macroscopic()

            if i % save == 0:
                simulation.save_pkl(dir, i)
                simulation.save_dat(dir, i)

if __name__ == '__main__':
    simulation = lbm(100, 100)
    simulation.define_tau(80, 0.04, 50 // 9)
    simulation.wall('up', False)
    simulation.wall('left')
    simulation.wall('right')
    simulation.wall('down')
    simulation.run(10000, 10, './data4')