import os
import jax
import time
from tqdm import tqdm
import jax.numpy as jnp
from functools import partial

class LBM():
    def __init__(self, nx, ny, tau=1.0, u0=0.1):
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.u0 = u0
        self.Q = 9

        # --------------------------------------------------
        # 1) D2Q9 constants
        # --------------------------------------------------
        # Weights for each discrete velocity direction
        self.wt = jnp.array([4/9] + [1/9]*4 + [1/36]*4, dtype=jnp.float32)
        # Discrete velocity vectors (ex, ey)
        self.ex = jnp.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=jnp.int32)
        self.ey = jnp.array([0, 0, 1,  0, -1, 1,  1, -1,-1], dtype=jnp.int32)
        # Opposite direction mapping for bounce-back boundary condition
        self.bounce_back = jnp.array([0,3,4,1,2,7,8,5,6], dtype=jnp.int32)

        # --------------------------------------------------
        # 2) Physical coordinates including ghost nodes
        # --------------------------------------------------
        # Domain indices: i=0..nx+1, j=0..ny+1
        self.x, self.y = jnp.meshgrid(
            jnp.arange(nx+2, dtype=jnp.float32) - 0.5,
            jnp.arange(ny+2, dtype=jnp.float32) - 0.5,
            indexing='ij'
        )

        # --------------------------------------------------
        # 3) Macroscopic fields initialization
        # --------------------------------------------------
        shape = (nx+2, ny+2)
        self.u       = jnp.zeros(shape, dtype=jnp.float32)
        self.v       = jnp.zeros(shape, dtype=jnp.float32)
        self.density = jnp.ones(shape,  dtype=jnp.float32)

        # --------------------------------------------------
        # 4) Initial equilibrium distribution (rest state)
        # --------------------------------------------------
        # f[k,i,j] for k=0..8, i=0..nx+1, j=0..ny+1
        feq0 = (self.density[None, :, :] * self.wt[:, None, None])
        self.f = feq0.copy()                  # initial state
        self.ferr = jnp.zeros_like(self.f)    # distribution error

    # ------------------------------------------------------
    # 5) Discrete Maxwell-Boltzmann equilibrium
    # ------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def compute_equilibrium(self, density, u, v):
        """
            fᵢᵉ = ρ Wᵢ (1 + 3 cᵢ ⋅ u + 9/2 (cᵢ ⋅ u)² - 3/2 ||u||₂²)

            fᵢᵉ : Equilibrium discrete velocities
            ρ   : Density
            cᵢ  : Lattice Velocities
            Wᵢ  : Lattice Weights
        """
        usq = u**2 + v**2                                # squared speed field
        cu  = (u[None] * self.ex[:,None,None] +
               v[None] * self.ey[:,None,None])          # dot product e_i · u
        feq = density[None] * self.wt[:,None,None] * (
              1 + 3*cu + 4.5*cu**2 - 1.5*usq[None]
        )

        return feq

    # ------------------------------------------------------
    # 6) BGK collision step
    # ------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def collide(self, f, feq):
        """
            Bhatnagar-Gross-Krook
            
            fᵢ = fᵢ - ω (fᵢ - fᵢᵉ)

            fᵢ  : Discrete velocities
            fᵢᵉ : Equilibrium discrete velocities
            ω   : Relaxation factor
        """
        # Relaxation towards equilibrium
        return f - (f - feq) / self.tau

    # ------------------------------------------------------
    # 7) Pure streaming step (functional, no side effects)
    # ------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def stream(self, f):
        ftemp = f
        # Shift each distribution component by its discrete velocity
        def shift_fk(fk, dx, dy):
            return jnp.roll(jnp.roll(fk, dx, axis=0), dy, axis=1)

        f_stream = jnp.stack([
            shift_fk(f[k], self.ex[k], self.ey[k])
            for k in range(self.Q)
        ], axis=0)
        return f_stream, ftemp

    # ------------------------------------------------------
    # 8) Bounce-back no-slip on all four boundaries
    # ------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def apply_bounce_back(self, f, ftemp):
        """
        Implements no-slip bounce-back:
          f[k, i,    0 ] = ftemp[opp[k], i,    1 ]  # bottom
          f[k, i, ny+1] = ftemp[opp[k], i, ny   ]  # top
          f[k,    0, j] = ftemp[opp[k], 1,    j]  # left
          f[k, nx+1, j] = ftemp[opp[k], nx,   j]  # right
        """
        k_all = jnp.arange(self.Q)
        i = jnp.arange(1, self.nx+1)
        j = jnp.arange(1, self.ny+1)

        # bottom boundary (j=0)
        K,I = jnp.meshgrid(k_all, i, indexing='ij')
        f = f.at[K, I, 0].set(ftemp[self.bounce_back[K], I, 1])

        # top boundary (j=ny+1)
        f = f.at[K, I, self.ny+1].set(ftemp[self.bounce_back[K], I, self.ny])

        # left boundary (i=0)
        J = jnp.arange(1, self.ny+1)
        K,J2 = jnp.meshgrid(k_all, J, indexing='ij')
        f = f.at[K, 0, J2].set(ftemp[self.bounce_back[K], 1, J2])

        # right boundary (i=nx+1)
        f = f.at[K, self.nx+1, J2].set(ftemp[self.bounce_back[K], self.nx, J2])

        return f

    # ------------------------------------------------------
    # 9) Neumann boundary on top (constant velocity)
    # ------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def compute_neumann_bc(self, f, ftemp, density):
        # apply Neumann condition for k=2,5,6 at j=ny (interior layer)
        k_inx = jnp.array([2,5,6], dtype=jnp.int32)
        i_arr = jnp.arange(1, self.nx+1, dtype=jnp.int32)
        # Create 2D index arrays (K x I)
        K, I = jnp.meshgrid(k_inx, i_arr, indexing='ij')
        j = self.ny

        term    = 6 * self.wt[K] * density[I,j] * self.ex[K] * self.u0
        new_vals= ftemp[K, I, j] - term
        bb      = self.bounce_back[K]
        # Update interior layer at j=ny
        f = f.at[bb, I, j].set(new_vals)
        return f

    # ------------------------------------------------------
    # 10) Compute macroscopic variables (density, velocity)
    # ------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def compute_macroscopic_variables(self, f):
        density = jnp.sum(f, axis=0)
        nonzero = density > 1e-8

        u = jnp.sum(f * self.ex[:,None,None], axis=0) / jnp.where(nonzero, density, 1.0)
        v = jnp.sum(f * self.ey[:,None,None], axis=0) / jnp.where(nonzero, density, 1.0)
        u = jnp.where(nonzero, u, 0.0)
        v = jnp.where(nonzero, v, 0.0)

        # You can insert your linear boundary mirror logic here
        return density, u, v

    # ------------------------------------------------------
    # 11) Single LBM step: all kernels functional and jitted
    # ------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def step(self, f, density, u, v):
        feq    = self.compute_equilibrium(density, u, v)
        fcol   = self.collide(f, feq)
        fstr, ftemp = self.stream(fcol)
        fbb    = self.apply_bounce_back(fstr, ftemp)
        fneu   = self.compute_neumann_bc(fbb, ftemp, density)
        rho, u, v = self.compute_macroscopic_variables(fneu)
        return fneu, rho, u, v

    # ------------------------------------------------------
    # 12) Execution loop and frame saving
    # ------------------------------------------------------
    def run(self, steps, save, dir='./data'):
        os.system('cls')

        os.makedirs(dir, exist_ok=True)
        f, rho, u, v = self.f, self.density, self.u, self.v

        for it in tqdm(range(steps)):
            f, rho, u, v = self.step(f, rho, u, v)
            if it % save == 0:
                jnp.save(f'{dir}/{it:07d}.npy', jnp.stack([jnp.array(u), jnp.array(v), jnp.array(rho)]))

        self.f, self.density, self.u, self.v = f, rho, u, v

if __name__ == '__main__':
    start = time.perf_counter()

    nx = 100
    ny = 100

    reynolds = 100
    velocity = 0.1

    sim = LBM(nx, ny, u0 = velocity)
    sim.run(10000, 10, './data_paper')

    print(f'Done ({(time.perf_counter() - start):.3f}s)')