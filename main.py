import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import dedalus.core as dc
import logging

logger = logging.getLogger(__name__)

Lx = 200
Nx = 2048
dtype = np.float64

coords = d3.CartesianCoordinates('x')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx))

u = dist.Field(name='u', bases=xbasis)
H = dist.Field(name='H', bases=xbasis)
b = dist.Field(name='b', bases=xbasis)

# Vector proxy for CFL (same spatial dependence as u)
u_vec = dist.VectorField(coords, name='u_vec', bases=xbasis)

c = 1.0
g_accel = 9.8

x_local = dist.local_grids(xbasis)[0]
# H['g'] = 1.0 + 0.5 * (
#     np.tanh(x_local - 4.0 * np.ones_like(x_local))
#     + np.tanh(x_local + 4.0 * np.ones_like(x_local))
# )
b['g'] = -1+0.1*np.sin(x_local*2*np.pi/20)
H['g'] = 1 -(0.5 * (
    np.tanh(x_local - 104.0 * np.ones_like(x_local))
    - np.tanh(x_local - 96.0 * np.ones_like(x_local))
))
u['g'] = 0.0

dx = lambda A: d3.Differentiate(A, coords['x'])

problem = d3.IVP([u, H], namespace=locals())

problem.add_equation(
    "dt(u - (c*c/12)*dx(dx(u))) = "
    "-(g_accel/c)*(dx(H) - dx(b)) "
    "- 2*(u - (c*c/12)*dx(dx(u)))*dx(u) "
    "- u*dx(u - (c*c/12)*dx(dx(u)))"
)
problem.add_equation("dt(H) = -dx(H*u)")

solver = problem.build_solver(dc.timesteppers.RK443)

timestep = 0.0005
t_final = 10

solver.stop_sim_time = t_final
solver.stop_iteration = int(t_final / timestep) + 1

# CFL controller using Dedalus built-in logic
cfl = d3.CFL(
    solver,
    initial_dt=timestep,
    cadence=10,
    safety=0.3,
    max_change=1.5,
    min_change=0.5,
    max_dt=timestep,
    min_dt=1e-6,
    threshold=0.0,
)
cfl.add_velocity(u_vec)

x_global = xbasis.global_grid(dist, scale=1)

diagnostic_interval = 10
snapshot_interval = 20

t_diagnostic = []
u_l2 = []
H_l2 = []
mass_H = []
u_snapshots = []
t_snapshots = []

while solver.proceed:
    # Mirror scalar u into vector proxy for CFL
    u_vec['g'][0] = u['g']

    dt = cfl.compute_timestep()
    solver.step(dt)

    if solver.iteration % diagnostic_interval == 0:
        u_global = u.allgather_data('g')
        H_global = H.allgather_data('g')
        if dist.comm.rank == 0:
            t_diagnostic.append(solver.sim_time)
            u_l2.append(np.sqrt(np.mean(u_global**2)))
            H_l2.append(np.sqrt(np.mean(H_global**2)))
            mass_H.append(np.trapezoid(H_global, x_global))

    if solver.iteration % snapshot_interval == 0:
        u_global = u.allgather_data('g')
        if dist.comm.rank == 0:
            u_snapshots.append(u_global.copy())
            t_snapshots.append(solver.sim_time)

    if solver.iteration % 1000 == 0:
        logger.info("Completed iteration %d at t = %.3f, dt = %.3e",
                    solver.iteration, solver.sim_time, dt)

if dist.comm.rank == 0:
    t_diagnostic = np.array(t_diagnostic)
    u_l2 = np.array(u_l2)
    H_l2 = np.array(H_l2)
    mass_H = np.array(mass_H)
    u_snapshots = np.array(u_snapshots)
    t_snapshots = np.array(t_snapshots)

    u_final = u.allgather_data('g')
    H_final = H.allgather_data('g')

    plt.figure(figsize=(6, 4))
    plt.plot(x_global, u_final, label='u')
    plt.plot(x_global, H_final, label='H')
    plt.xlabel('x')
    plt.legend()
    plt.title('Final profiles')
    plt.tight_layout()
    plt.savefig('plots/profiles_final.png', dpi=200)

    plt.figure(figsize=(6, 4))
    plt.plot(t_diagnostic, u_l2, label=r'||u||$_{L^2}$')
    plt.plot(t_diagnostic, H_l2, label=r'||H||$_{L^2}$')
    plt.xlabel('t')
    plt.ylabel('L2 norm')
    plt.legend()
    plt.title('L2 norms vs time')
    plt.tight_layout()
    plt.savefig('plots/norms_time.png', dpi=200)

    plt.figure(figsize=(6, 4))
    plt.plot(t_diagnostic, mass_H)
    plt.xlabel('t')
    plt.ylabel(r'$\int H\,dx$')
    plt.title('Mass of H vs time')
    plt.tight_layout()
    plt.savefig('plots/mass_time.png', dpi=200)

    if u_snapshots.shape[0] > 1:
        T, X = np.meshgrid(t_snapshots, x_global, indexing='ij')
        plt.figure(figsize=(6, 4))
        plt.pcolormesh(X, T, u_snapshots, shading='gouraud')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('u(x,t) snapshots')
        plt.tight_layout()
        plt.savefig('plots/u_spacetime.png', dpi=200)
