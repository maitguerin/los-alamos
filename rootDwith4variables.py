import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import dedalus.core as dc
import logging

logger = logging.getLogger(__name__)

Lx = 200
Ly = 200
Nx = 256
Ny = 256
dtype = np.float64

coords = d3.CartesianCoordinates('x','y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx))
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly))

u = dist.Field(name='u', bases=(xbasis,ybasis))
v = dist.Field(name='v', bases=(xbasis,ybasis))
H = dist.Field(name='H', bases=(xbasis,ybasis))
b = dist.Field(name='b', bases=(xbasis,ybasis))
K = dist.Field(name='K', bases=(xbasis,ybasis))

# Vector proxy for CFL (same spatial dependence as u)
u_vec = dist.VectorField(coords, name='u_vec', bases=(xbasis,ybasis))
v_vec = dist.VectorField(coords, name='v_vec', bases=(xbasis,ybasis))

c = 1
g_accel = 9.8

x_local = dist.local_grids(xbasis)[0]
y_local = dist.local_grids(ybasis)[0]

a = 1 #steepness of initial lock release

dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])


#initial conditions but for speed ad flat surface
b['g'] = 0
H['g'] = 1
v['g'] = 0.25*((
    np.tanh(a*(x_local - 120.0) * np.ones_like(x_local))
    - np.tanh(a*(x_local - 80.0) * np.ones_like(x_local))
))*(np.tanh(a*(y_local - 104.0) * np.ones_like(y_local))
    - np.tanh(a*(y_local - 96.0) * np.ones_like(y_local)))
u['g'] = 0.0
K['g'] = dy(v)['g']

problem = d3.IVP([u, v, H, K], namespace=locals())

problem.add_equation(
    "dt(u)+ c**2/6*dx(dt(K))=" \
    "-c**2/6*dx(K**2)" \
    "-u*dx(u)-v*dy(u)-g_accel*dx(H-b)"
    )

problem.add_equation(
    "dt(v) + c**2/6*dy(dt(K))=" \
    "-c**2/6*dy(K**2)" \
    "-u*dx(v)-v*dy(v)-g_accel*dy(H-b)"
    )

problem.add_equation(
    "dt(H)=-dx(H*u)-dy(H*v)+0.1*(dx(dx(H))+dy(dy(H)))"
    )

problem.add_equation(
    "K=-1/(2*H)*(dx(H*u)+dy(H*v))"
    )

solver = problem.build_solver(dc.timesteppers.RK443)

timestep = 0.1
t_final = 12
snapshotsrate = int(round(t_final/(5*timestep)))

solver.stop_sim_time = t_final
solver.stop_iteration = int(t_final / timestep) + 1

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
cfl.add_velocity(v_vec)

Hlist=[H.allgather_data('g')]
ulist=[u.allgather_data('g')]
vlist=[v.allgather_data('g')]

while solver.proceed:
    # Mirror scalar u into vector proxy for CFL
    u_vec['g'][0] = u['g']
    v_vec['g'][0] = v['g']

    dt = cfl.compute_timestep()
    solver.step(dt)
    if solver.iteration % snapshotsrate == 0:
        logger.info("Completed iteration %d at t = %.3f, dt = %.3e",
                    solver.iteration, solver.sim_time, dt)
        Hlist.append(H.allgather_data('g'))
        ulist.append(u.allgather_data('g'))
        vlist.append(v.allgather_data('g'))

u_final = u.allgather_data('g')
v_final = v.allgather_data('g')
H_final = H.allgather_data('g')
speed_final=u_final**2+v_final**2

x_global = xbasis.global_grid(dist, scale=1)
y_global = ybasis.global_grid(dist, scale=1)

plt.figure()
plt.plot(x_global,u_final[:,int(Nx/2)])
plt.plot(x_global,H_final[:,int(Nx/2)])
plt.savefig('plots/profiles_final_2d_throughcentrey.png', dpi=200)

plt.figure()
plt.plot(x_global,v_final[int(Ny/2),:])
plt.plot(x_global,H_final[int(Ny/2),:])
plt.savefig('plots/profiles_final_2d_throughcentrex.png', dpi=200)

plt.figure()
plt.pcolormesh(H_final)
plt.savefig('plots/profiles_final_2dgrid.png', dpi=200)

plt.figure()
plt.pcolormesh(u_final)
plt.savefig('plots/profiles_final_u_2dgrid.png', dpi=200)

plt.figure()
plt.pcolormesh(v_final)
plt.savefig('plots/profiles_final_v_2dgrid.png', dpi=200)

plt.figure()
plt.pcolormesh(speed_final)
plt.savefig('plots/profiles_final_speed_2dgrid.png', dpi=200)

plt.figure()
fig, ax = plt.subplots(2,3)
for i, ax in enumerate(ax.flat):
    ax.pcolormesh(Hlist[i])
plt.savefig('plots/snapshotsH.png',dpi=200)