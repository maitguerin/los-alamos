import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import dedalus.core as dc
import logging

logger = logging.getLogger(__name__)

Lx = 200
Ly = 200
Nx = 128
Ny = 128
dtype = np.float64

coords = d3.CartesianCoordinates('x','y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx))
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly))

u = dist.Field(name='u', bases=(xbasis,ybasis))
v = dist.Field(name='v', bases=(xbasis,ybasis))
H = dist.Field(name='H', bases=(xbasis,ybasis))
b = dist.Field(name='b', bases=(xbasis,ybasis))

# Vector proxy for CFL (same spatial dependence as u)
u_vec = dist.VectorField(coords, name='u_vec', bases=(xbasis,ybasis))
v_vec = dist.VectorField(coords, name='v_vec', bases=(xbasis,ybasis))

c = 1.0
g_accel = 9.8

x_local = dist.local_grids(xbasis)[0]
y_local = dist.local_grids(ybasis)[0]

#Set inital conditions
b['g'] = -1
H['g'] = 1 -((
    np.tanh(x_local - 104.0 * np.ones_like(x_local))
    - np.tanh(x_local - 96.0 * np.ones_like(x_local))
))*(np.tanh(y_local - 104.0 * np.ones_like(y_local))
    - np.tanh(y_local - 96.0 * np.ones_like(y_local)))
H['g']=1 -0.5*((
    np.tanh(x_local - 104.0 * np.ones_like(x_local))
    - np.tanh(x_local - 96.0 * np.ones_like(x_local))
))
u['g'] = 0.0
v['g'] = 0.0

dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])

problem = d3.IVP([u, v, H], namespace=locals())

#p=(u-(c*c)/12*(dx(dx(u))+dx(dy(v))))
#q=(v-(c*c)/12*(dx(dy(u))+dy(dy(v))))

problem.add_equation(
    "dt(u-(c*c)/12*(dx(dx(u))+dx(dy(v))))="
    "-u*dx(u-(c*c)/12*(dx(dx(u))+dx(dy(v))))"
    "-v*dy(u-(c*c)/12*(dx(dx(u))+dx(dy(v))))"
    "-2*dx(u)*(u-(c*c)/12*(dx(dx(u))+dx(dy(v))))"
    "-dx(v)*(v-(c*c)/12*(dx(dy(u))+dy(dy(v))))"
    "-dy(v)*(u-(c*c)/12*(dx(dx(u))+dx(dy(v))))"
    "+g_accel/c*dx(H-b)"
    )

problem.add_equation(
    "dt(v-(c*c)/12*(dx(dy(u))+dy(dy(v))))="
    "-u*dx(v-(c*c)/12*(dx(dy(u))+dy(dy(v))))"
    "-v*dy(v-(c*c)/12*(dx(dy(u))+dy(dy(v))))"
    "-dy(u)*(u-(c*c)/12*(dx(dx(u))+dx(dy(v))))"
    "-2*dy(v)*(v-(c*c)/12*(dx(dy(u))+dy(dy(v))))"
    "-dx(u)*(v-(c*c)/12*(dx(dy(u))+dy(dy(v))))"
    "+g_accel/c*dy(H-b)"
    )

problem.add_equation(
    "dt(H)=-dx(H*u)-dy(H*v)"
    )

solver = problem.build_solver(dc.timesteppers.RK443)

timestep = 0.00005
t_final = 1

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

while solver.proceed:
    # Mirror scalar u into vector proxy for CFL
    u_vec['g'][0] = u['g']

    dt = cfl.compute_timestep()
    solver.step(dt)
    if solver.iteration % 1000 == 0:
        logger.info("Completed iteration %d at t = %.3f, dt = %.3e",
                    solver.iteration, solver.sim_time, dt)