import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import dedalus.public as d3
import dedalus.core as dc
from mpi4py import MPI
import logging

class RankFilter(logging.Filter):
    def filter(self, record):
        record.rank = MPI.COMM_WORLD.rank
        return True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [rank %(rank)d] %(message)s",
)
logging.getLogger().addFilter(RankFilter())
logger = logging.getLogger(__name__)

Lx = 200.0
Ly = 200.0
Nx = 1024
Ny = 1024
dtype = np.float64

coords = d3.CartesianCoordinates("x", "y")
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords["x"], size=Nx, bounds=(0.0, Lx))
ybasis = d3.RealFourier(coords["y"], size=Ny, bounds=(0.0, Ly))

u = dist.Field(name="u", bases=(xbasis, ybasis))
v = dist.Field(name="v", bases=(xbasis, ybasis))
H = dist.Field(name="H", bases=(xbasis, ybasis))
b = dist.Field(name="b", bases=(xbasis, ybasis))

u_vec = dist.VectorField(coords, name="u_vec", bases=(xbasis, ybasis))
v_vec = dist.VectorField(coords, name="v_vec", bases=(xbasis, ybasis))

c = 1.0
g_accel = 9.8

x_local, = dist.local_grids(xbasis)
y_local, = dist.local_grids(ybasis)

comm = dist.comm
logger.info(
    "Initialised Distributor with global grid (%d, %d), local shape for u['g'] %s, size %d",
    Nx,
    Ny,
    repr(u["g"].shape),
    comm.size,
)

a = 1.0

background_undulation = 0.02 * (
    np.cos(2.0 * np.pi * x_local / Lx) * np.cos(2.0 * np.pi * y_local / Ly)
)

sub_center_x = 0.6 * Lx
sub_center_y = 0.5 * Ly
sub_sigma_x = 0.05 * Lx
sub_sigma_y = 0.02 * Ly
sub_amplitude = 0.08

submarine = sub_amplitude * np.exp(
    -(
        (x_local - sub_center_x) ** 2 / (2.0 * sub_sigma_x ** 2)
        + (y_local - sub_center_y) ** 2 / (2.0 * sub_sigma_y ** 2)
    )
)

b["g"] = background_undulation + submarine

H_lock = 1.0 + 0.5 * (
    np.tanh(a * (x_local - 120.0))
    - np.tanh(a * (x_local - 80.0))
) * (
    np.tanh(a * (y_local - 104.0))
    - np.tanh(a * (y_local - 96.0))
)

H["g"] = 1.0
u["g"] = 0.0
v["g"] = 0.25 * (
    np.tanh(a * (x_local - 120.0))
    - np.tanh(a * (x_local - 80.0))
) * (
    np.tanh(a * (y_local - 104.0))
    - np.tanh(a * (y_local - 96.0))
)

dx = lambda A: d3.Differentiate(A, coords["x"])
dy = lambda A: d3.Differentiate(A, coords["y"])

problem = d3.IVP([u, v, H], namespace=locals())

problem.add_equation(
    "dt(u-(c*c)/12*(dx(dx(u))+dx(dy(v))))="
    "-u*dx(u-(c*c)/12*(dx(dx(u))+dx(dy(v))))"
    "-v*dy(u-(c*c)/12*(dx(dx(u))+dx(dy(v))))"
    "-2*dx(u)*(u-(c*c)/12*(dx(dx(u))+dx(dy(v))))"
    "-dx(v)*(v-(c*c)/12*(dx(dy(u))+dy(dy(v))))"
    "-dy(v)*(u-(c*c)/12*(dx(dx(u))+dx(dy(v))))"
    "-g_accel/c*dx(H-b)"
)

problem.add_equation(
    "dt(v-(c*c)/12*(dx(dy(u))+dy(dy(v))))="
    "-u*dx(v-(c*c)/12*(dx(dy(u))+dy(dy(v))))"
    "-v*dy(v-(c*c)/12*(dx(dy(u))+dy(dy(v))))"
    "-dy(u)*(u-(c*c)/12*(dx(dx(u))+dx(dy(v))))"
    "-2*dy(v)*(v-(c*c)/12*(dx(dy(u))+dy(dy(v))))"
    "-dx(u)*(v-(c*c)/12*(dx(dy(u))+dy(dy(v))))"
    "-g_accel/c*dy(H-b)"
)

problem.add_equation("dt(H)=-dx(H*u)-dy(H*v)")

solver = problem.build_solver(dc.timesteppers.RK443)
logger.info("Solver built using RK443")

timestep = 0.002
t_final = 12.0
intervalsnapshot = int(t_final / (6.0 * timestep))
solver.stop_sim_time = t_final
solver.stop_iteration = int(t_final / timestep) + 1

cfl = d3.CFL(
    solver,
    initial_dt=timestep,
    cadence=10,
    safety=0.3,
    max_change=1.5,
    min_change=0.5,
    max_dt=5.0 * timestep,
    min_dt=1.0e-6,
    threshold=0.0,
)
cfl.add_velocity(u_vec)
cfl.add_velocity(v_vec)

if comm.rank == 0:
    Hlist = []
    ulist = []
    vlist = []
else:
    Hlist = None
    ulist = None
    vlist = None

output_directory = "plots"
if comm.rank == 0:
    os.makedirs(output_directory, exist_ok=True)

logger.info(
    "Starting time integration up to t_final = %.3f with initial dt = %.3e, snapshot interval = %d",
    t_final,
    timestep,
    intervalsnapshot,
)

while solver.proceed:
    u_vec["g"][0] = u["g"]
    v_vec["g"][0] = v["g"]

    dt = cfl.compute_timestep()
    solver.step(dt)

    if solver.iteration % intervalsnapshot == 0 or not solver.proceed:
        H_global = H.allgather_data("g")
        u_global = u.allgather_data("g")
        v_global = v.allgather_data("g")

        if comm.rank == 0:
            logger.info(
                "Iteration %d, t = %.6f, dt = %.3e, max|u| = %.3e, max|v| = %.3e",
                solver.iteration,
                solver.sim_time,
                dt,
                np.max(np.abs(u_global)),
                np.max(np.abs(v_global)),
            )
            if solver.iteration % intervalsnapshot == 0:
                Hlist.append(H_global)
                ulist.append(u_global)
                vlist.append(v_global)

u_final = u.allgather_data("g")
v_final = v.allgather_data("g")
H_final = H.allgather_data("g")

if comm.rank == 0:
    logger.info(
        "Finished integration at iteration %d, t = %.6f",
        solver.iteration,
        solver.sim_time,
    )

x_global = xbasis.global_grid(dist, scale=1)
y_global = ybasis.global_grid(dist, scale=1)

if comm.rank == 0:
    speed_final = u_final ** 2 + v_final ** 2

    plt.figure()
    plt.plot(x_global, u_final[:, int(Nx / 2)])
    plt.plot(x_global, H_final[:, int(Nx / 2)])
    plt.xlabel("x")
    plt.ylabel("u, H")
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "profiles_final_2d_throughcentrey.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(x_global, v_final[int(Ny / 2), :])
    plt.plot(x_global, H_final[int(Ny / 2), :])
    plt.xlabel("x")
    plt.ylabel("v, H")
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "profiles_final_2d_throughcentrex.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.pcolormesh(H_final)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "profiles_final_2dgrid.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.pcolormesh(u_final)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "profiles_final_u_2dgrid.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.pcolormesh(v_final)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "profiles_final_v_2dgrid.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.pcolormesh(speed_final)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "profiles_final_speed_2dgrid.png"), dpi=200)
    plt.close()

    if Hlist is not None and len(Hlist) >= 6:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            ax.pcolormesh(Hlist[i])
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, "snapshotsH.png"), dpi=200)
        plt.close()

logger.info("Rank %d finished cleanly", comm.rank)
