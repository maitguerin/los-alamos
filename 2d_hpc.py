import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dedalus.public as d3
import dedalus.core as dc
from mpi4py import MPI
import logging

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

logging.basicConfig(
    level=logging.INFO if rank == 0 else logging.WARNING,
    format="%(asctime)s [%(levelname)s] [rank %(rank)d] %(message)s",
)

class RankFilter(logging.Filter):
    def filter(self, record):
        record.rank = rank
        return True

logger = logging.getLogger(__name__)
logger.addFilter(RankFilter())

if rank == 0:
    logger.info("Starting 2D shallow water run with %d MPI ranks", size)

Lx = 200.0
Ly = 200.0

Nx = 768
Ny = 768

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

x_local = dist.local_grids(xbasis)[0]
y_local = dist.local_grids(ybasis)[0]

if rank == 0:
    logger.info(
        "Local grid shapes: x_local %s, y_local %s",
        x_local.shape,
        y_local.shape,
    )

a = 1.0

background_undulation = 0.02 * (
    np.cos(2.0 * np.pi * x_local / Lx) * np.cos(2.0 * np.pi * y_local / Ly)
)

sub_center_x = 0.6 * Lx
sub_center_y = 0.5 * Ly
sub_sigma_x = 0.12 * Lx
sub_sigma_y = 0.05 * Ly
sub_amplitude = 0.08

submarine = sub_amplitude * np.exp(
    -(
        (x_local - sub_center_x) ** 2 / (2.0 * sub_sigma_x ** 2)
        + (y_local - sub_center_y) ** 2 / (2.0 * sub_sigma_y ** 2)
    )
)

b["g"] = background_undulation + submarine

H["g"] = 1.0 + 0.5 * (
    np.tanh(a * (x_local - 120.0) * np.ones_like(x_local))
    - np.tanh(a * (x_local - 80.0) * np.ones_like(x_local))
) * (
    np.tanh(a * (y_local - 104.0) * np.ones_like(y_local))
    - np.tanh(a * (y_local - 96.0) * np.ones_like(y_local))
)

u["g"] = 0.0
v["g"] = 0.0

H["g"] = 1.0
v["g"] = 0.25 * (
    np.tanh(a * (x_local - 120.0) * np.ones_like(x_local))
    - np.tanh(a * (x_local - 80.0) * np.ones_like(x_local))
) * (
    np.tanh(a * (y_local - 104.0) * np.ones_like(y_local))
    - np.tanh(a * (y_local - 96.0) * np.ones_like(y_local))
)

if rank == 0:
    b_sample = b.allgather_data("g")
    logger.info(
        "Bathymetry statistics: min %.4e, max %.4e",
        np.min(b_sample),
        np.max(b_sample),
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

problem.add_equation("dt(H) = -dx(H*u) - dy(H*v)")

solver = problem.build_solver(dc.timesteppers.RK443)

timestep_initial = 0.002
t_final = 12.0

intervalsnapshot = int(t_final / (6.0 * timestep_initial))
solver.stop_sim_time = t_final
solver.stop_iteration = int(t_final / timestep_initial) + 1

cfl = d3.CFL(
    solver,
    initial_dt=timestep_initial,
    cadence=10,
    safety=0.3,
    max_change=1.5,
    min_change=0.5,
    max_dt=5.0 * timestep_initial,
    min_dt=1e-6,
    threshold=0.0,
)
cfl.add_velocity(u_vec)
cfl.add_velocity(v_vec)

if rank == 0:
    logger.info(
        "Time-stepping configured: t_final=%.3f, initial dt=%.3e, snapshot every %d iterations",
        t_final,
        timestep_initial,
        intervalsnapshot,
    )

Hlist = []
ulist = []
vlist = []

start_time = MPI.Wtime()

while solver.proceed:
    u_vec["g"][0] = u["g"]
    v_vec["g"][0] = v["g"]

    dt = cfl.compute_timestep()
    solver.step(dt)

    if solver.iteration % intervalsnapshot == 0:
        H_snapshot = H.allgather_data("g")
        u_snapshot = u.allgather_data("g")
        v_snapshot = v.allgather_data("g")

        Hlist.append(H_snapshot)
        ulist.append(u_snapshot)
        vlist.append(v_snapshot)

        if rank == 0:
            logger.info(
                "Completed iteration %d at t = %.3f, dt = %.3e",
                solver.iteration,
                solver.sim_time,
                dt,
            )

end_time = MPI.Wtime()

u_final = u.allgather_data("g")
v_final = v.allgather_data("g")
H_final = H.allgather_data("g")
b_final = b.allgather_data("g")

eta_final = H_final + b_final
speed_final = u_final ** 2 + v_final ** 2

x_global = xbasis.global_grid(dist, scale=1)
y_global = ybasis.global_grid(dist, scale=1)

if rank == 0:
    logger.info("Total wall-clock time: %.3f s", end_time - start_time)

if rank == 0:
    os.makedirs("plots", exist_ok=True)

    midx = Nx // 2
    midy = Ny // 2

    plt.figure()
    plt.plot(x_global, u_final[:, midy], label="u(x, y_mid)")
    plt.plot(x_global, H_final[:, midy], label="H(x, y_mid)")
    plt.plot(x_global, eta_final[:, midy], label="eta=H+b")
    plt.legend()
    plt.savefig("plots/profiles_final_2d_throughcentrey.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(x_global, v_final[midy, :], label="v(x_mid, y)")
    plt.plot(x_global, H_final[midy, :], label="H(x_mid, y)")
    plt.plot(x_global, eta_final[midy, :], label="eta=H+b")
    plt.legend()
    plt.savefig("plots/profiles_final_2d_throughcentrex.png", dpi=200)
    plt.close()

    plt.figure()
    plt.pcolormesh(H_final)
    plt.colorbar()
    plt.title("H final")
    plt.savefig("plots/profiles_final_2dgrid_H.png", dpi=200)
    plt.close()

    plt.figure()
    plt.pcolormesh(b_final)
    plt.colorbar()
    plt.title("Bathymetry b")
    plt.savefig("plots/bathymetry_b.png", dpi=200)
    plt.close()

    plt.figure()
    plt.pcolormesh(eta_final)
    plt.colorbar()
    plt.title("Free surface eta = H + b")
    plt.savefig("plots/free_surface_eta.png", dpi=200)
    plt.close()

    plt.figure()
    plt.pcolormesh(u_final)
    plt.colorbar()
    plt.title("u final")
    plt.savefig("plots/profiles_final_u_2dgrid.png", dpi=200)
    plt.close()

    plt.figure()
    plt.pcolormesh(v_final)
    plt.colorbar()
    plt.title("v final")
    plt.savefig("plots/profiles_final_v_2dgrid.png", dpi=200)
    plt.close()

    plt.figure()
    plt.pcolormesh(speed_final)
    plt.colorbar()
    plt.title("speed^2 final")
    plt.savefig("plots/profiles_final_speed_2dgrid.png", dpi=200)
    plt.close()

    if len(Hlist) >= 6:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            pcm = ax.pcolormesh(Hlist[i])
            ax.set_title(f"H snapshot {i}")
        fig.colorbar(pcm, ax=axes.ravel().tolist())
        plt.tight_layout()
        plt.savefig("plots/snapshotsH.png", dpi=200)
        plt.close()

    logger.info("Plotting complete on rank 0.")
