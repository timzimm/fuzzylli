import argparse
import logging
import socket

from mpi4py import MPI

import h5py

import jax
import jax.numpy as jnp
from jax.config import config
import pyvista as pv
import numpy as np
from yt.utilities.decompose import get_psize

from fuzzylli.domain import UniformHypercube
from fuzzylli.units import set_schroedinger_units
from fuzzylli.filaments import (
    init_finite_straight_filament_spine_gas,
    generic_finite_cylinder_density,
    finite_straight_filament_spine_params,
    Rphiz_from_xyz,
)
import fuzzylli.wavefunction as psi
from fuzzylli.potential import AxialSymmetricPotential
from fuzzylli.density import SteadyStateCylinder

from fuzzylli.io_utils import (
    create_ds,
    load_or_compute_model,
    save_dict_to_group,
    dictify,
    load_parameters_from_config,
)
from fuzzylli.eigenstates import eigenstate_library, init_eigenstate_library
from fuzzylli.ray import init_ray_params
from fuzzylli.cosmology import h
from fuzzylli.df import ConstantAnisotropyDistribution


def init_wavefunction(
    load_if_cached, init_routine, optimize_args, rho_kwargs, df_kwargs, N
):
    """
    Bulk init of all componenets leading to wavfunction fit, including
    wavefunction itself.
    """
    a_fit = rho_kwargs["scalefactor"]
    rho = SteadyStateCylinder(**rho_kwargs)
    V = AxialSymmetricPotential(lambda R: rho(R) / a_fit, rho.R999)
    eigenstate_args = {
        "for_name": [a_fit, V, V(rho.R99), N],
        "for_compute": [a_fit, V, V(rho.R99), N],
    }
    eigenstate_lib = load_or_compute_model(
        load_if_cached,
        cache_dir,
        eigenstate_library.compute_name,
        init_eigenstate_library,
        **eigenstate_args,
    )
    df = ConstantAnisotropyDistribution(
        rho_kwargs["beta"],
        a_fit,
        V,
        rho,
        **df_kwargs,
        R_max=rho.R999,
    )
    optimization_args = {
        "for_name": [
            init_routine,
            eigenstate_lib,
            rho,
            rho.R99,
            *optimize_args,
        ],
        "for_compute": [eigenstate_lib, rho, df, rho.R99, *optimize_args],
    }

    wavefunction_params = load_or_compute_model(
        load_if_cached,
        cache_dir,
        psi.wavefunction_params.compute_name,
        init_routine,
        **optimization_args,
    )
    return wavefunction_params


# 3D densities as finite extrusions ...
def _fdm_cylinder(x, t, spine_params, wavefunction_params):
    return generic_finite_cylinder_density(
        lambda R, phi, wavefunction_params: jnp.abs(
            psi.psi(R, phi, t, wavefunction_params)
        )
        ** 2,
        x,
        spine_params,
        wavefunction_params,
    )


def _wdm_cylinder(x, spine_params, wavefunction_params):
    return generic_finite_cylinder_density(
        lambda R, phi, wavefunction_params: psi.rho(R, wavefunction_params),
        x,
        spine_params,
        wavefunction_params,
    )


def _cdm_cylinder(x, spine_params, rho_bg, R_cut):
    return generic_finite_cylinder_density(
        lambda R, phi, wavefunction_params: rho_bg.truncated_density(R, R_cut),
        x,
        spine_params,
        None,
    )


# ... batched on coordinate x. map instead of vmap to limit memory footprint
def eval_fdm_cylinder_density(x, t, spine_params, wavefunction_params):
    return jax.lax.map(
        lambda x: _fdm_cylinder(x, t, spine_params, wavefunction_params), x
    )


def eval_wdm_cylinder_density(x, spine_params, wavefunction_params):
    return jax.lax.map(lambda x: _wdm_cylinder(x, spine_params, wavefunction_params), x)


def eval_cdm_cylinder_density(x, spine_params, rho_bg, R_cut):
    return jax.lax.map(lambda x: _cdm_cylinder(x, spine_params, rho_bg, R_cut), x)


def eval_r(x, spine_params, rho_bg):
    return jax.lax.map(lambda x: Rphiz_from_xyz(x, spine_params)[1], x)


init_rays_params = jax.vmap(init_ray_params)

config.update("jax_enable_x64", True)
logging.basicConfig(
    level=logging.INFO,
    format="\x1b[33;20m%(asctime)s {}\x1b[0m: %(message)s".format(socket.gethostname()),
)
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser()
parser.add_argument("yaml_file", type=str, help="Path to YAML parameter file")
args = parser.parse_args()

params = load_parameters_from_config(args.yaml_file)

filename = params["general"]["output_file"]
cache_dir = params["general"]["cache"]
load_if_cached = params["general"]["load_if_cached"]

# Units setup
m22 = params["cosmology"]["m22"]
u = set_schroedinger_units(m22)

# Box setup
MM = params["domain"]["N"]
L = params["domain"]["L"] * u.from_Mpc / h
if rank == 0:
    logger.info(f"dx = {L * u.to_Mpc/MM} Mpc")

seed = params["general"]["seed"]
K = params["eigenstate_library"]["N"]
z_fit = params["cosmology"]["z"]
a_fit = 1.0 / (1 + z_fit)
beta = params["background_density"]["beta"]
N = len(params["filament_ensemble"]["orientation"])

init_routines = {
    "least_square": psi.init_wavefunction_params_least_square,
    "aLASSO": psi.init_wavefunction_params_adaptive_lasso,
    "regPoisson": psi.init_wavefunction_params_poisson_process,
}
init_routine = init_routines[params["wave_function"]["minimize"]]

spine_gas_params = None
cylinders = None
ts = None
steady_state_density_params = None
df_kwargs = None
optimize_args = None

domain = UniformHypercube(
    [1, 1, 1],
    jnp.array([[0.0, L], [0.0, L], [0.0, L]]),
)

if rank == 0:
    f = h5py.File(filename, "w")
    f["/"].attrs.create("L_Mpc_div_h", L * h * u.to_Mpc)
    f["/"].attrs.create("m22_1e-22eV", m22)
    f["/"].attrs.create("N", N)
    f["/"].attrs.create("M", MM)
    f["/"].attrs.create("z", z_fit)

    M_cylinder = jnp.array(params["filament_ensemble"]["mass"])

    spine_gas_params = []
    for i in range(N):
        length_i = jnp.array(params["filament_ensemble"]["length"][i]) * u.from_Mpc / h
        spine_dir_i = jnp.array(params["filament_ensemble"]["direction"][i])
        spine_orient_i = jnp.array(params["filament_ensemble"]["orientation"][i])
        spine_origin_i = jnp.array(params["filament_ensemble"]["origin"][i])
        spine_dir_i = spine_dir_i / np.linalg.norm(spine_dir_i)

        spine_gas_params.append(
            finite_straight_filament_spine_params(
                ray_params=init_rays_params(L * spine_origin_i, spine_dir_i),
                length=length_i,
                orientation=spine_orient_i,
            )
        )

    logger.info(f"{len(spine_gas_params)} cylinder(s) placed")

    steady_state_density_params = []
    df_kwargs = []
    optimize_args = []
    cylinder_grps = [f.create_group(f"cylinders/{i}") for i in range(N)]
    for i, M in enumerate(M_cylinder):
        beta_i = params["background_density"]["beta"][i]
        r0_i = params["background_density"]["r0"][i]
        sigma_i = params["background_density"]["sigma"][i]
        line_mass_per_particle_i = (
            params["wave_function"]["mass_per_particle"]
            * u.from_Msun
            / (jnp.array(params["filament_ensemble"]["length"][i]) * u.from_Mpc)
        )

        steady_state_density_params.append(
            {
                "beta": beta_i,
                "scalefactor": a_fit,
                "r0": r0_i * u.from_Mpc / h,
                "sigma2": (sigma_i * u.from_kms) ** 2,
            }
        )
        cylinder_grps[i].attrs.create("M", M)
        cylinder_grps[i].attrs.create("beta", beta_i)
        cylinder_grps[i].attrs.create("r0_Mpc_div_h", r0_i)
        cylinder_grps[i].attrs.create("scalefactor", a_fit)
        cylinder_grps[i].attrs.create("sigma2", sigma_i**2)
        save_dict_to_group(f, f"/cylinders/{i}/", dictify(spine_gas_params[i]))

        df_kwargs.append(
            {
                "R_min": params["phasespace_distribution"]["R_min"][i] * u.from_Mpc / h,
                "epochs": params["phasespace_distribution"]["epochs"][i],
            }
        )
        select_optimize_args = {
            "least_square": [seed],
            "aLASSO": [seed],
            "regPoisson": [line_mass_per_particle_i, seed],
        }
        optimize_args.append(select_optimize_args[params["wave_function"]["minimize"]])

    ts = [0.0]


spine_gas_params = comm.bcast(spine_gas_params, root=0)
ts = comm.bcast(ts, root=0)
df_kwargs = comm.bcast(df_kwargs, root=0)
optimize_args = comm.bcast(optimize_args, root=0)
K = comm.bcast(K, root=0)

steady_state_density_params = comm.bcast(steady_state_density_params, root=0)
if rank < len(steady_state_density_params):
    logger.info(
        f"r0={steady_state_density_params[rank]['r0'].item() * u.to_Kpc:.2f} kpc, "
        f'sigma={jnp.sqrt(steady_state_density_params[rank]["sigma2"].item()) * u.to_kms:.2f} km/s, '
        f"length={spine_gas_params[rank].length.item() * u.to_Mpc:.2f} Mpc"
    )

cylinder_idx = np.array_split(np.arange(N), size)
nocdm = not params["general"]["save_cdm"]
nowdm = not params["general"]["save_wdm"]
nofdm = not params["general"]["save_fdm"]
if not (nowdm and nofdm):
    wavefunctions_params = [
        init_wavefunction(
            load_if_cached,
            init_routine,
            optimize_args[i],
            steady_state_density_params[i],
            df_kwargs[i],
            K[i],
        )
        for i in cylinder_idx[rank]
    ]
    wavefunctions_params = [
        params for params in comm.allgather(wavefunctions_params) if params != []
    ]
    wavefunctions_params = [
        item for sublist in wavefunctions_params for item in sublist
    ]
    if rank == 0:
        for i, wavefunction_params in enumerate(wavefunctions_params):
            save_dict_to_group(
                f, f"/cylinders/{i}/wavefunction_params/", dictify(wavefunction_params)
            )

if not nocdm:
    rhos_bg = [
        SteadyStateCylinder(**steady_state_density_params[i])
        for i in cylinder_idx[rank]
    ]
    rhos_bg = [params for params in comm.allgather(rhos_bg) if params != []]
    rhos_bg = [item for sublist in rhos_bg for item in sublist]

if rank == 0:
    logger.info("All cylinders initialized")

# Pencil decomposition in yz
M = get_psize(np.array([1, MM, MM]), size) if size > 1 else np.array([1, 1, 1])
dx = L / M
MM_loc = (np.array([MM, MM, MM]) / M).astype(int)
pencil = comm.Create_cart(dims=M[1:])
ij = pencil.Get_coords(rank)

domain = UniformHypercube(
    MM_loc,
    jnp.array(
        [
            [0, L],
            [ij[0] * dx[1], (ij[0] + 1) * dx[1]],
            [ij[1] * dx[2], (ij[1] + 1) * dx[2]],
        ]
    ),
)
logger.info(f"Local pencil: {domain.N}")
box = pv.Cube(bounds=domain.extends.flatten())

filaments_in_box = {}
translations = UniformHypercube(
    [3, 3, 3], jnp.array([[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])
)
for idx in range(N):
    vs = []
    for v in jnp.array(jnp.array([L, L, L]) * translations.cell_centers_cartesian):
        # for v in [0]:
        r = rhos_bg[idx].R99 if nofdm and nowdm else wavefunctions_params[idx].R_fit
        cylinder = pv.Cylinder(
            center=spine_gas_params[idx].ray_params.origin + v,
            direction=spine_gas_params[idx].ray_params.direction,
            height=spine_gas_params[idx].length,
            radius=r,
        )
        inside = cylinder.select_enclosed_points(box)
        N_enclosed = cylinder.extract_points(
            inside["SelectedPoints"].view(bool),
        ).n_points
        N_collision = cylinder.collision(box)[1]
        if N_enclosed > 0 or N_collision > 0:
            vs.append(v)
    if len(vs) > 0:
        filaments_in_box[idx] = vs


comm.Barrier()
if rank == 0:
    f.close()
f = h5py.File(filename, "r+", driver="mpio", comm=comm)
f.atomic = False

# if args.save_coord:
#     if "coordinates" not in f:
#         grp = f.create_group("coordinates")
#     grp = f["coordinates"]
#     dset = create_ds(grp, "radius", (MM, MM, MM))

#     R_xyz = np.zeros(shape=MM_loc)
#     if len(filaments_in_box) > 0:
#         xyz = jax.lax.stop_gradient(domain.cell_centers_cartesian)
#     for idx, vs in filaments_in_box.items():
#         for v in vs:
#             filament_spine = finite_straight_filament_spine_params(
#                 ray_params=init_ray_params(
#                     origin=spine_gas_params[idx].ray_params.origin + v,
#                     direction=spine_gas_params[idx].ray_params.direction,
#                 ),
#                 length=spine_gas_params[idx].length,
#                 orientation=spine_gas_params[idx].orientation,
#             )
#             R_xyz[:] += np.asarray(eval_r(xyz, filament_spine, None)).reshape(
#                 R_xyz.shape
#             )
#             logging.info(
#                 f"Pencil {ij} constructed R for cylinder {idx} from {v/jnp.array([L,L,L])}"
#             )

#     with dset.collective:
#         dset[
#             :,
#             ij[0] * MM_loc[1] : (ij[0] + 1) * MM_loc[1],
#             ij[1] * MM_loc[2] : (ij[1] + 1) * MM_loc[2],
#         ] = R_xyz

#     logging.info(f"Pencil {ij} saved WDM rho")

for i, t in enumerate(ts):
    comm.Barrier()

    rho_xyz = np.zeros(shape=MM_loc)
    if len(filaments_in_box) > 0:
        xyz = jax.lax.stop_gradient(domain.cell_centers_cartesian)

    if not nofdm:
        if rank == 0:
            logging.info(f"Construct density at t={t * u.to_Myr}")
        if i == 0:
            grp = f.create_group("density")
            grp = f.create_group("density/fdm")
        dset = create_ds(grp, f"{i}", (MM, MM, MM))
        dset.attrs.create("t_Myr", t * u.to_Myr)
        for idx, vs in filaments_in_box.items():
            for v in vs:
                filament_spine = finite_straight_filament_spine_params(
                    ray_params=init_ray_params(
                        origin=spine_gas_params[idx].ray_params.origin + v,
                        direction=spine_gas_params[idx].ray_params.direction,
                    ),
                    length=spine_gas_params[idx].length,
                    orientation=spine_gas_params[idx].orientation,
                )
                rho_xyz[:] += np.asarray(
                    eval_fdm_cylinder_density(
                        xyz, t, filament_spine, wavefunctions_params[idx]
                    )
                ).reshape(rho_xyz.shape)

                logging.info(
                    f"Pencil {ij} constructed FDM rho with cylinder {idx} from {v/jnp.array([L,L,L])}"
                )
        with dset.collective:
            dset[
                :,
                ij[0] * MM_loc[1] : (ij[0] + 1) * MM_loc[1],
                ij[1] * MM_loc[2] : (ij[1] + 1) * MM_loc[2],
            ] = rho_xyz
        logging.info(f"Pencil {ij} saved FDM rho")

if not nowdm:
    if "density" not in f:
        grp = f.create_group("density")
    grp = f["density"]
    dset = create_ds(grp, "wdm", (MM, MM, MM))

    if len(filaments_in_box) > 0:
        rho_xyz[:] = 0.0
    for idx, vs in filaments_in_box.items():
        for v in vs:
            filament_spine = finite_straight_filament_spine_params(
                ray_params=init_ray_params(
                    origin=spine_gas_params[idx].ray_params.origin + v,
                    direction=spine_gas_params[idx].ray_params.direction,
                ),
                length=spine_gas_params[idx].length,
                orientation=spine_gas_params[idx].orientation,
            )
            rho_xyz[:] += np.asarray(
                eval_wdm_cylinder_density(
                    xyz, filament_spine, wavefunctions_params[idx]
                )
            ).reshape(rho_xyz.shape)
            logging.info(
                f"Pencil {ij} constructed WDM rho with cylinder {idx} from {v/jnp.array([L,L,L])}"
            )

    with dset.collective:
        dset[
            :,
            ij[0] * MM_loc[1] : (ij[0] + 1) * MM_loc[1],
            ij[1] * MM_loc[2] : (ij[1] + 1) * MM_loc[2],
        ] = rho_xyz

    logging.info(f"Pencil {ij} saved WDM rho")

if not nocdm:
    if "density" not in f:
        grp = f.create_group("density")
    grp = f["density"]
    dset = create_ds(grp, "cdm", (MM, MM, MM))

    if len(filaments_in_box) > 0:
        rho_xyz[:] = 0.0
    for idx, vs in filaments_in_box.items():
        R_cut = rhos_bg[idx].R99 if nofdm and nowdm else wavefunctions_params[idx].R_fit
        for v in vs:
            filament_spine = finite_straight_filament_spine_params(
                ray_params=init_ray_params(
                    origin=spine_gas_params[idx].ray_params.origin + v,
                    direction=spine_gas_params[idx].ray_params.direction,
                ),
                length=spine_gas_params[idx].length,
                orientation=spine_gas_params[idx].orientation,
            )
            rho_xyz[:] += np.asarray(
                eval_cdm_cylinder_density(xyz, filament_spine, rhos_bg[idx], R_cut)
            ).reshape(rho_xyz.shape)
            logging.info(
                f"Pencil {ij} constructed CDM rho with cylinder {idx} from {v/jnp.array([L,L,L])}"
            )

    with dset.collective:
        dset[
            :,
            ij[0] * MM_loc[1] : (ij[0] + 1) * MM_loc[1],
            ij[1] * MM_loc[2] : (ij[1] + 1) * MM_loc[2],
        ] = rho_xyz
    logging.info(f"Pencil {ij} saved CDM rho")

comm.Barrier()
if rank == 0:
    logging.info(f"Saved in {filename}")
f.close()
