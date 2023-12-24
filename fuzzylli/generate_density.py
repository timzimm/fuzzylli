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
)
from fuzzylli.eigenstates import eigenstate_library, init_eigenstate_library
from fuzzylli.ray import init_ray_params
from fuzzylli.parameters_from_ellipsoidal_collapse import (
    cylinder_length_physical_Mpc,
    sample_mass_from_powerlaw_dn_dM,
    cylinder_scale_radius_physical_Mpc,
    cylinder_sqrt_v_dispersion_physical_kms,
)
from fuzzylli.cosmology import h
from fuzzylli.df import ConstantAnisotropyDistribution


def init_wavefunction(params_rho, df_args, N, seed):
    a_fit = params_rho["scalefactor"]
    rho = SteadyStateCylinder(**params_rho)
    V = AxialSymmetricPotential(lambda R: rho(R) / a_fit, rho.R999)
    eigenstate_args = {
        "for_name": [a_fit, V, rho.R99, K],
        "for_compute": [a_fit, V, rho.R99, K],
    }
    eigenstate_lib = load_or_compute_model(
        True,
        cache_dir,
        eigenstate_library.compute_name,
        init_eigenstate_library,
        **eigenstate_args,
    )
    df = ConstantAnisotropyDistribution(
        params_rho["beta"],
        a_fit,
        V,
        rho,
        **df_args,
        R_max=rho.R999,
    )
    optimization_args = {
        "for_name": [eigenstate_lib, rho, rho.R99, seed],
        "for_compute": [eigenstate_lib, rho, df, rho.R99, seed],
    }

    wavefunction_params = load_or_compute_model(
        True,
        cache_dir,
        psi.wavefunction_params.compute_name,
        psi.init_wavefunction_params_least_square,
        **optimization_args,
    )
    return wavefunction_params


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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--nocdm", help="omit CDM density construction", action="store_true"
)
parser.add_argument(
    "--nowdm", help="omit WDM density construction", action="store_true"
)
parser.add_argument(
    "--nofdm", help="omit FDM density construction", action="store_true"
)
parser.add_argument("--save_coord", help="save coordinates", action="store_true")

parser.add_argument("save_dir", type=str, help="Directory HDF5 will be saved to")
parser.add_argument("prefix", type=str, help="filename prefix")
args = parser.parse_args()

cache_dir = f"{args.save_dir}/cache"
prefix = args.prefix

logging.basicConfig(
    level=logging.INFO,
    format="\x1b[33;20m%(asctime)s {}\x1b[0m: %(message)s".format(socket.gethostname()),
)
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Units setup
m22 = 2.0
u = set_schroedinger_units(m22)

# Box setup
MM = 512
L = 0.25 * h * u.from_Mpc / h
if rank == 0:
    logger.info(f"dx = {L * u.to_Mpc/MM} Mpc")

seed = 42
K = 2**13
z_fit = 4
a_fit = 1.0 / (1 + z_fit)
beta = 0.0
N = 8
N_true = 1
# Sampling of FMF only works at z=4
assert z_fit == 4

spine_gas_params = None
cylinders = None
ts = None
steady_state_density_params = None
df_params = None

domain = UniformHypercube(
    [1, 1, 1],
    jnp.array([[0.0, L], [0.0, L], [0.0, L]]),
)
filename = f"{args.save_dir}/{args.prefix}_m_{m22:.2f}_beta_{beta:.2f}_z_{z_fit:.2f}_L_{L*u.to_Mpc:.2f}_N_{N_true}_M_{MM}.h5"

if rank == 0:
    Mmin = 3e9  # Msun h^-1

    M_cylinder = sample_mass_from_powerlaw_dn_dM(N, Mmin, seed=seed)  # Msun h^-1
    M_cylinder = jnp.array([5e9])
    lengths = cylinder_length_physical_Mpc(M_cylinder) * u.from_Mpc / (h * a_fit)
    r0 = h**-1 * cylinder_scale_radius_physical_Mpc(M_cylinder, beta) * u.from_Mpc

    lengths = lengths[:N_true]
    r0 = r0[:N_true]
    M_cylinder = M_cylinder[:N_true]
    print(M_cylinder, lengths * u.to_Mpc)

    d = jnp.max(lengths)
    r = 10 * jnp.max(r0)

    # spine_gas_params = init_finite_straight_filament_spine_gas(
    #     N, r, d, seed + 1, domain, lengths
    # )
    # spine_gas_params = spine_gas_params[:N_true]

    spine_dir = jnp.array([1.0, 0.0, 0.0])
    spine_dir = spine_dir / np.linalg.norm(spine_dir)
    spine_gas_params = [
        finite_straight_filament_spine_params(
            ray_params=init_rays_params(
                jnp.array([0.5 * L, 0.5 * L, 0.5 * L]), spine_dir
            ),
            length=jnp.asarray([L]),
            orientation=jnp.array([0.0, 1.0, 0.0]),
        )
    ]

    logger.info(f"{len(spine_gas_params)} cylinder(s) placed")

    steady_state_density_params = []
    f = h5py.File(filename, "w")
    f["/"].attrs.create("L_Mpc_div_h", L * h * u.to_Mpc)
    f["/"].attrs.create("m22_1e-22eV", m22)
    f["/"].attrs.create("N", N_true)
    f["/"].attrs.create("M", MM)
    f["/"].attrs.create("z", z_fit)
    cylinder_grps = [f.create_group(f"cylinders/{i}") for i in range(N_true)]

    for i, M in enumerate(M_cylinder):
        steady_state_density_params.append(
            {
                "beta": 0.0,
                "scalefactor": a_fit,
                "r0": h**-1
                * cylinder_scale_radius_physical_Mpc(M, beta)
                * u.from_Mpc,
                "sigma2": (
                    cylinder_sqrt_v_dispersion_physical_kms(M, beta) * u.from_kms
                )
                ** 2,
            }
        )
        cylinder_grps[i].attrs.create("M", M)
        cylinder_grps[i].attrs.create("beta", steady_state_density_params[-1]["beta"])
        cylinder_grps[i].attrs.create(
            "r0_Mpc_div_h", h * steady_state_density_params[-1]["r0"] * u.to_Mpc
        )
        cylinder_grps[i].attrs.create(
            "scalefactor", steady_state_density_params[-1]["scalefactor"]
        )
        cylinder_grps[i].attrs.create(
            "sigma2", steady_state_density_params[-1]["sigma2"] * u.to_kms
        )
        save_dict_to_group(f, f"/cylinders/{i}/", dictify(spine_gas_params[i]))

    df_params = {"R_min": 0.1 * u.from_Kpc, "epochs": 5000}

    # ts = np.linspace(0, 400, 401) * u.from_Myr
    ts = [0.0]
    df_params = N_true * [df_params]


spine_gas_params = comm.bcast(spine_gas_params, root=0)
ts = comm.bcast(ts, root=0)
df_params = comm.bcast(df_params, root=0)
steady_state_density_params = comm.bcast(steady_state_density_params, root=0)
N_true = len(spine_gas_params)
if rank < len(steady_state_density_params):
    logger.info(
        f"r0={steady_state_density_params[rank]['r0'].item() * u.to_Kpc:.2f} kpc, "
        f'sigma={jnp.sqrt(steady_state_density_params[rank]["sigma2"].item()) * u.to_kms:.2f} km/s, '
        f"length={spine_gas_params[rank].length.item() * u.to_Mpc:.2f} Mpc"
    )

cylinder_idx = np.array_split(np.arange(N_true), size)
if not (args.nowdm and args.nofdm):
    wavefunctions_params = [
        init_wavefunction(steady_state_density_params[i], df_params[i], K, seed)
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


if not args.nocdm:
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
for idx in range(N_true):
    vs = []
    # for v in jnp.array(jnp.array([L, L, L]) * translations.cell_centers_cartesian):
    for v in [0]:
        r = (
            rhos_bg[idx].R99
            if args.nofdm and args.nowdm
            else wavefunctions_params[idx].R_fit
        )
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

if args.save_coord:
    if "coordinates" not in f:
        grp = f.create_group("coordinates")
    grp = f["coordinates"]
    dset = create_ds(grp, "radius", (MM, MM, MM))

    R_xyz = np.zeros(shape=MM_loc)
    if len(filaments_in_box) > 0:
        xyz = jax.lax.stop_gradient(domain.cell_centers_cartesian)
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
            R_xyz[:] += np.asarray(eval_r(xyz, filament_spine, None)).reshape(
                R_xyz.shape
            )
            logging.info(
                f"Pencil {ij} constructed R for cylinder {idx} from {v/jnp.array([L,L,L])}"
            )

    with dset.collective:
        dset[
            :,
            ij[0] * MM_loc[1] : (ij[0] + 1) * MM_loc[1],
            ij[1] * MM_loc[2] : (ij[1] + 1) * MM_loc[2],
        ] = R_xyz

    logging.info(f"Pencil {ij} saved WDM rho")

for i, t in enumerate(ts):
    comm.Barrier()

    rho_xyz = np.zeros(shape=MM_loc)
    if len(filaments_in_box) > 0:
        xyz = jax.lax.stop_gradient(domain.cell_centers_cartesian)

    if not args.nofdm:
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

if not args.nowdm:
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

if not args.nocdm:
    if "density" not in f:
        grp = f.create_group("density")
    grp = f["density"]
    dset = create_ds(grp, "cdm", (MM, MM, MM))
    logger.info(vs)

    if len(filaments_in_box) > 0:
        rho_xyz[:] = 0.0
    for idx, vs in filaments_in_box.items():
        R_cut = (
            rhos_bg[idx].R99
            if args.nofdm and args.nowdm
            else wavefunctions_params[idx].R_fit
        )
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
