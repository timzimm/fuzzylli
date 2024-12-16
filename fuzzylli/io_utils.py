import math
import pickle
import logging
import os
import numpy as np
import jax.numpy as jnp
import h5py
import jax
import ruamel.yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_parameters_from_config(path_to_file):
    yaml = ruamel.yaml.YAML(typ="safe", pure=True)
    with open(path_to_file, "r") as file:
        parameters = yaml.load(file)

    check_parameters(parameters)
    return parameters


def check_parameters(params):
    if not (
        len(params["background_density"]["beta"])
        == len(params["background_density"]["sigma"])
        == len(params["background_density"]["r0"])
        == len(params["phasespace_distribution"]["epochs"])
        == len(params["phasespace_distribution"]["R_min"])
        == len(params["eigenstate_library"]["N"])
        == len(params["filament_ensemble"]["mass"])
        == len(params["filament_ensemble"]["origin"])
        == len(params["filament_ensemble"]["direction"])
        == len(params["filament_ensemble"]["orientation"])
    ):
        raise ValueError()
    if params["cosmology"]["z"] != 4.0:
        raise ValueError()


def create_ds(h5file, name, shape):
    spaceid = h5py.h5s.create_simple(shape)
    plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    plist.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
    name = bytes(name, encoding="utf-8")
    datasetid = h5py.h5d.create(h5file.id, name, h5py.h5t.NATIVE_DOUBLE, spaceid, plist)
    dset = h5py.Dataset(datasetid)
    return dset


def dictify(namedtuple):
    dic = namedtuple._asdict()
    for key, item in dic.items():
        if hasattr(item, "_asdict"):
            dict_item = dictify(item)
            dic[key] = dict_item
    return dic


def save_dict_to_group(h5file, path, dic):
    for key, item in dic.items():
        if isinstance(item, (jax.Array, np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            save_dict_to_group(h5file, path + key + "/", item)
        else:
            raise ValueError("Cannot save %s type" % type(item))


def load_dict_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = jnp.asarray(item)
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = load_dict_from_group(h5file, path + key + "/")
    return ans


def load_model(cache_dir, name):
    logger.info(f"Loading {name}...")
    with open(f"{cache_dir}/{name}", mode="rb") as file:
        model = pickle.load(file)
    return model


def save_model(overwrite, cache_dir, name, model):
    if not os.path.exists(f"{cache_dir}/{name}") or overwrite:
        logger.info(f"Saving {name}...")
        with open(f"{cache_dir}/{name}", mode="wb") as file:
            pickle.dump(model, file)


def compute_and_save_model(cache_dir, name_of, compute, **kwargs):
    compute_args = kwargs["for_compute"]
    name_args = kwargs["for_name"]
    model = compute(*compute_args)
    name = name_of(*name_args)
    save_model(True, cache_dir, name, model)
    return name, model


def load_or_compute_model(load_if_available, cache_dir, name_of, compute, **kwargs):
    compute_args = kwargs["for_compute"]
    name_args = kwargs["for_name"]
    name = name_of(*name_args)
    if os.path.exists(f"{cache_dir}/{name}") and load_if_available:
        return name, load_model(cache_dir, name)
    else:
        logger.info("No model found/Recompute enforced. Compute it...")
        return name, compute(*compute_args)


# TODO: This is horrible. Make it better
def load_wavefunctions_params_from_hdf5(file):
    N = len(file["cylinders"])
    wavefunctions_params = []
    for i in range(N):
        raw_dict = load_dict_from_group(file, f"/cylinders/{i}/wavefunction_params/")
        R_j_params = raw_dict["eigenstate_library"]["R_j_params"]
        wavefunctions_params.append(
            wavefunction_params(
                total_mass=raw_dict["total_mass"],
                R_fit=raw_dict["R_fit"],
                a_j=raw_dict["a_j"],
                phase_j=raw_dict["phase_j"],
                eigenstate_library=eigenstate_library(
                    E_j=raw_dict["eigenstate_library"]["E_j"],
                    l_of_j=raw_dict["eigenstate_library"]["l_of_j"],
                    n_of_j=raw_dict["eigenstate_library"]["n_of_j"],
                    R_j_params=interpolation_params(
                        a=R_j_params["a"],
                        dx=R_j_params["dx"],
                        f=R_j_params["f"],
                        lb=R_j_params["lb"],
                        ub=R_j_params["ub"],
                    ),
                ),
            )
        )
    return wavefunctions_params


def hash_to_int64(digested_hash):
    num = int(digested_hash, 16)
    num = num // 10 ** (int(math.log(num, 10)) - 18 + 1)
    return jnp.int64(num)
