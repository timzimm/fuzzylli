import pickle
import logging
import os
import numpy as np
import jax.numpy as jnp
import h5py
import jax

from fuzzylli.wavefunction import wavefunction_params
from fuzzylli.eigenstates import eigenstate_library
from fuzzylli.interpolation_jax import interpolation_params


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
Some functions are taken from https://github.com/yt-project/yt
"""


def SIEVE_PRIMES(x):
    return x and x[:1] + SIEVE_PRIMES([n for n in x if n % x[0]])


def decompose_to_primes(max_prime):
    """Decompose number into the primes"""
    for prime in SIEVE_PRIMES(list(range(2, max_prime))):
        if prime * prime > max_prime:
            break
        while max_prime % prime == 0:
            yield prime
            max_prime //= prime
    if max_prime > 1:
        yield max_prime


def factorize_number(pieces):
    """Return array consisting of prime, its power and number of different
    decompositions in three dimensions for this prime
    """
    factors = np.array(list(decompose_to_primes(pieces)))
    temp = np.bincount(factors)
    return np.array(
        [
            (prime, temp[prime], (temp[prime] + 1) * (temp[prime] + 2) // 2)
            for prime in np.unique(factors)
        ]
    ).astype(np.int64)


def evaluate_domain_decomposition(n_d, pieces, ldom):
    """Evaluate longest to shortest edge ratio
    BEWARE: lot's of magic here"""
    eff_dim = (n_d > 1).sum()
    exp = float(eff_dim - 1) / float(eff_dim)
    ideal_bsize = eff_dim * pieces ** (1.0 / eff_dim) * np.prod(n_d) ** exp
    mask = np.where(n_d > 1)
    nd_arr = np.array(n_d, dtype=np.float64)[mask]
    bsize = int(np.sum(ldom[mask] / nd_arr * np.prod(nd_arr)))
    load_balance = float(np.prod(n_d)) / (
        float(pieces) * np.prod((n_d - 1) // ldom + 1)
    )

    # 0.25 is magic number
    quality = load_balance / (1 + 0.25 * (bsize / ideal_bsize - 1.0))
    # \todo add a factor that estimates lower cost when x-direction is
    # not chopped too much
    # \deprecated estimate these magic numbers
    quality *= 1.0 - (0.001 * ldom[0] + 0.0001 * ldom[1]) / pieces
    if np.any(ldom > n_d):
        quality = 0

    return quality


def get_pencil_size(n_d, pieces):
    """Calculate the best division of array into px*py*pz subarrays.
    The goal is to minimize the ratio of longest to shortest edge
    to minimize the amount of inter-process communication.
    """
    fac = factorize_number(pieces)
    nfactors = len(fac[:, 2])
    best = 0.0
    p_size = np.ones(2, dtype=np.int64)
    if pieces == 1:
        return p_size

    while np.all(fac[:, 2] > 0):
        ldom = np.ones(2, dtype=np.int64)
        for nfac in range(nfactors):
            i = int(np.sqrt(0.25 + 2 * (fac[nfac, 2] - 1)) - 0.5)
            k = fac[nfac, 2] - (1 + i * (i + 1) // 2)
            i = fac[nfac, 1] - i
            ldom *= fac[nfac, 0] ** np.array([i, k])

        quality = evaluate_domain_decomposition(n_d, pieces, ldom)
        if quality > best:
            best = quality
            p_size = ldom
        # search for next unique combination
        for j in range(nfactors):
            if fac[j, 2] > 1:
                fac[j, 2] -= 1
                break
            else:
                if j < nfactors - 1:
                    fac[j, 2] = int((fac[j, 1] + 1) * (fac[j, 1] + 2) / 2)
                else:
                    fac[:, 2] = 0  # no more combinations to try

    return p_size


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


def load_model(cache_dir, name_of, *name_args):
    name = name_of(*name_args).hexdigest()
    if os.path.exists(f"{cache_dir}/{name}"):
        logger.info(f"Model already computed. Load {name}...")
        with open(f"{cache_dir}/{name}", mode="rb") as file:
            return pickle.load(file)
    else:
        logger.error("No model found. Abort ...")
        exit(1)


def compute_and_save_model(cache_dir, name_of, compute, **kwargs):
    name_args = kwargs["for_name"]
    compute_args = kwargs["for_compute"]
    name = name_of(*name_args).hexdigest()
    model = compute(*compute_args)
    with open(f"{cache_dir}/{name}", mode="wb") as file:
        pickle.dump(model, file)
    return model


def load_or_compute_model(load_if_available, cache_dir, name_of, compute, **kwargs):
    name = name_of(*kwargs["for_name"]).hexdigest()
    if os.path.exists(f"{cache_dir}/{name}") and load_if_available:
        return load_model(cache_dir, name_of, *kwargs["for_name"])
    else:
        logger.info("No model found/Recompute enforced. Compute it...")
        return compute_and_save_model(cache_dir, name_of, compute, **kwargs)


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
