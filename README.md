# fuzzylli: Reconstruction for Fuzzy Dark Matter Filaments
<div align="center">
<img
src="https://github.com/timzimm/fuzzylli/blob/5b5b6bd7d66a69f3de60b184f99e43f75080f0c9/images/crosssections.png" alt="logo"></img>
</div>

## What is fuzzylli
fuzzylli is a tool to postprocess fuzzy dark matter (FDM) interference fringes
onto cosmic filaments. It does so by both adapting and unifying existing approaches for halo
interference reconstruction, most notably:

[1] [Yavetz et al. (2021)](https://arxiv.org/abs/2109.06125):
Construction of Wave Dark Matter Halos: Numerical Algorithm and Analytical Constraints
<br>
[2] [Lin et al. (2018)](https://arxiv.org/abs/1801.02320):
Self-consistent construction of virialized wave dark matter halos
<br>
[3] [Dalal et al. (2021)](https://arxiv.org/abs/2011.13141):
Don't cross the streams: caustics from fuzzy dark matter

The result is an effective, and efficient surrogate model for the FDM wave function in 
cylindrical symmetry. We refer to [our paper](#citing-fuzzylli)
for an in depth exposition of fuzzylli's underlying assumptions, physics and application
to cosmology.

fuzzylli is built on [jax](https://github.com/google/jax) such that compuation of observables 
involving the FDM wavefunction
(including its derivatives via `jax.grad`) are trivial (thanks to `jax.vmap`) and 
efficient (thanks to `jax.jit`) to implement.

The name is inspired by the Fussilli noodle (wavy long pasta). We gloss over the
fact that the final filaments look more like Rigatoni at the moment
<div align="center">
<img
src="https://github.com/timzimm/fuzzylli/blob/2f6bcfa22dd8e5464ef896dec482c9aafd6bd6e0/images/volume_rendering.png" alt="logo" width="600"></img>
</div>

**This is a research project. Expect bugs, report bugs, fix more bugs than you
create.**

## Installation
fuzzylli is pre-alpha and currently meant as an internal tool. No real efforts
have been put into packaging at this point. Please install in development mode
via:
```console
$ git clone https://github.com/timzimm/fuzzylli.git
$ cd fuzzylli
$ pip install -e .
```
If you want to be able to populate three dimensional boxes with a cylinder gas
(i.e. using [generate_density.py](https://github.com/timzimm/fuzzylli/blob/3c4ba7e048d663de7ae41750a81384ec8030dd77/fuzzylli/generate_density.py)),
replace the last command with:
```console
$ pip install -e ".[3D]"
```
This will require `mpi4py` as well as a parallel `h5py` build.
Make sure these exist in your envirnoment.
The latter might need some manual compilation labor. We refer to the [h5py
Documentation](https://docs.h5py.org/en/latest/mpi.html) for more information.

## Example
To recreate the "eyes of sauron" plot at the top, i.e. filament cross sections as a 
function of FDM mass,  open ... which showcases the main functionalty of fuzzylli.

## Citing fuzzylli
If you use fuzzylli for any publication we kindly ask you to cite
TODO

## Acknowledgement
<div align="center">
<img
src="https://github.com/timzimm/fuzzylli/blob/820bc2c270556f4b9208f09224c53764eb6651d1/images/eu_acknowledgement_compsci_3.png" alt="logo"></img>
</div>
