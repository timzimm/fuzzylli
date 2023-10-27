# fuzzylli: Reconstruction for Fuzzy Dark Matter Filaments
<div align="center">
<img
src="https://github.com/timzimm/fuzzylli/blob/16ed0c6c6734a110f25903b610e091ab10298fb0/images/filament_crosssections.png" alt="logo"></img>
</div>

## What is fuzzylli
fuzzylli is a tool to postprocess fuzzy dark matter (FDM) interference fringes
onto cosmic filaments. It does so by both adapting and unifying existing approaches for halo
interference reconstruction. The result is an effective, and efficient surrogate
model for the FDM wave function in cylindrical symmetry. We refer to [our
paper](#citing-fuzzylli)
for an in depth exposition of fuzzylli's underlying assumptions, physics and application
to cosmology.

fuzzylli is built on jax such that compuation of observables involving the FDM wavefunction
(including its derivatives via `jax.grad`) are trivial (thanks to `jax.vmap`) and 
efficient (thanks to `jax.jit`) to implement.

The name is inspired by the Fussilli noodle (wavy long pasta). We gloss over the
fact that the final filaments look more like Rigatoni at the moment

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
If you want to be able to populate three dimensional boxes with a cylinder gas,
replace the last command with:
```console
$ pip install -e ".[3D]"
```
This will require `mpi4py` as well as a parallel `h5py` build.
Make sure these exist in your envirnoment.
The latter might need some manual compilation labor. We refer to the [h5py
Documentation](https://docs.h5py.org/en/latest/mpi.html) for more information.

## Example
To recreate the "eyes of sauron" plot at the top open ... which showcases the
main functionalty of fuzzylli.

## Citing fuzzylli
If you use fuzzylli for any publication we kindly ask you to cite
TODO

<div align="center">
<img
src="https://github.com/timzimm/fuzzylli/blob/820bc2c270556f4b9208f09224c53764eb6651d1/images/eu_acknowledgement_compsci_3.png" alt="logo"></img>
</div>
