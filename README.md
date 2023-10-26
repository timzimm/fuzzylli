# fuzzylli: Reconstruction for Fuzzy Dark Matter Filaments
<div align="center">
<img
src="https://github.com/timzimm/fuzzylli/blob/16ed0c6c6734a110f25903b610e091ab10298fb0/images/filament_crosssections.png" alt="logo"></img>
</div>

## What is fuzzylli
fuzzylli is a tool to postprocess fuzzy dark matter (FDM) interference fringes
onto cosmic filaments. It does so by both adapting and unifying existing approaches for halo
interference reconstruction. The result is an effective, and efficient surrogate
model for the FDM wave function in cylindrical symmetry. fuzzylli is built on
jax such that compuation of observables involving the FDM wavefunction
(including its derivatives via `jax.grad`) are trivial (thanks to `jax.vmap`) and 
efficient (thanks to `jax.jit`) to implement.


An in depth exposition of our model and its application can be found our
paper...
This is a research project. Expect bugs.

## Installation
fuzzylli is pre-alpha and currently meant as an internal tool.For installation
...

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
