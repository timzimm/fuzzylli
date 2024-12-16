<div align="center">
<img
src="https://github.com/timzimm/fuzzylli/blob/3e972ebae28f78273f88248e488dea43f7fcc31a/images/logo.png" alt="logo" width="150"></img>
</div>

# fuzzylli: Interference Fringes for Cosmic Filaments
[**What is Fuzzylli**](#what-is-fuzzylli)
| [**Installation**](#installation)
| [**Example**](#example)
| [**Citing Fuzzylli**](#citing-fuzzylli)

## What is fuzzylli
fuzzylli is a tool to postprocess fuzzy dark matter (FDM) interference fringes
onto cosmic filaments. It does so by both adapting and unifying existing approaches for halo
interference reconstruction, most notably:

[1] [Yavetz et al. (2021)](https://arxiv.org/abs/2109.06125):
_Construction of Wave Dark Matter Halos: Numerical Algorithm and Analytical Constraints_
<br>
[2] [Lin et al. (2018)](https://arxiv.org/abs/1801.02320):
_Self-consistent construction of virialized wave dark matter halos_
<br>
[3] [Dalal et al. (2021)](https://arxiv.org/abs/2011.13141):
_Don't cross the streams: caustics from fuzzy dark matter_
<br>
[4] [Zimmermann et al. (2024)](https://arxiv.org/abs/2405.20374):
_Dwarf galaxies imply dark matter is heavier thani 2.2 zeV_

<figure>
  <img src="https://github.com/timzimm/fuzzylli/blob/0d792d8d018cb6a44108581965902cfc148f8aeb/images/comparison.png" alt="" width="750" align="center">
  <figcaption align="center">Comparison between fuzzylli and AXIREPO, a state of the
  integrator for the full-fledged Schrödinger-Poisson equation</figcaption>
</figure>
<br/><br/>

The result is a surrogate model for the FDM wave function in 
cylindrical symmetry. We refer to [our paper](#citing-fuzzylli)
for an in depth exposition of fuzzylli's underlying assumptions, physics and application
to cosmology.

fuzzylli is built on [jax](https://github.com/google/jax) such that compuation of observables 
involving the FDM wavefunction
(including its derivatives via `jax.grad`) are trivial (thanks to `jax.vmap`) and 
reasonably efficient (thanks to `jax.jit`).

<figure>
  <img src="https://github.com/timzimm/fuzzylli/blob/fca7a9bca0a1dbd60dfa59b4b30f2cdb3e930b81/images/crossection.png" alt="" width="750" align="center">
  <figcaption align="center">Filament cross sections as a function of axion/FDM mass</figcaption>
</figure>
<br/><br/>
The name is inspired by the Fussilli noodle (wavy long pasta). We gloss over the
fact that the final filaments look more like Rigatoni at the moment
:man_shrugging:.

Loosely related, fuzzylli also provides a differentiable jax implementation of
the ellipsoidal collapse model, making an end-to-end computation of mass
functions (filaments, haloes, sheets) possible without the need to invoke literature fit 
results.

**This is a research project. Expect bugs**

## Installation
fuzzylli is pre-alpha and currently meant as an internal tool. No real efforts
have been put into packaging at this point. Please install in development mode
via:
```console
$ git clone https://github.com/timzimm/fuzzylli.git
$ cd fuzzylli
$ pip install -e .
```

## Example
We refer to our [paper repository](https://github.com/timzimm/fdm_filaments) for
examples.

## Citing fuzzylli
If you use fuzzylli for any publication we kindly ask you to cite
TODO

## Acknowledgement
![eu](https://github.com/timzimm/fuzzylli/blob/fca7a9bca0a1dbd60dfa59b4b30f2cdb3e930b81/images/eu_acknowledgement_compsci_3.png#gh-light-mode-only)
![eu](https://github.com/timzimm/fuzzylli/blob/fca7a9bca0a1dbd60dfa59b4b30f2cdb3e930b81/images/eu_acknowledgement_compsci_3_white.png#gh-dark-mode-only)
