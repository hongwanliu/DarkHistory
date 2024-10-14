# Changelog

## [Current]

[README](README.md)

Changes:
- Custom injection API updated. Custom functions now takes `next_rs`, `dt` keyword arguments for injection with rapidly varying power. See `main.evolve` docstring.

Fixes:
- Updated dataset and some ICS code by downgrading `float128` arrays to `float64` for compatibility with M2, M3 chips. The updated dataset can be found [here]() (same repository as v1.1.2). This update does not lead to any change in result above machine precision.
- Updated scipy dependency (replaced the removed `interp2d`). Now compatible with scipy 1.14.

## [1.1.2] - 2024/10/04

- Changed cosmology parameters in *darkhistory/physics.py* to astropy.cosmology.Planck18 by default. Please choose your cosmological parameters to be consistent with your purpose.
- Package dependency is now updated in *pyproject.toml* and *requirements.txt*. Installing via `pip install .` will automatically install the required packages. Installing via `pip install darkhistory` also works, but remember to check the code's version!
- For versatility, all data files required to use DarkHistory have been converted to either HDF5, JSON, or plain text files. All active branches of DarkHistory (`master`, `lowengelec_upgrade`, and`early_halo_cooling`) have been updated to use the new set of data files. You can download the new data files at the [following link](https://doi.org/10.5281/zenodo.13259509). See below for older datasets.