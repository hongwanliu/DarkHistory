# Changelog

## [Current]

[README](README.md)

Changes:
- Custom injection now passes tracked state (Tm, xHII, xHeII, photon spectrum) to injection functions. See [this example](examples/v1.1/Ex2_State_Dependent_Custom_Injection.ipynb).

## [1.1.2] - 2025/02/23

Changes:
- Custom injection API updated. Custom functions now takes `next_rs`, `dt` keyword arguments for injection with rapidly varying power. See `main.evolve` docstring.

Fixes:
- Downgraded `float128` arrays to `float64` in some portion of ICS code and dataset for compatibility with M2, M3 chips. The updated dataset can be found [here](https://doi.org/10.5281/zenodo.13931543) (same repository as v1.1.2). This update does not lead to any change in results above machine precision.
- Updated scipy dependency (replaced the removed `interp2d`). Now compatible with scipy 1.14.

## 2024/10/04

- Changed cosmology parameters in *darkhistory/physics.py* to astropy.cosmology.Planck18 by default. Please choose your cosmological parameters to be consistent with your purpose.
- Package dependency is now updated in *pyproject.toml* and *requirements.txt*. Installing via `pip install .` will automatically install the required packages. Installing via `pip install darkhistory` also works, but remember to check the code's version!
- For versatility, all data files required to use DarkHistory have been converted to either HDF5, JSON, or plain text files. All active branches of DarkHistory (`master`, `lowengelec_upgrade`, and`early_halo_cooling`) have been updated to use the new set of data files. You can download the new data files at the [following link](https://doi.org/10.5281/zenodo.13259509). See below for older datasets.