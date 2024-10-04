# DarkHistory

<!-- [<img src="https://travis-ci.org/hongwanliu/DarkHistory.svg?branch=master">](https://travis-ci.org/hongwanliu/DarkHistory) -->
[![arXiv](https://img.shields.io/badge/arXiv-1904.09296%20-green.svg)](https://arxiv.org/abs/1904.09296)
[<img src="https://readthedocs.org/projects/darkhistory/badge/?version=master">](https://readthedocs.org/projects/darkhistory/)

DarkHistory is a Python code package that calculates the global temperature and ionization history of the universe given an exotic source of energy injection, such as dark matter annihilation or decay. DarkHistory is described in a paper available at [arXiv:1904.09296](https://arxiv.org/abs/1904.09296). Please cite this paper if you use DarkHistory in a scientific publication. For detailed information, please visit our readthedocs webpage [here](https://darkhistory.readthedocs.io).

# Installation
*Updated 2024/10/04*

- Clone this repository. Checkout the branch you would like to use. (Current active branches: `master`, `lowengelec_upgrade`, `early_halo_cooling`)
- Create a new virtual environment for DarkHistory (recommended). For example, using `conda`:
```bash
conda create -n darkhistory python=3.12 pip
conda activate darkhistory
```
- Install via `pip` in the *DarkHistory/* directory
```bash
pip install .
```
- Download [data files](https://doi.org/10.5281/zenodo.13259509) required to run DarkHistory and save to an arbitrary location.
- Inform DarkHistory of the location by setting the variable `data_path_default` in `darkhistory/config.py` or the environment variable `DH_DATA_DIR` to the directory containing data files.
- Now you should be able to run DarkHistory. Test with the example code below. You can also familiarize yourself with DarkHistory using notebooks in *examples/*.

Notes:
- 2024/10/04: Please make sure to set cosmology parameters in *darkhistory/physics.py* consistent with your purpose! The current `master` branch may have updated parameters compared to earlier versions.
- 2024/10/04: Package dependency is now updated in *pyproject.toml* and *requirements.txt*. Installing via `pip install .` will automatically install the required packages. We currently do not recommend `pip install darkhistory`, as the PyPI version may not be most up-to-date.
- 2024/08/12: For versatility, all data files required to use DarkHistory have been converted to either HDF5, JSON, or plain text files. All active branches of DarkHistory (`master`, `lowengelec_upgrade`, and`early_halo_cooling`) have been updated to use the new set of data files. You can download the new data files at the [following link](https://doi.org/10.5281/zenodo.13259509). See below for older datasets.

# Available Versions

## [DarkHistory v1.1.2](https://github.com/hongwanliu/DarkHistory) for [DM21cm](https://github.com/yitiansun/DM21cm)

The version of DarkHistory used in [DM21cm](https://github.com/yitiansun/DM21cm), a semi-numerical simulation of inhomogemeous dark matter energy injection based on DarkHistory and [21cmFAST](https://github.com/joshwfoster/21cmFAST). DM21cm is described in [arXiv:2312.11608](https://arxiv.org/abs/2312.11608). Branch: `master`.

## [DarkHistory v2.0](https://github.com/hongwanliu/DarkHistory/releases/tag/v2.0.0), with improved treatment of low energy electrons and spectral distortions

The branch containing the upgraded treatment for low energy electrons and spectral distortions can be found [here](https://github.com/hongwanliu/DarkHistory/tree/lowengelec_upgrade). In additional to the data files needed for v1.0, this upgrade requires [additional data files](https://doi.org/10.5281/zenodo.7651517).

The upgrades are described in a paper available at [arXiv:2303.07366](https://arxiv.org/abs/2303.07366), and examples of applications are given in [arXiv:2303.07370](https://arxiv.org/abs/2303.07370). Please cite these as well as [arXiv:1904.09296](https://arxiv.org/abs/1904.09296) if you use this version of DarkHistory in a scientific publication. Branch: `lowengelec_upgrade`.

## [DarkHistory v1.1](https://github.com/hongwanliu/DarkHistory/releases/tag/v1.1.0) with Neural Network transfer functions

Added Neural Network transfer functions to optionally replace large tabulated transfer functions. Requires [Tensorflow 2.0](https://www.tensorflow.org/install) in addition to v1.0 dependencies, and a [compact dataset](https://doi.org/10.5281/zenodo.6819281) to use the Neural Network transfer functions. (To upgrade from v1.0, one can simply add the compact dataset to the existing data directory). To use the tabulated transfer functions, a [full dataset](https://doi.org/10.5281/zenodo.6819310) is required. (This version of DarkHistory also works with v1.0 dataset with setting `use_v1_0_data=True` in config.py.)

The update is described in a paper available at [arXiv:2207.06425](https://arxiv.org/abs/2207.06425). Please cite this paper as well as [arXiv:1904.09296](https://arxiv.org/abs/1904.09296) if you use this version of DarkHistory in a scientific publication. The release for this version can be found [here](https://github.com/hongwanliu/DarkHistory/releases/tag/v1.1.0).

## [DarkHistory v1.0](https://github.com/hongwanliu/DarkHistory/releases/tag/v1.0.0)

First release of DarkHistory. DarkHistory v1.0 is described in a paper available at [arXiv:1904.09296](https://arxiv.org/abs/1904.09296). Please cite this paper if you use DarkHistory in a scientific publication. The data files for required for this version can be found [here](https://doi.org/10.7910/DVN/DUOUWA). The release for this version can be found [here](https://github.com/hongwanliu/DarkHistory/releases/tag/v1.0.0). For more information, please visit our webpage [here](https://darkhistory.readthedocs.io).

# Example usage

```python
from darkhistory.main import evolve

solution = evolve(
    DM_process = 'decay', # 'decay' or 'swave'
    mDM = 1e8,            # [eV]
    lifetime = 3e25,      # [s]
    primary='elec_delta', # primary decay channel
    start_rs = 3000,      # 1+z
    coarsen_factor = 12,  # log(1+z) would change by 0.001 * coarsen_factor for next step
    backreaction = True,  # Enable injection backreaction on matter temperature and ionization levels.
    helium_TLA = True,    # Enable Helium Three Level Atom (TLA).
    reion_switch = True,  # Enable a pre-defined reionization energy injection.
)

solution.keys() # 'rs', 'x', 'Tm', 'highengphot', 'lowengphot', 'lowengelec', 'f'
```
