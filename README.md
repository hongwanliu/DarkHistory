# [DarkHistory v1.1.1](https://github.com/hongwanliu/DarkHistory/releases/tag/v1.1.1) for [DM21cm](https://github.com/yitiansun/DM21cm)

The branch of DarkHistory used in [DM21cm](https://github.com/yitiansun/DM21cm), a semi-numerical simulation of inhomogemeous dark matter energy injection based on DarkHistory and [21cmFAST](https://github.com/joshwfoster/21cmFAST). The data files used for this release is the same as v1.1 (see below). DM21cm is described in [arXiv:2312.11608](https://arxiv.org/abs/2312.11608).

# [DarkHistory v2.0](https://github.com/hongwanliu/DarkHistory/releases/tag/v2.0.0), with improved treatment of low energy electrons and spectral distortions

The branch containing the upgraded treatment for low energy electrons and spectral distortions can be found [here](https://github.com/hongwanliu/DarkHistory/tree/lowengelec_upgrade). In additional to the data files needed for v1.0, this upgrade requires [additional data files](https://doi.org/10.5281/zenodo.7651517).

The upgrades are described in a paper available at [arXiv:2303.07366](https://arxiv.org/abs/2303.07366), and examples of applications are given in [arXiv:2303.07370](https://arxiv.org/abs/2303.07370). Please cite these as well as [arXiv:1904.09296](https://arxiv.org/abs/1904.09296) if you use this version of DarkHistory in a scientific publication.

# [DarkHistory v1.1](https://github.com/hongwanliu/DarkHistory/releases/tag/v1.1.0) with Neural Network transfer functions

<!-- [<img src="https://travis-ci.org/hongwanliu/DarkHistory.svg?branch=master">](https://travis-ci.org/hongwanliu/DarkHistory)
[<img src="https://readthedocs.org/projects/darkhistory/badge/?version=master">](https://readthedocs.org/projects/darkhistory/) -->

Added Neural Network transfer functions to optionally replace large tabulated transfer functions. Requires [Tensorflow 2.0](https://www.tensorflow.org/install) in addition to v1.0 dependencies, and a [compact dataset](https://doi.org/10.5281/zenodo.6819281) to use the Neural Network transfer functions. (To upgrade from v1.0, one can simply add the compact dataset to the existing data directory). To use the tabulated transfer functions, a [full dataset](https://doi.org/10.5281/zenodo.6819310) is required. (This version of DarkHistory also works with v1.0 dataset with setting `use_v1_0_data=True` in config.py.)

The update is described in a paper available at [arXiv:2207.06425](https://arxiv.org/abs/2207.06425). Please cite this paper as well as [arXiv:1904.09296](https://arxiv.org/abs/1904.09296) if you use this version of DarkHistory in a scientific publication. The release for this version can be found [here](https://github.com/hongwanliu/DarkHistory/releases/tag/v1.1.0). For more information, please visit our webpage [here](https://darkhistory.readthedocs.io).

# [DarkHistory v1.0](https://github.com/hongwanliu/DarkHistory/releases/tag/v1.0.0)

[<img src="https://travis-ci.org/hongwanliu/DarkHistory.svg?branch=development">](https://travis-ci.org/hongwanliu/DarkHistory)
[<img src="https://readthedocs.org/projects/darkhistory/badge/?version=development">](https://readthedocs.org/projects/darkhistory/)

DarkHistory is a Python code package that calculates the global temperature and ionization history of the universe given an exotic source of energy injection, such as dark matter annihilation or decay. DarkHistory is described in a paper available at [arXiv:1904.09296](https://arxiv.org/abs/1904.09296). Please cite this paper if you use DarkHistory in a scientific publication. The data files for required for this version can be found [here](https://doi.org/10.7910/DVN/DUOUWA). The release for this version can be found [here](https://github.com/hongwanliu/DarkHistory/releases/tag/v1.0.0). For more information, please visit our webpage [here](https://darkhistory.readthedocs.io).
