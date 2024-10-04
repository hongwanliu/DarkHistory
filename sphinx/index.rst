.. DarkHistory documentation master file, created by
   sphinx-quickstart on Thu Sep  7 16:43:07 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DarkHistory
#######################################

.. contents:: Table of Contents
    :depth: 2

Welcome to the DarkHistory page! Here, users will find installation instructions and detailed documentation of the code. DarkHistory is described in `arXiv:1904.09296 <https://arxiv.org/abs/1904.09296>`_ (referred to as Paper I in the examples). Please cite this paper if you use DarkHistory in a scientific publication.

Announcements
======================================

*Please check this page frequently for new updates!*

*2024-10-04* -- DarkHistory is now compatible with python >= 3.10, <= 3.12. The following packages are required, with recommended version numbers. See ``requirements.txt`` or ``pyproject.toml`` for more information.

* astropy>=5.3
* h5py
* matplotlib
* numpy>=1.25.2
* numpydoc
* scipy>=1.11.2, <=1.13.1
* tqdm

*2024-08-12* We have converted all data files to more stable and versatile formats (HDF5, JSON, and plain text), which you can download `here <https://doi.org/10.5281/zenodo.13259509>`_.

*2019-04-22* -- First release of DarkHistory (v1.0.0). 

What is DarkHistory?
=======================================

DarkHistory is a Python code package that calculates the global temperature and ionization history of the universe given an exotic source of energy injection, such as dark matter annihilation or decay. In particular, it makes the temperature constraint calculations significantly more streamlined, self-consistent, and accurate. It has a modular structure, allowing users to easily adjust individual inputs to the calculation -- e.g. by changing the reionization model, or the spectrum of particles produced by dark matter decay. Compared to past codes developed for such analyses [1]_ it has a number of important new features:

1. *The first fully self-consistent treatment of exotic energy injection*. Exotic energy injections can modify the evolution of the IGM temperature :math:`T_\mathrm{IGM}` and free electron fraction :math:`x_e`\ , and previously this modification has been treated perturbatively, assuming the backreaction effect on the cooling of injected particles is negligible. This assumption can break down toward the end of the cosmic dark ages for models that are not yet excluded [2]_. DarkHistory solves simultaneously for the temperature and ionization evolution and the cooling of the injected particles, avoiding this assumption.

2.  *A self-contained treatment of astrophysical sources of heating and reionization*, allowing the study of the interplay between exotic and conventional sources of energy injection. 

3. *A large speed-up factor for computation* of the full cooling cascade for high-energy injected particles (compared to the code employed in e.g. [2]_), via pre-computation of the relevant transfer functions as a function of particle energy, redshift and ionization level.

4. *Support for treating the effects of helium ionization and recombination*, including exotic energy injections.

5. *A new and more correct treatment of inverse Compton scattering (ICS)* for mildly relativistic and non-relativistic electrons; previous work in the literature has relied on approximate rates which are not always accurate.

Due to these improvements, DarkHistory allows for rapid scans over many different prescriptions for reionization, either in the form of photoheating and photoionization rates, or a hard-coded background evolution for :math:`x_e`\ . The epoch of reionization is currently rather poorly constrained, making it important to understand the observational signatures of different scenarios, and the degree to which exotic energy injections might be separable from uncertainties in the reionization model. Previous attempts to model the effects of DM annihilation and decay into the reionization epoch have typically either assumed a fixed ionization history [3]_ -- requiring a slow re-computation of the cooling cascade if that history is changed [2]_ -- or made an approximation for the effect of a modified ionization fraction on the cooling of high-energy particles [3]_ [4]_ [5]_ [6]_ [7]_.

Installation
=======================================

Updated 2024/10/04

Clone the `GitHub repository <https://github.com/hongwanliu/DarkHistory/>`_ using ``git``, for example:

.. sourcecode:: bash

    $ git clone git@github.com:hongwanliu/DarkHistory.git

Check out the specific version of the code you want to use, for example:

.. sourcecode:: bash

    $ git checkout lowengelec_upgrade

For legacy versions of the code, check out published versions:

.. sourcecode:: bash

    $ git checkout v1.0.0

We recommend creating a new virtual environment for DarkHistory and using ``pip`` to install the required packages. To create a new virtual environment using for example ``conda``, run the following command:

.. sourcecode:: bash

    $ conda create -n darkhistory python=3.12 pip

From the ``DarkHistory/`` directory, install the DarkHistory and required packages by:

.. sourcecode:: bash

    $ pip install .

Download data files required to run DarkHistory `here <https://doi.org/10.5281/zenodo.13259509>`_.

Let DarkHistory know the data location by setting variable ``data_path_default`` in ``darkhistory/config.py`` or the environment variable ``DH_DATA_DIR`` to the directory containing data files. Now you should be able to run DarkHistory. You can familiarize yourself with DarkHistory using notebooks in ``examples/``.


Getting Help
=======================================

For questions regarding physics and using the code, please contact the authors directly (see GitHub). For suspected bugs or issues with the code, please open a new issue on GitHub. 

Documentation
=======================================

Start here for the complete documentation of the code. 

.. toctree::
   :maxdepth: 2

   modules

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. rubric:: Footnotes

.. [1]  P. Stöcker, M. Krämer, J. Lesgourgues, and V. Poulin, JCAP **1803**, 018 (2018), 1801.01871.
.. [2] H. Liu, T. R. Slatyer, and J. Zavala, Phys. Rev. **D94**, 063507 (2016), 1604.02457.
.. [3] L. Lopez-Honorez, O. Mena, A. Moliné, S. Palomares-Ruiz, and A. C. Vincent, JCAP 1608, 004 (2016), 1603.06795.
.. [4] R. Diamanti, L. Lopez-Honorez, O. Mena, S. Palomares-Ruiz, and A. C. Vincent, JCAP **1402**, 017 (2014), 1308.2578.
.. [5] L. Lopez-Honorez, O. Mena, S. Palomares-Ruiz, and A. C. Vincent, JCAP **1307**, 046 (2013), 1303.5094. 
.. [6] V. Poulin, P. D. Serpico and J. Lesgourgues, JCAP **1512**, 041 (2015), 1508.01370. 
.. [7] V. Poulin, J. Lesgourgues, and P. D. Serpico, JCAP **1703**, 043 (2017), 1610.10051.
