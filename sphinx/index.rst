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

*Please check this page frequently for new updates, as the code is still in its infancy!*

*2024-08-12* -- DarkHistory is now compatible with python >= 3.10. The following packages are required, with recommended version numbers. See ``requirements.txt`` or ``pyproject.toml`` for more information.

* astropy>=5.3
* h5py
* matplotlib
* numpy>=1.25.2
* numpydoc
* scipy>=1.11.2, <=1.13.1
* tqdm

We have also converted all data files to more stable and versatile formats (HDF5, JSON, and plain text), which you can download `here <https://doi.org/10.5281/zenodo.13259509>`_.

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

There are two parts to DarkHistory: the code stored as a `GitHub repository <https://github.com/hongwanliu/DarkHistory/>`_, and the data files. 

Code
-------------------

The code can be downloaded as a ``.zip`` file from the `GitHub repository <https://github.com/hongwanliu/DarkHistory/>`_ page, or a local version of the repository can be cloned using ``git``. 

Data Files
--------------------

The data files that are necessary for DarkHistory to perform its calculations can be found on `Dataverse <https://doi.org/10.7910/DVN/DUOUWA>`_. 

Using DarkHistory Without Backreaction or Spectral Information
---------------------------------------------------------------

Some of the data files are several gigabytes in size, and are required when taking into account backreaction, so that the actual spectrum of particles in the bath are important. Users can choose not to download the following files:

* The propagating photon, low-energy photon and low-energy electron transfer functions, *highengphot_tf_interp.raw.zip*, *lowengphot_tf_interp.raw.zip*, *lowengelec_tf_interp.raw.zip*;
* The high-energy deposition transfer function, *highengdep_interp.raw*; and
* Tabulated inverse Compton scattering transfer functions, *ics_thomson_ref_tf.raw*, *ics_thomson_ref_tf.raw* and *engloss_ref_tf.raw*. 

Without these files, users *cannot* use the function :func:`main.evolve`, but can evaluate temperature and ionization histories using :func:`.tla.get_history` using a tabulated set of :math:`f_c(z)` data. DarkHistory comes with its own tabulated set of :math:`f_c(z)` data for dark matter annihilation and decay into :math:`e^+e^-` and :math:`\gamma \gamma`.

Getting Started
=======================================

DarkHistory uses python >= 3.10, and uses the following packages:

* astropy>=5.3
* h5py
* matplotlib
* numpy>=1.25.2
* numpydoc
* scipy>=1.11.2, <=1.13.1
* tqdm

DarkHistory has been tested with the package versions shown above, and using different versions may result in unexpected behavior. We recommend using `Conda <https://conda.io/en/latest/>`_, which helps manage libraries, dependencies and environments. To install all of these packages with the recommended versions, users can simply do

.. sourcecode:: bash

    $ conda config --add channels conda-forge
    $ conda install --file requirements.txt

from the ``DarkHistory/`` directory. The user can also choose to install packages individually; the unofficial Jupyter notebook extensions must be installed from the ``conda-forge`` channel within Conda.

Alternatively, if the user would like to use ``pip`` instead, installing all of the relevant packages can be done by the following command:

.. sourcecode:: bash

    $ pip install -r requirements.txt

For the ``master`` branch, the user can directly install the code along with requirements by running the following command from the ``DarkHistory/`` directory:

.. sourcecode:: bash

    $ pip install .

After installation, users can specify the location of the downloaded data files, so that DarkHistory knows where they're stored. This is done by inserting the following line into ``config.py`` (found in the ``DarkHistory/`` directory): 

.. sourcecode:: python

    # Location of all data files. CHANGE THIS FOR DARKHISTORY TO ALWAYS
    # LOOK FOR THESE DATA FILES HERE. 

    data_path = '/foo/bar'

where ``/foo/bar`` is the directory in which the data files are stored. 

Within the ``examples/`` directory of the repository are several Jupyter notebooks aimed at helping the user learn how to use DarkHistory. To begin, navigate to the ``DarkHistory/`` directory and ensure that you are in a Python 3 environment. If you are using Conda, please see the `Conda documentation <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ for instructions on how to create a Python 3 environment. 

Next, execute the following line:

.. sourcecode:: bash

    $ jupyter notebook

This should open a window showing a list of files and directories in the ``DarkHistory/`` directory. The user should be able to run all of the examples in ``DarkHistory/examples/`` to learn how to use this code. 

Getting Help
=======================================

For questions regarding physics and using the code, please contact the authors directly. For suspected bugs or issues with the code, please open a new issue on GitHub. 

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
