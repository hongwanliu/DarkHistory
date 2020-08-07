.. DarkHistory documentation master file, created by
   sphinx-quickstart on Thu Sep  7 16:43:07 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DarkHistory
#######################################

Welcome to the DarkHistory page! Here, users will find installation instructions and detailed documentation of the code. 

.. contents:: Table of Contents
    :depth: 2

What is DarkHistory?
=======================================

DarkHistory is a Python code package that calculates the global temperature and ionization history of the universe given an exotic source of energy injection, such as dark matter annihilation or decay. In particular, it makes the temperature constraint calculations significantly more streamlined, self-consistent, and accurate. It has a modular structure, allowing users to easily adjust individual inputs to the calculation -- e.g. by changing the reionization model, or the spectrum of particles produced by dark matter decay. Compared to past codes developed for such analyses [1]_ it has a number of important new features:

1. *The first fully self-consistent treatment of exotic energy injection*. Exotic energy injections can modify the evolution of the IGM temperature :math:`T_\mathrm{IGM}` and free electron fraction :math:`x_e`\ , and previously this modification has been treated perturbatively, assuming the backreaction effect on the cooling of injected particles is negligible. This assumption can break down toward the end of the cosmic dark ages for models that are not yet excluded [2]_. DarkHistory solves simultaneously for the temperature and ionization evolution and the cooling of the injected particles, avoiding this assumption.

2.  *A self-contained treatment of astrophysical sources of heating and reionization*, allowing the study of the interplay between exotic and conventional sources of energy injection. 

3. *A large speed-up factor for computation* of the full cooling cascade for high-energy injected particles (compared to the code employed in e.g. [2]_), via pre-computation of the relevant transfer functions as a function of particle energy, redshift and ionization level.

4. *Support for treating the effects of helium ionization and recombination*, including exotic energy injections.

5. *A new and more correct treatment of inverse Compton scattering (ICS)* for mildly relativistic and non-relativistic electrons; previous work in the literature has relied on approximate rates which are not always accurate.

Due to these improvements, DarkHistory allows for rapid scans over many different prescriptions for reionization, either in the form of photoheating and photoionization rates, or a hard-coded background evolution for :math:`x_e`\ . The epoch of reionization is currently rather poorly constrained, making it important to understand the observational signatures of different scenarios, and the degree to which exotic energy injections might be separable from uncertainties in the reionization model. Previous attempts to model the effects of DM annihilation and decay into the reionization epoch have typically either assumed a fixed ionization history [3]_ -- requiring a slow re-computation of the cooling cascade if that history is changed [2]_ -- or performed a somewhat ad-hoc analytic approximation for the effect of a modified ionization fraction on the cooling of high-energy particles [4]_.

Installation
=======================================

There are two parts to DarkHistory: the code stored as a `GitHub repository <https://github.com/hongwanliu/DarkHistory/>`_, and the data files. 

Code
-------------------

The code can be downloaded as a ``.zip`` file from the `GitHub repository <https://github.com/hongwanliu/DarkHistory/>`_ page, or a local version of the repository can be cloned using ``git``. 

Data Files
--------------------

The data files that are necessary for DarkHistory to perform its calculations can be found on ?????. 

Getting Started
=======================================

DarkHistory is written in Python 3, and uses the following Python packages: 

* `Numpy <http://www.numpy.org/>`_
* `Scipy <https://scipy.org/scipylib/index.html>`_
* `Matplotlib <https://matplotlib.org/>`_
* `Jupyter <https://jupyter.org/>`_
* `Unofficial Jupyter Notebook Extensions <https://jupyter-contrib-nbextensions.readthedocs.io/>`_
* `tqdm <https://github.com/tqdm/tqdm>`_

We recommend users use `Conda <https://conda.io/en/latest/>`_, which helps users manage libraries, dependencies and environments. To install all of these packages, users can simply do

.. sourcecode:: bash

    $ conda install --file requirements.txt

from the ``DarkHistory/`` directory. Alternatively, if the user would like to use pip instead, installing all of the relevant packages can be done by the following command:

.. sourcecode:: bash

    $ pip install -r requirements.txt

Within the ``examples/`` directory of the repository are several Jupyter notebooks aimed at helping the user learn how to use DarkHistory. To begin, navigate to the ``DarkHistory/`` directory and ensure that you are in a Python 3 environment. Next, execute the following line:

.. sourcecode:: bash

    $ jupyter notebook

This should open a window showing a list of files and directories in the ``DarkHistory/`` directory. 

Users should first run the Jupyter notebook labeled 'Example 0' in the ``examples/`` directory of the repository to check that all relevant packages have been downloaded, and that packages within DarkHistory can be correctly imported for use. 

After running Example 0, users can specify the location of the downloaded data files, so that DarkHistory knows where they're stored. This is done by inserting the following line into ``config.py`` (found in the ``DarkHistory/`` directory): 

.. sourcecode:: python

    # Location of all data files. CHANGE THIS FOR DARKHISTORY TO ALWAYS
    # LOOK FOR THESE DATA FILES HERE. 

    data_path = '/foo/bar'

where ``/foo/bar`` is the directory in which the data files are stored. 

With this complete, the user should be able to run the rest of the examples to learn how to use this code. 

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

.. [1] Cite past codes, CLASS etc.
.. [2] Liu, Slatyer and Zavala, 2016
.. [3] ExoClass
.. [4] Aaron Vincent and co. 