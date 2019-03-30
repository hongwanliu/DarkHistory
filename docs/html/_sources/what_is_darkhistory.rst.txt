What is DarkHistory?
=======================================

DarkHistory is a Python code package that calculates the global temperature and ionization history of the universe given an exotic source of energy injection, such as dark matter annihilation or decay. In particular, it makes the temperature constraint calculations significantly more streamlined, self-consistent, and accurate. It has a modular structure, allowing users to easily adjust individual inputs to the calculation -- e.g. by changing the reionization model, or the spectrum of particles produced by dark matter decay. Compared to past codes developed for such analyses [1]_ it has a number of important new features:

1. The first fully self-consistent treatment of exotic energy injection. Exotic energy injections can modify the evolution of the IGM temperature :math:`T_\mathrm{IGM}` and free electron fraction :math:`x_e`\ , and previously this modification has been treated perturbatively, assuming the backreaction effect on the cooling of injected particles is negligible. This assumption can break down toward the end of the cosmic dark ages for models that are not yet excluded [2]_. DarkHistory solves simultaneously for the temperature and ionization evolution and the cooling of the injected particles, avoiding this assumption.


2.  A self-contained treatment of astrophysical sources of heating and reionization, allowing the study of the interplay between exotic and conventional sources of energy injection. 


3. A large speed-up factor for computation of the full cooling cascade for high-energy injected particles (compared to the code employed in e.g. [2]_), via pre-computation of the relevant transfer functions as a function of particle energy, redshift and ionization level.

4. Support for treating the effects of helium ionization and recombination, including exotic energy injections.


5. A new and more correct treatment of inverse Compton scattering (ICS) for mildly relativistic and non-relativistic electrons; previous work in the literature has relied on approximate rates which are not always accurate.


Due to these improvements, DarkHistory allows for rapid scans over many different prescriptions for reionization, either in the form of photoheating and photoionization rates, or a hard-coded background evolution for :math:`x_e`\ . The epoch of reionization is currently rather poorly constrained, making it important to understand the observational signatures of different scenarios, and the degree to which exotic energy injections might be separable from uncertainties in the reionization model. Previous attempts to model the effects of DM annihilation and decay into the reionization epoch have typically either assumed a fixed ionization history [3]_ -- requiring a slow re-computation of the cooling cascade if that history is changed [2]_ -- or performed a somewhat ad-hoc analytic approximation for the effect of a modified ionization fraction on the cooling of high-energy particles [4]_.

