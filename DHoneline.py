import os
import sys
import argparse
import numpy as np
from main_CLASS import evolve_for_CLASS

### Use argparse to read in all the parameters
parser = argparse.ArgumentParser()

# Save settings
parser.add_argument("save_dir", help="directory where outputs are saved to")
parser.add_argument("file_name_str", help="identifier appended to output file names")
parser.add_argument("--save_DH", help="if set, saves output of DarkHistory", type=bool, default=False) #action='store_true')

# Specify injections from decays/annihilations with monochromatic injection spectra
parser.add_argument("--DM_process", help="specifies decays or annihilations. Should be one of {'decay', 'swave', 'pwave'}")
parser.add_argument("--primary", help="Primary channel of annihilation/decay. See :func:`.get_pppc_spec`for complete list. Use 'elec_delta' or 'phot_delta' for delta function injections of a pair of photons/an electron-positron pair.")
parser.add_argument("--mDM", help="Dark matter mass in [eV]", type=float)
parser.add_argument("--sigmav", help="Thermally averaged annihilation cross section in [cm^3 / s]", type=float)
parser.add_argument("--lifetime", help="Decay lifetime in [s]", type=float)

# # More general options for injection spectra and rate functions. Not yet implemented
# parser.add_argument("--in_spec_elec", help="injected spectrum of electrons", type=)
# parser.add_argument("--in_spec_phot", help="injected spectrum of photons", type=)
# parser.add_argument("--rate_func_N", help="number of injections per volume per time in units of [cm^-3 s^-1]", type=)
# parser.add_argument("--rate_func_eng", help="energy injected per volume per time in units of [eV cm^-3 s^-1]", type=)

# Key redshifts
parser.add_argument("--start_rs", help="Starting redshift for evolution. Default is 3000.", type=float, default=3000)
parser.add_argument("--high_rs", help="Threshold redshift for dealing with stiff ODE", type=float, default=np.inf)
parser.add_argument("--end_rs", help="Final redshift to evolve to. Default is 4.", type=float, default=4)

# Cosmology choices
parser.add_argument("--struct_boost", help="Structure formation boost factor. Currently implemented models are {'einasto_subs', 'einasto_no_subs', 'NFW_subs', 'NFW_no_subs', 'erfc', 'pwave_NFW_no_subs'}, see phys.struct_boost_func for details.")
parser.add_argument("--helium_TLA", help="If True, the TLA is solved with helium. Default is False.", type=bool, default=False) #action='store_true')

# Reionization
parser.add_argument("--reion_switch", help="Reionization model included if True, default is False", type=bool, default=False) #action='store_true')
parser.add_argument("--reion_rs", help="Redshift at which reionization effects turn on", type=float)
parser.add_argument("--reion_method", help="Reionization model, options are {'Puchwein', 'early', 'middle', 'late'}", default='Puchwein')
parser.add_argument("--heat_switch", help="If True, includes photoheating during reionization.", type=bool, default=False) #ction='store_true')
# parser.add_argument("--photoion_rate_func", help="", type=)
# parser.add_argument("--photoheat_rate_func", help="", type=)
# parser.add_argument("--xe_reion_func", help="", type=)
parser.add_argument("--DeltaT", help="For fixed reionization models, constant of proportionality for photoheating. See arXiv:2008.01084.", type=float)
parser.add_argument("--alpha_bk", help="Post-reionization heating power law. See arXiv:2008.01084.", type=float)

# Energy deposition calculation
parser.add_argument("--compute_fs_method", help="Method for evaluating helium ionization, should be one of {'no_He', 'He_recomb', 'He', 'HeII'}. See DarkHistory function main for details.", default='no_He')
parser.add_argument("--elec_method", help="Method for evaluation electron energy deposition. Should be one of {'new', 'old', 'eff'}. See DarkHistory function main for details.", default='new')

# Atomic physics and spectral distortions
parser.add_argument("--distort", help="If True, calculate spectral distortions. Default is False.", type=bool, default=False) #action='store_true')
parser.add_argument("--fudge", help="Value of Recfast fudge factor. Default is 1.125.", type=float, default=1.125)
parser.add_argument("--nmax", help="If distort is True, sets the maximum H principal quantum number that the MLA tracks. Default is 10.", type=int, default=10)
parser.add_argument("--fexc_switch", help="If True, include the source term b_DM to the MLA steady-state equation. Default is True.", type=bool, default='True')
parser.add_argument("--reprocess_distortion", help="If True, set Delta_f != 0, accounting for distortion photons from earlier redshifts to be absorbed or stimulate emission, i.e. be reprocessed. Default is True.", type=bool, default='True')
parser.add_argument("--simple_2s1s", help="If set, fixes the decay rate to 8.22 s^-1.", type=bool, default=False) #action='store_true')
parser.add_argument("--iterations", help="Number of iterations to run for the MLA iterative method.", type=int, default=1)


# Initial conditions, precision options, misc.
# parser.add_argument("--init_cond", help="Specifies the initial (xH, xHe, Tm).", type=)
parser.add_argument("--coarsen_factor", help="Coarsening to apply to the transfer function matrix. Default is 1.", type=int, default=1)
parser.add_argument("--backreaction", help="If False, uses the baseline TLA solution to calculate. Default is True.", type=bool, default=True)
parser.add_argument("--mxstep", help="The maximum number of steps allowed for each integration point. Default is 1000.", type=int, default=1000)
parser.add_argument("--rtol", help="The relative error of the solution. Default is 1e-4.", type=float, default=1e-4)
parser.add_argument("--use_tqdm", help="If True, uses tqdm to track progress.", type=bool, default=False) #action='store_true')
parser.add_argument("--tqdm_jupyter", help="Uses tqdm in Jupyter notebooks if True. Otherwise, uses tqdm for terminals. Default is False.", type=bool, default=False) #action='store_true')
parser.add_argument("--verbose", help="Do we need some verbose? default is none", type=int, default=False) #action='store_true')

# # These do not be specified from CLASS, just internal for DarkHistory
# parser.add_argument("--cross_check", help="", type=)
# parser.add_argument("--first_iter", help="", type=)
# parser.add_argument("--prev_output", help="", type=)
# parser.add_argument("--MLA_funcs", help="", type=)

args = parser.parse_args()

### Run DarkHistory
sys.exit(evolve_for_CLASS(**vars(args)))
