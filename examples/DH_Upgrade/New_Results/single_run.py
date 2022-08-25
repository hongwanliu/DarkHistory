import os
from tty import CFLAG
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

# print('PLEASE CHECK idx!')

# idx = 0

import sys
sys.path.append('../../..')
import numpy as np
from scipy.interpolate import interp1d, interp2d

import config
import main
import pickle
import darkhistory.physics as phys
from darkhistory.spec.spectra import Spectra
from darkhistory.spec.spectrum import Spectrum
import darkhistory.low_energy.atomic as atomic
import darkhistory.spec.spectools as spectools

print('PLEASE CHECK output_dir!')
output_dir = '/scratch/gpfs/hongwanl/DarkHistory/nmax_convergence/'

# Choose to load data (True) or start a new scan (False)
load_data = False

nmax_ary = [10, 20, 50, 100, 200]
#######################################
#       Actual Iteration              #
#######################################

### Parameters
start_rs, high_rs, end_rs = 3e3, 1.555e3, 4
cf = 16
rtol = 1e-6
nmax = nmax_ary[idx] # maximum number of hydrogen levels to track
iter_max = 5 # number of times to iterate over recombination/ionization rates

params_list = [{'pri':'phot', 'DM_process':'decay', 'mDM':1e8, 'inj_param':1e40}]

options_dict = {
    'start_rs': start_rs, 'high_rs': high_rs, 'end_rs':end_rs,
    'reion_switch':False, 'reion_method':'Puchwein', 'heat_switch':True,
    'coarsen_factor':cf, 'distort':True, 'fexc_switch': True,
    'MLA_funcs':None,
    'reprocess_distortion':False, 'simple_2s1s':True,
    'nmax':nmax, 'rtol':rtol, 'use_tqdm':True, 
    'tqdm_jupyter':False, 'iterations':iter_max
}

# main.embarrassingly_parallel_evolve(
#     params_list, idx, options_dict, output_dir, 'max_CMB_nmax_200_Puchwein'
# )
main.embarrassingly_parallel_evolve(
    params_list, 0, options_dict, output_dir, 'std_distort_nmax_'+str(nmax)+'_simple_2s1s_no_reion'
)






# We use the iterative method.
# First solve the evolution equations assuming Recfast's
# alpha_B and beta_B rates (e.g. by setting recfast_TLA = True).
# One output is the a new, more accurate set of rates
# These rates can then be plugged back into the MLA equations
# Continued iteration leads to more accurate rates
# but this process converges quickly (after one iteration)
# for iteration in range(iter_max):
#     print('~~~Iteration ', iteration, '~~~')
#     data[iteration] = {model : [] for model in models}
#     # for pri in ['elec', 'phot']:
#     #     for DM_process in ['decay', 'swave']:
#     for pri in ['phot']: 
#         for DM_process in ['decay']:
#             model = pri+'_'+DM_process
#             print('starting', model)

#             if pri=='elec':
#                 if DM_process=='decay':
#                     mDM_list = 2*(10**log10eng + phys.me)
#                 else:
#                     mDM_list = 10**log10eng + phys.me
#             elif pri=='phot':
#                 if DM_process=='decay':
#                     mDM_list = 2*10**log10eng
#                 else:
#                     mDM_list = 10**log10eng

#             mDM = mDM_list[idx]

#             print('mDM: ', mDM)

#             param = param_bound(mDM, DM_process, pri)

#             # If this is first iteration, use Recfast TLA rates
#             if iteration == 0:
#                 TLA_switch = True
#                 MLA_funcs = None
#             # For subsequent iterations, use rates calculated from previous run
#             else:
#                 TLA_switch = False
#                 rates = data[iteration-1][model][0]['MLA']
#                 MLA_funcs = [interp1d(rates[0], rates[i], fill_value='extrapolate')
#                     for i in range(1,4)]

#             goods = main.evolve(
#                 DM_process=DM_process, mDM=mDM,
#                 lifetime=param, sigmav = param,
#                 primary=pri+'_delta',
#                 start_rs = start_rs, high_rs = high_rs, end_rs=end_rs,
#                 reion_switch=True, reion_method='Puchwein', heat_switch=True,
#                 coarsen_factor=cf,
#                 distort=True, recfast_TLA=TLA_switch, MLA_funcs=MLA_funcs,
#                 fexc_switch = True, reprocess_distortion=True, 
#                 nmax=nmax, rtol=rtol, use_tqdm=False
#             )

#             # Add the data for this iteration, model, and mass point
#             data[iteration][model].append(goods)

#             pickle.dump(data, open(
#                 output_dir
#                 +f'all_models_log10mDM_'
#                 +'{0:2.4f}'.format(np.log10(mDM)).replace('.', '_')
#                 +f'_nmax_{nmax}.p','wb')
#             )

            