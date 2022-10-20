import os
# idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

print('PLEASE CHECK idx!')

idx = 4

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
output_dir = '/scratch/gpfs/hongwanl/DarkHistory/full_distortion/scratch/'

# Choose to load data (True) or start a new scan (False)
load_data = False

#######################################
#           CMB Constraints           #
#######################################


# CMB constraints on interaction rates
input_dir = config.data_path

f_elec_CMB_raw = np.loadtxt(input_dir+'/CMB_limits_elec_swave.dat', delimiter=',')
log10eng_elec_CMB  = f_elec_CMB_raw[0:2760:69, 0]
log10rs_elec_CMB = f_elec_CMB_raw[0:69, 1]

f_phot_CMB_raw = np.loadtxt(input_dir+'/CMB_limits_phot_swave.dat', delimiter=',')
log10eng_phot_CMB  = f_phot_CMB_raw[0:2800:70, 0]
log10rs_phot_CMB = f_phot_CMB_raw[0:70, 1]

f_elec_CMB_raw = np.transpose(np.reshape(f_elec_CMB_raw[:,2], (40,69)))
f_phot_CMB_raw = np.transpose(np.reshape(f_phot_CMB_raw[:,2], (40,70)))

f_elec_CMB = interp2d(log10eng_elec_CMB, log10rs_elec_CMB, f_elec_CMB_raw)
f_phot_CMB = interp2d(log10eng_phot_CMB, log10rs_phot_CMB, f_phot_CMB_raw)

decay_elec_CMB_raw = np.loadtxt(input_dir+'/CMB_limits_elec_decay.csv', delimiter=',')
decay_phot_CMB_raw = np.loadtxt(input_dir+'/CMB_limits_phot_decay.csv', delimiter=',')

decay_elec_CMB = interp1d(np.transpose(decay_elec_CMB_raw)[0,:], np.transpose(decay_elec_CMB_raw)[1,:])
decay_phot_CMB = interp1d(np.transpose(decay_phot_CMB_raw)[0,:], np.transpose(decay_phot_CMB_raw)[1,:])

#Derived from Planck 2018 cosmological parameters
p_ann = 3.5e-28

def param_bound_elec_CMB(mDM, DM_process):
    if DM_process == 'swave':
        return p_ann*(mDM*1e-9)/f_elec_CMB(np.log10(mDM-phys.me), np.log10(601))[0]
    elif DM_process == 'decay':
        return np.array([decay_elec_CMB(mDM*1e-9)])[0]

def param_bound_phot_CMB(mDM, DM_process):
    if DM_process == 'swave':
        return p_ann*(mDM*1e-9)/f_phot_CMB(np.log10(mDM), np.log10(601))[0]
    elif DM_process == 'decay':
        return np.array([decay_phot_CMB(mDM*1e-9)])[0]
    
def param_bound(mDM, DM_process, pri):
    if pri == 'elec':
        return param_bound_elec_CMB(mDM, DM_process)
    else:
        return param_bound_phot_CMB(mDM, DM_process)

#######################################
#       Actual Iteration              #
#######################################

### Parameters
start_rs, high_rs, end_rs = 3e3, 1.555e3, 4
cf = 16
rtol = 1e-6
nmax = 10 # maximum number of hydrogen levels to track
iter_max = 2 # number of times to iterate over recombination/ionization rates

log10eng0 = 3.6989700794219966
log10eng_ary = np.array([log10eng0 + 0.23252559*i for i in np.arange(40)])
log10eng_ary[-1] = 12.601505994846297

models = ['elec_decay', 'elec_swave', 'phot_decay', 'phot_swave']

params_list = []

for pri in ['elec', 'phot']:
    for DM_process in ['swave', 'decay']: 
        for log10eng in log10eng_ary:
            if pri=='elec':
                if DM_process=='decay':
                    mDM = 2*(10**log10eng + phys.me)
                else:
                    mDM = 10**log10eng + phys.me
            elif pri=='phot':
                if DM_process=='decay':
                    mDM = 2*10**log10eng
                else:
                    mDM = 10**log10eng

            params_list.append({
                'pri':pri, 'DM_process':DM_process, 
                'mDM':mDM, 'inj_param':param_bound(mDM, DM_process, pri)
            })

print(len(params_list))

options_dict = {
    'start_rs': start_rs, 'high_rs': high_rs, 'end_rs':end_rs,
    'reion_switch':True, 'reion_method':'Puchwein', 'heat_switch':True,
    'coarsen_factor':cf, 'distort':True, 'fexc_switch': True, 
    'reprocess_distortion':True, 'nmax':nmax, 'rtol':rtol, 'use_tqdm':True, 'tqdm_jupyter':False, 'iterations':iter_max
}


# main.embarrassingly_parallel_evolve(
#     params_list, idx, options_dict, output_dir, 'max_CMB_nmax_200_Puchwein'
# )

main.embarrassingly_parallel_evolve(
    params_list, idx, options_dict, output_dir, 'scratch'
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

            