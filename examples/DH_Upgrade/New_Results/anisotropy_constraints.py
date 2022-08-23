import os
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

# idx = 3

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

output_dir = '/scratch/gpfs/hongwanl/DarkHistory/CMB_anisotropy/max_CMB_nmax_200_Puchwein/'

#######################################
#       Actual Iteration              #
#######################################

### Parameters
start_rs, high_rs = 3e3, 1.555e3
cf = 16
rtol = 1e-6
nmax = 200 # maximum number of hydrogen levels to track
iter_max = 5 # number of times to iterate over recombination/ionization rates

log10eng_ary = np.logspace(0, 12, num=60)

models = ['elec_decay', 'elec_swave', 'phot_decay', 'phot_swave']

params_list = []

for pri in ['elec', 'phot']:
    for DM_process in ['swave', 'decay']:
        # Set ending redshift based on decay or swave constraint
        if DM_process == 'decay':
            end_rs = 290
        else:
            end_rs = 590
            
        # Get DM mass
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

options_dict = {
    'start_rs': start_rs, 'high_rs': high_rs, 'end_rs':end_rs,
    #'reion_switch':True, 'reion_method':'Puchwein', 'heat_switch':True,
    'coarsen_factor':cf, 'distort':True, 'fexc_switch': True, 
    'reprocess_distortion':True, 'nmax':nmax, 'rtol':rtol, 'use_tqdm':False,
    'backreaction':False
}

main.embarrassingly_parallel_evolve(
    params_list, idx, options_dict, output_dir, 'max_CMB_nmax_200_Puchwein'
)