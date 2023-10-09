import os
import sys

idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
print(f"run {idx}")

import numpy as np
from scipy.interpolate import interp1d
import pickle

import config
import main
import darkhistory.physics as phys
import H2

output_dir = '/data/submit/wenzerq/H2_data/'

#######################################
#  Run evolve() and collapse scan     #
#######################################

def embarrassingly_parallel_collapse(DM_params, ind, evolve_options_dict, save_dir, file_name_str):
    # Get IGM evolution with DM energy injection
    main.embarrassingly_parallel_evolve(
        DM_params, ind, evolve_options_dict, save_dir, file_name_str
    )

    # Load data
    params = DM_params[ind]
    fn = (
        output_dir
        +params['pri']+'_'+params['DM_process']
        +'_'+'log10mDM_'+'{0:2.4f}'.format(np.log10(params['mDM']))
        +'_'+'log10param_'+'{0:2.4f}'.format(np.log10(params['inj_param']))
        +'_'+file_name_str+'_ind_'+str(ind)+'.p'
    )
    DM_data = pickle.load(open(fn, 'rb'))

    # Make new Spectra object that gives TOTAL distortion at each redshift
    DM_specs = DM_data['data'][-1]['distortions'].copy()
    for ii, rs in enumerate(DM_data['data'][-1]['rs']):
        temp_dist = DM_data['data'][-1]['distortions'].copy()
        temp_dist.redshift(rs)
        weights = np.zeros_like(DM_data['data'][-1]['rs'])
        weights[:ii] = 1
        DM_specs[ii] = temp_dist.sum_specs(weight=weights)

    # Make interpolation functions for f's
    f_func_list = {}
    for key in DM_data['data'][-1]['f']:
        f_func_list[key] = interp1d(
            DM_data['data'][-1]['rs'], DM_data['data'][-1]['f'][key],
            bounds_error=False, fill_value=0
        )

    # Package DM information together
    test_DM = (
        DM_data['DM_params']['mDM'], # mDM in [eV]
        DM_data['DM_params']['inj_param'], # lifetime in [s]
        DM_data['DM_params']['DM_process'], # injection type
        DM_data['DM_params']['pri'], # injected particle
        f_func_list, # energy deposition f's
    )

    # Get critical collapse curves
    rs_vir_list = np.logspace(np.log10(200.), np.log10(7.), num=30)
    res = H2.shooting_scheme(rs_vir_list, dists=DM_specs, 
                             H2_cool_rate='new', DM_switch=True, DM_args=test_DM)

    # Pickle the result
    collapse_fn = (
        output_dir
        +'critcollapse_'+params['pri']+'_'+params['DM_process']
        +'_'+'log10mDM_'+'{0:2.4f}'.format(np.log10(params['mDM']))
        +'_'+'log10param_'+'{0:2.4f}'.format(np.log10(params['inj_param']))
        +'_'+file_name_str+'_ind_'+str(ind)+'.p'
    )
    pickle.dump(res, open(collapse_fn, 'wb'))
    print('Critical collapse data at: ', collapse_fn)

    return None


#######################################
#       Actual Iteration              #
#######################################

### Parameters
start_rs, high_rs, end_rs = 3e3, 1.555e3, 4
cf = 16
rtol = 1e-6
nmax = 100 # maximum number of hydrogen levels to track
iter_max = 3 # number of times to iterate over recombination/ionization rates

params_list = []

len_mDM = 10
len_inj = 10

DM_process = 'swave'

# Range of masses to scan over depends on primary
for pri in ['elec', 'phot']:
    if pri == 'phot':
        mDM_arr = np.logspace(4.01, 12.78, num=len_mDM)
    else:
        mDM_arr = np.logspace(6.01, 12.78, num=len_mDM)
    
    # Array of lifetimes for decay; cross-sections for swave and pwave
    # Be careful about overwriting struct_boost for a given DM_process

    if DM_process == 'decay':
        struct_boost = None
        inj_par_array = np.logspace(22, 30, num=len_inj) # lifetime
    elif DM_process == 'swave':
        struct_boost = phys.struct_boost_func(model='NFW_no_subs')
        # inj_par_array = np.logspace(-33, -23, num=len_inj) # sigmav
        inj_par_array = np.logspace(-38, -35, num=len_inj) # sigmav / mDM
    else:
        struct_boost = phys.struct_boost_func(model='pwave_NFW_no_subs')
        #inj_par_array = np.logspace(-30, -14, num=len_inj) # sigmav
        inj_par_array = np.logspace(-39, -25, num=len_inj) # sigmav / mDM
           
    for mDM in mDM_arr:
        for inj_param in inj_par_array:
            if DM_process != 'decay':
                inj_param *= mDM
            params_list.append({
                'pri':pri, 'DM_process':DM_process, 
                'mDM':mDM, 'inj_param':inj_param
            })

options_dict = {
    'start_rs': start_rs, 'high_rs': high_rs, 'end_rs':end_rs,
    'struct_boost':struct_boost, 'heat_switch':False,
    'coarsen_factor':cf, 'rtol':rtol, 'use_tqdm':True, 'tqdm_jupyter':False, 
    'distort':True, 'fexc_switch': True, 
    'reprocess_distortion':True, 'nmax':nmax, 'iterations':iter_max
}


embarrassingly_parallel_collapse(
    params_list, idx, options_dict, output_dir, 'nmax100_no_reion'
)
