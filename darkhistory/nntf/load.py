""" Load Neural Network transfer functions (NNTFs).
"""

import sys
sys.path.append('..')

from darkhistory.config import data_path
from darkhistory.nntf.nntf import *

glob_dep_nntfs = None
glob_ics_nntfs = None

def load_model(model_type, verbose=1):
    """ Loads NNTF models. 

    Parameters
    ----------
    model_type : {'dep_nntf', 'ics_nntf'}
        Type of models to load. The options are: 

        - *'dep_nntf'* -- NNTFMultiR / LEP_TF instances for propagating photon (or high energy photon) and low energy photon, low energy electron transfer functions. Corresponding to *'dep_tf'* in load_data;

        - *'ics_nntf'* -- ICS_NNTF instances for ICS secondary spectra in the Thomson regime and relativistic regime, and scattered electron energy-loss spectra;
        
    verbose : {0, 1}
        Set verbosity.

    Returns
    --------
    dict
        A dictionary of the data requested.
    """
    
    global glob_dep_nntfs
    global glob_ics_nntfs
    
    model_path = data_path + '/nntf_models'
    
    if model_type == 'dep_nntf':
        
        if glob_dep_nntfs is None:
            
            if verbose >= 1:
                print('****** Loading (NN) transfer functions... ******')
                print('Using data at %s' % model_path)
                print('    for propagating photons (compounded)...  ', end='', flush=True)
            hep_p12_nntf = NNTFMultiR([model_path+'/hep_p12_r0_weights.h5',
                                       model_path+'/hep_p12_r1_weights.h5',
                                       model_path+'/hep_p12_r2_weights.h5'], HEP_NNTF, 'hep_p12')
            if verbose >= 1:
                print('Done!')
                print('    for propagating photons (propagator)...  ', end='', flush=True)
            hep_s11_nntf = NNTFMultiR([model_path+'/hep_s11_r0_weights.h5',
                                       model_path+'/hep_s11_r1_weights.h5',
                                       model_path+'/hep_s11_r2_weights.h5'], HEP_NNTF, 'hep_s11')
            if verbose >= 1:
                print('Done!')
                print('    for low-energy electrons...  ', end='', flush=True)
            lee_nntf = NNTFMultiR([model_path+'/lee_r0_weights.h5',
                                   model_path+'/lee_r1_weights.h5',
                                   model_path+'/lee_r2_weights.h5'], LEE_NNTF, 'lee')
            if verbose >= 1:
                print('Done!')
                print('    for low-energy photons...  ', end='', flush=True)
            lep_tf  = LEP_TF()
            if verbose >= 1:
                print('Done!')
                print('****** Loading complete! ******', flush=True)
            
            glob_dep_nntfs = {
                'hep_p12' : hep_p12_nntf,
                'hep_s11' : hep_s11_nntf,
                'lee' : lee_nntf,
                'lep' : lep_tf
            }
        return glob_dep_nntfs
        
    elif model_type == 'ics_nntf':
        
        if glob_ics_nntfs is None:
            
            if verbose >= 1:
                print('****** Loading (NN) transfer functions... ******')
                print('Using models at %s' % model_path)
                print('    for inverse Compton (Thomson)... ', end='', flush=True)
            ics_thomson_nntf = ICS_NNTF(model_path+'/ics_thomson_weights.h5', 'ics_thomson')
            if verbose >= 1:
                print('Done!')
                print('    for inverse Compton (energy loss)... ', end='', flush=True)
            ics_engloss_nntf = ICS_NNTF(model_path+'/ics_engloss_weights.h5', 'ics_engloss')
            if verbose >= 1:
                print('Done!')
                print('    for inverse Compton (relativistic)... ', end='', flush=True)
            ics_rel_nntf     = ICS_NNTF(model_path+'/ics_rel_weights.h5', 'ics_rel')
            if verbose >= 1:
                print('Done!')
                print('****** Loading complete! ******', flush=True)
            
            glob_ics_nntfs = {
                'ics_thomson' : ics_thomson_nntf,
                'ics_engloss' : ics_engloss_nntf,
                'ics_rel'     : ics_rel_nntf
            }
        return glob_ics_nntfs
