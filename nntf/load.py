"""Loading function for neural network transfer functions
"""

import sys

sys.path.append('..')
from config import data_path
from nntf.nntf import NNTF_Rs, NNTF_ref
from nntf.predtf import LEPTF

glob_dep_nntfs = None
glob_ics_nntfs = None

def load_model(model_type):
    
    global glob_dep_nntfs, glob_ics_nntfs
    
    model_path = data_path + 'nntf_models/'
    
    if model_type == 'dep_nntf':
        
        if glob_dep_nntfs is None:
            
            rs_nodes = [40, 1600]
            
            print('****** Loading (NN) transfer functions... ******')
            print('Using data at %s' % model_path)
            print('    for propagating photons (compounded)...  ', end='')
            hep_nntf = NNTF_Rs([model_path+'hep_p12_r0',
                                model_path+'hep_p12_r1',
                                model_path+'hep_p12_r2'],
                                rs_nodes, 'hep_p12')
            print('Done!')
            print('    for propagating photons (propagator)...  ', end='')
            prp_nntf = NNTF_Rs([model_path+'hep_s11_r0',
                                model_path+'hep_s11_r1',
                                model_path+'hep_s11_r2'],
                               rs_nodes, 'hep_s11')
            print('Done!')
            print('    for low-energy electrons...  ', end='')
            lee_nntf = NNTF_Rs([model_path+'lee_r0',
                                model_path+'lee_r1',
                                model_path+'lee_r2'],
                               rs_nodes, 'lee')
            print('Done!')
            print('    for low-energy photons...  ', end='')
            lep_pdtf  = LEPTF()
            print('Done!')
            print('****** Loading complete! ******')
            
            glob_dep_nntfs = {
                'hep' : hep_nntf, # hep refers to hep_p12
                'prp' : prp_nntf, # prp refers to hep_s11 (propagating)
                'lee' : lee_nntf,
                'lep' : lep_pdtf
            }
            return glob_dep_nntfs
        
    elif model_type == 'ics_nntf':
        
        if glob_ics_nntfs is None:
            
            print('****** Loading (NN) transfer functions... ******')
            print('Using models at %s' % model_path)
            print('    for inverse Compton (Thomson)... ', end='')
            ics_thomson_nntf = NNTF_ref(model_path+'ics_thomson', 'ics_thomson')
            print('Done!')
            print('    for inverse Compton (energy loss)... ', end='')
            ics_engloss_nntf = NNTF_ref(model_path+'ics_engloss', 'ics_engloss')
            print('Done!')
            print('    for inverse Compton (relativistic)... ', end='')
            ics_rel_nntf     = NNTF_ref(model_path+'ics_rel', 'ics_rel')
            print('Done!')
            print('****** Loading complete! ******')
            
            glob_ics_nntfs = {
                'ics_thomson' : ics_thomson_nntf,
                'ics_engloss' : ics_engloss_nntf,
                'ics_rel'     : ics_rel_nntf
            }
            return glob_ics_nntfs