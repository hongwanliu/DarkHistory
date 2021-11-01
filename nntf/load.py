"""Loader for Neural Network Transfer Functions
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
    
    model_path = data_path + '/nntf_models/'
    
    if model_type == 'dep_nntf':
        
        if glob_dep_nntfs is None:
            
            rs_nodes = [40, 1600]
            hep_nntf = NNTF_Rs([model_path+'20210122_HEPp12_R0_long_run0',
                                model_path+'20210122_HEPp12_R1_long_run0',
                                model_path+'20210524_HEPp12_R2_run0'],
                                rs_nodes, 'hep_p12')
            prp_nntf = NNTF_Rs([model_path+'20210122_HEPs11_R0_run0',
                                model_path+'20210122_HEPs11_R1_run0',
                                model_path+'20210524_HEPs11_R2_run0'],
                               rs_nodes, 'hep_s11')
            lee_nntf = NNTF_Rs([model_path+'LEE_R0_run0',
                                model_path+'LEE_R1_run0',
                                model_path+'LEE_R2_run0'],
                               rs_nodes, 'lee')
            lep_pdtf  = LEPTF()
            
            glob_dep_nntfs = {
                'hep' : hep_nntf,
                'prp' : prp_nntf,
                'lee' : lee_nntf,
                'lep' : lep_pdtf
            }
            return glob_dep_nntfs
        
    elif model_type == 'ics_nntf':
        
        if glob_ics_nntfs is None:
            
            ics_thomson_nntf = NNTF_ref(model_path+'ICST_run0', 'ics_thomson')
            ics_engloss_nntf = NNTF_ref(model_path+'ICSE_run0', 'ics_engloss')
            ics_rel_nntf     = NNTF_ref(model_path+'ICSR_run0', 'ics_ref')
            
            
            glob_ics_nntfs = {
                'ics_thomson' : ics_thomson_nntf,
                'ics_engloss' : ics_engloss_nntf,
                'ics_enl'     : ics_rel_nntf
            }
            return glob_ics_nntfs