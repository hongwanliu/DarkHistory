""" Load Neural Network transfer functions (NNTFs)."""

import os
import logging

logger = logging.getLogger('darkhistory.nntf.load')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(name)s: %(message)s'))
handler.setLevel(logging.INFO)
logger.addHandler(handler)


def load_model(model_type, prefix=None):
    """ Loads NNTF models. 

    Parameters
    ----------
    model_type : {'dep_nntf', 'ics_nntf'}
        Type of models to load. The options are: 
        - *'dep_nntf'* -- NNTFMultiR / LEP_TF instances for propagating photon (or high energy photon) and low energy photon, low energy electron transfer functions. Corresponding to *'dep_tf'* in load_data;
        - *'ics_nntf'* -- ICS_NNTF instances for ICS secondary spectra in the Thomson regime and relativistic regime, and scattered electron energy-loss spectra;
        
    prefix : str, optional
        Path to the data directory. If not specified, the path is taken from the environment variable DH_DATA_DIR.

    Returns
    --------
    dict
        A dictionary of the data requested.
    """
    
    model_path = prefix if prefix is not None else os.environ['DH_DATA_DIR'] + '/nntf_models'
    
    if model_type == 'dep_nntf':
        from darkhistory.nntf.nntf import NNTFMultiR, HEP_NNTF, LEE_NNTF, LEP_TF
        tf_dict = {'lep' : LEP_TF()}
        for k, TF_class in zip(['hep_p12', 'hep_s11', 'lee'], [HEP_NNTF, HEP_NNTF, LEE_NNTF]):
            tf_dict[k] = NNTFMultiR([f'{model_path}/{k}_r0_weights.h5',
                                     f'{model_path}/{k}_r1_weights.h5',
                                     f'{model_path}/{k}_r2_weights.h5'], TF_class, k)
        logger.info('Loaded NNTF models for deposition.')
        return tf_dict
        
    elif model_type == 'ics_nntf':
        from darkhistory.nntf.nntf import ICS_NNTF
        tf_dict = {}
        for k in ['thomson', 'engloss', 'rel']:
            tf_dict[k] = ICS_NNTF(f'{model_path}/ics_{k}_weights.h5', f'ics_{k}')
        logger.info('Loaded NNTF models for ICS.')
        return tf_dict
