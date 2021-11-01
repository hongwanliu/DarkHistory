"""Classes for General Transfer Functions
"""

import numpy as np
import sys

sys.path.append('..')
from config import load_data
from darkhistory.spec.spectrum import Spectrum

class PredTF: # abstract class for general predicting transfer functions
    """
    Abstract class for general predicting transfer functions.
    
    Methods
    --------
    __call__ : Returns :class:`.Spectrum` when acting on :class:`.Spectrum`
    """
    #def __init__(self):
    #    self.abscs = [None, None]
        
    #def predict_TF(self, rs=4.0, xH=None, xHe=None):
    #    self.rs = rs
    #    self.TF = None
    #    return self.TF
    
    def __call__(self, in_spec):
        rs = self.rs if hasattr(self, 'rs') else in_spec.rs
        out_spec_N = np.dot(in_spec.N, self.TF)
        return Spectrum(self.abscs[1], out_spec_N, rs=rs, spec_type='N')

    
########################################
# small instances

class LEPTF (PredTF):
    def __init__(self):
        binning_data = load_data('binning')
        self.abscs = [binning_data['phot'], binning_data['phot']]
        
        tf_helper_data = load_data('tf_helper')
        #self.lep_dis_interp = tf_helper_data['lep_dis']
    
    def predict_TF(self, rs=4.0, xH=None, xHe=None):
        #di1, di2 = self.lep_dis_interp.get_val(xH, xHe, rs)
        self.rs = rs
        self.TF = None
        return self.TF
    
    
########################################
# below is not in use
class ICSTF (PredTF):
    def __init__(self, TF_type):
        
        binning_data = load_data('binning')
        if self.TF_type in ['ics']:
            self.abscs = [binning_data['ics_eng'],
                          binning_data['ics_eng']]
        elif self.TF_type in ['ics_rel']:
            self.abscs = [binning_data['ics_rel_eng'],
                          binning_data['ics_rel_eng']]
        else:
            raise ValueError('Invalid TF_type.')
            
        print('Initializing constant transfer function: '+TF_type+'...')
        self.predict_TF()
        print('done.')
    
    def predict_TF(self):
        if self.TF_type == 'ics':
            pass