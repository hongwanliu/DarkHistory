"""Classes for General Transfer Functions
"""

import numpy as np
import sys

sys.path.append('..')
from config import load_data
import darkhistory.physics as phys
from darkhistory.spec.spectrum import Spectrum
import darkhistory.spec.spectools as spectools

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

class LEPTF (PredTF):
    def __init__(self):
        binning_data = load_data('binning')
        self.abscs = [binning_data['phot'], binning_data['phot']]
        
        tf_helper_data = load_data('tf_helper')
        #self.lep_dis_interp = tf_helper_data['lep_dis']
    
    def predict_TF(self, rs=4.0, xH=None, xHe=None, E_arr=None, dlnz=0.001):
        l = len(E_arr)
        self.TF = np.zeros((l,l))
        
        rs_step = np.exp(-dlnz)
        # unnormalized cmb spec
        cmb_un = spectools.discretize(self.abscs[1],
                                      phys.CMB_spec,
                                      phys.TCMB(rs*rs_step)) # next rs step
        cmb_un_E = cmb_un.toteng()
        for i in range(l):
            if E_arr[i] > 0:
                self.TF[i][i] = E_arr[i]/self.abscs[1][i]
            elif E_arr[i] < 0:
                cmb_E = E_arr[i]
                self.TF[i] += (cmb_E/cmb_un_E) * cmb_un.N
                
        return self.TF
                