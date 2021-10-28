import numpy as np
import sys

sys.path.append('..')
from config import load_data
from darkhistory.spec.spectrum import Spectrum

class LEPTF:
    def __init__(self):
        binning_data = load_data('binning')
        self.abscs = [binning_data['phot'], binning_data['phot']]
        
        tf_helper_data = load_data('tf_helper')
        #self.lep_dis_interp = tf_helper_data['lep_dis']
        self.TF = None
        self.rs = None
    
    def get_TF(self, rs=4.0, xH=None, xHe=None):
        #di1, di2 = self.lep_dis_interp.get_val(xH, xHe, rs)
        self.rs = rs
        return self.TF
    
    def __call__(self, in_spec):
        out_spec_N = np.dot(in_spec.N, self.TF)
        return Spectrum(self.abscs[1], out_spec_N, rs=self.rs, spec_type='N')