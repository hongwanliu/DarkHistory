""" Classes for Neural Network transfer functions (NNTFs).
"""

import numpy as np

import sys
sys.path.append('..')

import darkhistory.physics as phys
from darkhistory.nntf.utils import *
from darkhistory.nntf.tfbase import *

class LEP_TF (TFBase):
    """ Low energy photon transfer function.
    """
    
    def __init__(self):
        super().__init__()
        binning_data = load_data('binning')
        self.abscs = [binning_data['phot'], binning_data['phot']]
        self.TF_shape = (len(self.abscs[0]), len(self.abscs[1]))
        self.spec_type = 'N'
    
    def predict_TF(self, rs=4, E_arr=None, **params):
        self.rs = rs
        self.TF = np.zeros(self.TF_shape)
        np.fill_diagonal(self.TF, E_arr/self.abscs[1])
        
        
class HEP_NNTF (NNTFBase):
    """ High energy photon NNTF. """
        
    def _init_helpers(self):
        tf_helper_data = load_data('tf_helper')
        self.lci_interp = tf_helper_data['lci']
        self.hci_interp = tf_helper_data['hci']
        
    def _init_abscs(self):
        binning_data = load_data('binning')
        self.abscs = [binning_data['phot'], binning_data['phot']]
        
    def _init_spec_type(self):
        self.spec_type = 'N'
        
    def _init_mask(self):
        self.mask = np.zeros(self.TF_shape)
        for ii in range(173, 500):
            self.mask[ii,:ii+1] += 1
    
    def _set_pred_in(self, rs=4, xH=0, xHe=0, **params):
        rs_in = np.log(rs)
        xH_in = xH * 10
        xHe_in = xHe * 100
        col_shape = tf.TensorShape( [self._pred_in_2D.shape[0], 1] )
        
        if rs > RS_NODES[1]:
            rs_col = tf.cast(tf.fill(col_shape, rs_in), tf.float32)
            self.pred_in = tf.concat([rs_col, self._pred_in_2D], axis=1)
        elif rs > RS_NODES[0]:
            xH_col = tf.cast(tf.fill(col_shape, xH_in), tf.float32)
            rs_col = tf.cast(tf.fill(col_shape, rs_in), tf.float32)
            self.pred_in = tf.concat([xH_col, rs_col, self._pred_in_2D], axis=1)
        else:
            xH_col  = tf.cast(tf.fill(col_shape, xH_in),  tf.float32)
            xHe_col = tf.cast(tf.fill(col_shape, xHe_in), tf.float32)
            rs_col  = tf.cast(tf.fill(col_shape, rs_in),  tf.float32)
            self.pred_in = tf.concat([xH_col, xHe_col, rs_col, self._pred_in_2D], axis=1)
        
    def _postprocess_TF(self, rs=4, xH=0, xHe=0, E_arr=None):
        ## restore negative values
        iz = distortion_zero_est(rs)
        for i in range(223, 500):
            iz = distortion_zero(self.TF[i], iz)
            self.TF[i][:iz] *= -1
            self.TF[i][iz] = self.TF[i][iz-1] + (self.TF[i][iz+1]-self.TF[i][iz-1]) / (self.abscs[-1][iz+1]-self.abscs[-1][iz-1]) * (self.abscs[-1][iz]-self.abscs[-1][iz-1])
            
        ## cut below lci
        lci = int(np.round(self.lci_interp.get_val(xH, xHe, rs)))
        self.TF[:lci,:] = 0
        if self.TF_type == 'hep_s11':
            for i in range(self.TF_shape[0]):
                self.TF[i][i] += 1
                
        ## adjust for E_arr
        if E_arr is not None:
            hci = int(np.round(self.hci_interp.get_val(xH, xHe, rs)))
            i_st = 12 if self.TF_type == 'hep_s11' else lci
            for i in range(i_st, 500):
                if i <= hci:
                    scale_to_E(self.abscs[1], self.TF[i], i-12, i, E_arr[i])
                else:
                    self.TF[i,i:] = 0
                    scale_to_E(self.abscs[1], self.TF[i], 0, i, E_arr[i])
    
    
class LEE_NNTF (HEP_NNTF):
    """ Low energy electron NNTF. """
        
    def _init_abscs(self):
        binning_data = load_data('binning')
        self.abscs = [binning_data['phot'], binning_data['elec']]
        
    def _init_mask(self):
        self.mask = np.zeros(self.TF_shape)
        self.mask[223:,:136] += 1 # Eout < 3 keV, Ein > 3 keV
        for ii in range(200, 223):
            Ein = self.abscs[0][ii]
            Eomax = (2 * Ein**2) / (phys.me + 2*Ein) # max kinetic energy of outgoing electron in compton scattering
            oimax = np.searchsorted(self.abscs[1], Eomax) # max oi
            self.mask[ii,:oimax+1] += 1
        
    def _postprocess_TF(self, rs=4, xH=0, xHe=0, E_arr=None):
        ## cut below lci
        lci = int(np.round(self.lci_interp.get_val(xH, xHe, rs)))
        self.TF[:lci,:] = 0
        ## adjust for E_arr
        if E_arr is not None:
            i_st = lci
            for i in range(i_st, 500):
                scale_to_E(self.abscs[1], self.TF[i], 0, 135, E_arr[i])
    
    
class ICS_NNTF (NNTFBase):
    """ Inverse compton scattering NNTF. """
    
    def __init__(self, model_str, TF_type):
        super().__init__(model_str, TF_type)
        self.predict_TF(rs=np.full_like(self.abscs[0], 400)) # predict once as reference
        
    def _init_abscs(self):
        binning_data = load_data('binning')
        if self.TF_type == 'ics_rel':
            self.abscs = [binning_data['ics_rel_eng_1k'], binning_data['ics_rel_eng_1k']]
        else:
            self.abscs = [binning_data['ics_eng_1k'], binning_data['ics_eng_1k']]
        
    def _init_spec_type(self):
        self.spec_type = 'dNdE'
        
    def _init_mask(self):
        self.mask = np.zeros(self.TF_shape)
        for ii in range(len(self.abscs[0])):
            oimax = np.searchsorted(self.abscs[1], ics_pred_Eout_max(self.abscs[0][ii], self.TF_type)) + 1
            self.mask[ii,:oimax+1] += 1
    
    def _set_pred_in(self, **params):
        self.pred_in = self._pred_in_2D
        
        
class NNTFMultiR (TFBase):
    """ NNTF for multiples redshift regimes. """
    
    def __init__(self, model_str_list, TF_class, TF_type):
        super().__init__()
        self.NNTF_list = [ TF_class(model_str, TF_type) for model_str in model_str_list ]
        self.abscs = self.NNTF_list[0].abscs
        self.TF_shape = self.NNTF_list[0].TF_shape
        self.spec_type = self.NNTF_list[0].spec_type
    
    def predict_TF(self, rs=4.0, **params):
        self.rs = rs
        ri = np.searchsorted(RS_NODES, rs)
        self.NNTF_list[ri].predict_TF(rs=rs, **params)
        self.TF = self.NNTF_list[ri].TF