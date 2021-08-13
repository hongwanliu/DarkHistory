"""Classes for Neural Network Transfer Functions

"""


import numpy as np
import time
import sys
import pickle

import tensorflow as tf
from tensorflow import keras

DH_DIR = '/zfs/yitians/darkhistory/DarkHistory/'
sys.path.append(DH_DIR)
from darkhistory.spec.spectrum import Spectrum

from nntf.interps.interps import Interps
HEP_L = Interps(DH_DIR + 'nntf/interps/HEP_L.interps')
LB = Interps(DH_DIR + 'nntf/interps/lowerbound.interps')

NNDH_DIR = '/zfs/yitians/darkhistory/NNDH/'
sys.path.append(NNDH_DIR + 'training')
import losses
tf.keras.utils.get_custom_objects().update({"msep": losses.msep})


EPSILON = 1e-100
LOG_EPSILON = np.log(EPSILON)
LOG10_EPSILON = np.log10(EPSILON)

XMAX = (np.tanh(4)+1)/2
XMIN = (np.tanh(-5)+1)/2

RS_01 = 48.
RS_12 = 1600.

def test_smooth(Earr, start_i, end_i):
    new_Earr = np.zeros_like(Earr)
    for i in range(start_i+1, end_i-1):
        new_Earr[i] = (Earr[i-1]+Earr[i]+Earr[i+1])/3
    #new_Earr *= (np.sum(Earr)/np.sum(new_Earr))
    return new_Earr

class NNTFRaw:
    """ Neural Network Transfer Function (NNTF) raw predictions.
    [Update!]----------
    model : tensorflow.keras.Model custom subclass
        Predict the log value of (dE/dloge -> dE/dloge) transfer functions.
    io_abscs : [1D np.ndarray, 1D np.ndarray]
        [input (energy) abscissa, ouput abscissa]
    log_io_abscs : [1D np.ndarray, 1D np.ndarray]
        log of [Input (energy) abscissa, ouput abscissa]. Defined for speed in cycle.
    pred_style : string
        Identifier for setting up the most efficient prediction input array.
    log_TF : 2D np.ndarray
        Store log value of latest TF prediction
    _pred_in_2D : 2D np.ndarray
        List of [log_Ein, log_Eout] for model prediction for faster prediction.
    """
    
    def __init__(self, model, option):
        self.model = keras.models.load_model(model) if isinstance(model, str) else model
        self.option = option
        abscs = pickle.load( open(DH_DIR+'nntf/abscs.p', 'rb') )
        if self.option == 'HEP' or self.option == 'CTF':
            self.io_abscs = [abscs['photE'], abscs['photE']]
        self.log_io_abscs = np.log(np.clip(self.io_abscs, EPSILON, None))
        self.log_TF = None
        
        self._pred_in_2D = []
        if self.option == 'HEP':
            for in_i in range(173, 500):     # start from 173
                for out_i in range(in_i):  # ignore (i,i) diagonals
                    self._pred_in_2D.append( [self.log_io_abscs[0][in_i], self.log_io_abscs[1][out_i]] )
            self._pred_in_2D = np.array(self._pred_in_2D, dtype='float32')
        elif self.option == 'CTF':
            for in_i in range(173, 500):     # start from 173
                for out_i in range(in_i+1):  # DONT ignore (i,i) diagonals
                    self._pred_in_2D.append( [self.log_io_abscs[0][in_i], self.log_io_abscs[1][out_i]] )
            self._pred_in_2D = np.array(self._pred_in_2D, dtype='float32')
        else:
            raise ValueError('Undefined option.')
    
    def predict_log_TF(self, verbose=False, **params):
        
        # build pred_in from _pred_in_2D
        rs = params['rs']
        if 'xH' in params:
            xH = np.clip(params['xH'], XMIN, XMAX)
        else:
            xH = None
        if 'xHe' in params:
            #xHe = np.clip(params['xHe']/0.08112582781456953, XMAX, XMIN)
            xHe = np.clip(params['xHe']/0.07894737, XMIN, XMAX)  # convert to idl code's convention
        else:
            xHe = None
        
        rs_in = np.log(rs)
        if rs > RS_12: # regime 2
            xH_in  = 4.0
            xHe_in = 4.0
        elif rs > RS_01: # regime 1
            xH_in  = np.arctanh(2*xH -1)
            xHe_in = -5.0
        else: # regime 0
            xH_in  = np.arctanh(2*xH  -1)
            xHe_in = np.arctanh(2*xHe -1)
        
        pred_in_shape = (len(self._pred_in_2D),)
        pred_in = np.c_[ np.full( pred_in_shape, xH_in , dtype='float32' ),
                         np.full( pred_in_shape, xHe_in, dtype='float32' ),
                         np.full( pred_in_shape, rs_in , dtype='float32' ),
                         self._pred_in_2D ]
        pred_out = np.array(self.model.predict(pred_in, batch_size=len(pred_in), verbose=verbose)).flatten()
        
        # build pred_log_TF
        self.log_TF = np.full((500,500), LOG_EPSILON)
        pred_out_i = 0
        if self.option == 'HEP':
            for in_i in range(173, 500):
                for out_i in range(in_i):
                    self.log_TF[in_i][out_i] = pred_out[pred_out_i]
                    pred_out_i += 1
        elif self.option == 'CTF':
            for in_i in range(173, 500):
                for out_i in range(in_i+1):
                    self.log_TF[in_i][out_i] = pred_out[pred_out_i]
                    pred_out_i += 1

                            
class NNTF (NNTFRaw):
    """ Neural Network Transfer Function (NNTF) full predictions.
    [Update!]----------
    TF : 2D np.ndarray
        Store fully predicted TF
    """
    
    def __init__(self, model, option):
        super().__init__(model, option)
        self.TF = None
    
    def predict_TF(self, **params):
        
        self.predict_log_TF(**params)
        self.TF = np.exp(self.log_TF)
        
        if self.option == 'HEP':
            
            ### fix prediction ###
            
            # erase value between 173 and in_start_i (tmp)
            in_start_i = int(round(HEP_L(**params)))
            # tmp fix
            if 'in_start_i' in params:
                in_start_i = params['in_start_i']
            for in_i in range(173, in_start_i):
                for out_i in range(in_i+1):
                    self.TF[in_i][out_i] = EPSILON
            # pred_range (tmp)
            if 'pred_range' in params:
                self.TF[np.logical_not(params['pred_range'])] = EPSILON
            
            ### test smooth ###
            #for in_i in range(173, 500):
            #    self.TF[in_i] = test_smooth(self.TF[in_i], 0, in_i)
            #    self.TF[in_i] = test_smooth(self.TF[in_i], 0, in_i)
            #    self.TF[in_i] = test_smooth(self.TF[in_i], 0, in_i)
            #    self.TF[in_i] = test_smooth(self.TF[in_i], 0, in_i)
            #    self.TF[in_i] = test_smooth(self.TF[in_i], 0, in_i)
            #    #self.TF[in_i] = test_smooth(self.TF[in_i]*self.io_abscs[1], 0, in_i)/self.io_abscs[1]
            
            ### energy conservation delta ###
            # add (i,i) delta via energy conservation
            hep_E_arr = np.zeros((500,))
            for in_i in range(173, 500):
                hep_E_arr[in_i] = Spectrum(self.io_abscs[1], self.TF[in_i], spec_type='N').toteng()
            tot_E_arr = hep_E_arr + params['oth_E_arr']
            diag = 1 - tot_E_arr/self.io_abscs[0]
            for i in range(173, 500):
                self.TF[i][i] = diag[i]
            
            # redshift relevant indices
            self.unredshifted_TF = self.TF.copy()
            LB_i = np.searchsorted(self.io_abscs[1], LB(**params))
            dlnphoteng = np.log(5565952217145.328/1e-4)/500
            dlnz = 0.001
            rate = dlnz/dlnphoteng
            for in_i in range(LB_i, 500):
                self.TF[in_i][in_i-1] += rate
                self.TF[in_i][in_i]   -= rate
                
        elif self.option == 'CTF':
            None # no post-processing
            
    
    def __call__(self, in_spec, new_pred=True, **params):
        if not np.all(in_spec.eng == self.io_abscs[0]):
            raise ValueError('Incompatible input abscissa.')
        
        if new_pred:
            self.predict_TF(**params)
            
        if 'unredshifted' in params:
            TF = self.unredshifted_TF
        else:
            TF = self.TF
        
        out_spec = np.dot(in_spec.N, TF)
        return Spectrum(self.io_abscs[1], out_spec, rs=params['rs'], in_eng=self.io_abscs[0], spec_type='N')