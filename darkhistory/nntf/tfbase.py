""" Base classes for Neural Network transfer functions (NNTFs).
"""

import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras

import sys
sys.path.append('..')

from darkhistory.config import load_data
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec.transferfunction import TransFuncAtRedshift

from darkhistory.nntf.utils import *


def tf_dnn_from_h5(file_path):

    with h5py.File(file_path, 'r') as f:
        layers = []
        i = 0
        weights_list = []
        while str(i) in f:
            weights = f[str(i)][()]
            biases = f[str(i+1)][()]
            if i == 0:
                layers.append(tf.keras.layers.Input(shape=(weights.shape[0],)))
            activation = 'relu' if i < len(f)-2 else 'linear'
            layers.append(tf.keras.layers.Dense(weights.shape[1], activation=activation))
            weights_list.append([weights, biases])
            i += 2

    model = tf.keras.Sequential(layers)
    for layer, (weights, biases) in zip(model.layers, weights_list):
        layer.set_weights([weights, biases])
        
    return model


class TFBase:
    """ Transfer functions with action base class.
    
    Methods
    --------
    __call__ : Returns :class:`.Spectrum` when acting on :class:`.Spectrum`
    
    TransFuncAtRedshift : Returns :class:`.TransFuncAtRedshift` instance.
    """
    
    def __init__(self):
        self.abscs = [None, None] # set during init
        self.TF_shape = None      # set during init
        self.spec_type = None     # set during init
        self.rs = None # set during predict_TF
        self.TF = None # set during predict_TF
    
    def __call__(self, in_spec):
        """ Action on a Spectrum object. Requires self.TF, self.abscs,
        self.rs, and self.spec_type to be set. """
        if self.TF is None:
            raise ValueError('Run predict_TF first!')
        if self.spec_type == 'N':
            return Spectrum(
                self.abscs[1], np.dot(in_spec.N, self.TF),
                rs=self.rs, spec_type='N'
            )
        elif self.spec_type == 'dNdE':
            return Spectrum(
                self.abscs[1], np.dot(in_spec.dNdE, self.TF),
                rs=self.rs, spec_type='dNdE'
            )
    
    def TransFuncAtRedshift(self):
        """ Get TransFuncAtRedshift instance. Requires self.TF, self.abscs,
        self.rs, and self.spec_type to be set. """
        if self.TF is None:
            raise ValueError('Run predict_TF first!')
        return TransFuncAtRedshift(
            self.TF, in_eng=self.abscs[0], eng=self.abscs[1],
            rs=self.rs, spec_type=self.spec_type, with_interp_func=True
        )
    
        
class NNTFBase (TFBase):
    """ Neural Network transfer function (NNTF) base class.
    
    Methods
    --------
    predict_TF : Let the NN model predict the tranfer function of given rs, xH, xHe key arguments.
    """
    
    def __init__(self, model_dir, TF_type):
        
        super().__init__()
        self.model = tf_dnn_from_h5(model_dir)
        self.TF_type = TF_type
        self._init_helpers()   # define helpers
        self._init_abscs()     # define self.abscs
        self._init_spec_type() # define self.spec_type
        self.TF_shape = (len(self.abscs[0]), len(self.abscs[1]))
        self.io_abscs = np.log(self.abscs)
        
        self._init_mask()    # define self.mask (same for all of (rs, xH, xHe))
        self._pred_in_2D = []
        for ii in range(self.TF_shape[0]):
            for oi in range(self.TF_shape[1]):
                if self.mask[ii][oi]:
                    self._pred_in_2D.append( [self.io_abscs[0][ii], self.io_abscs[1][oi]] )
        self._pred_in_2D = tf.convert_to_tensor(self._pred_in_2D, dtype=tf.float32)
        
        if len(self._pred_in_2D) > 1e6:
            self.predict_func = lambda x: self.model.predict(x, batch_size=1e6)
        else:
            self.predict_func = self.model.predict_on_batch
            #self.predict_func = self.model.__call__
        
    def _init_helpers(self):
        pass
        
    def _init_abscs(self):
        self.abscs = None
        
    def _init_spec_type(self):
        self.spec_type = None
        
    def _init_mask(self):
        self.mask = np.zeros(self.TF_shape)
        
    def predict_TF(self, **params):
        """ Core prediction function. Expect kwargs from rs, xH, xHe, and possibly E_arr depending on usage. """
        
        self.rs = params['rs']
        self._set_pred_in(**params)
        
        pred_out = np.array(self.predict_func(self.pred_in)).flatten()
        
        raw_TF = np.full(self.TF_shape, LOG_EPSILON)
        pred_out_i = 0
        for ii in range(self.TF_shape[0]):
            for oi in range(self.TF_shape[1]):
                if self.mask[ii][oi]:
                    raw_TF[ii][oi] = pred_out[pred_out_i]
                    pred_out_i += 1
        self.TF = np.exp(raw_TF)
        self._postprocess_TF(**params)
    
    def _set_pred_in(self, **params):
        self.pred_in = None
        
    def _postprocess_TF(self, **params):
        pass