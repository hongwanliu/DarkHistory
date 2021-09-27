"""Classes for Neural Network Transfer Functions
"""

import numpy as np
import time
import sys
import pickle

#import tensorflow as tf
from tensorflow import keras

sys.path.append('..')
from config import load_data
from darkhistory.spec.spectrum import Spectrum
import darkhistory.physics as phys

#from nntf.interps.interps import Interps
#HEP_L = Interps(DH_DIR + 'nntf/interps/HEP_L.interps')
#LB = Interps(DH_DIR + 'nntf/interps/lowerbound.interps')

#NNDH_DIR = '/zfs/yitians/darkhistory/NNDH/'
#sys.path.append(NNDH_DIR + 'training')
#import losses
#tf.keras.utils.get_custom_objects().update({"msep": losses.msep})

####################
### CONSTANTS

EPSILON = 1e-100
LOG_EPSILON = np.log(EPSILON)
LOG10_EPSILON = np.log10(EPSILON)

XMAX = (np.tanh(4)+1)/2
XMIN = (np.tanh(-5)+1)/2

RS_NODES = [40, 1600]

def clog(x):
    return np.log(np.clip(x, EPSILON, 1e100))


class NNTFRaw:
    """ Neural Network Transfer Function (NNTF) raw predictions.
    
    Parameters
    -----------
    
    in_spec_elec : :class:`.Spectrum`, optional
        Spectrum per injection event into electrons. *in_spec_elec.rs*
        of the :class:`.Spectrum` must be the initial redshift. 
    in_spec_phot : :class:`.Spectrum`, optional
        Spectrum per injection event into photons. *in_spec_phot.rs* 
        of the :class:`.Spectrum` must be the initial redshift. 
    rate_func_N : function, optional
        Function returning number of injection events per volume per time, with redshift :math:`(1+z)` as an input.  
    rate_func_eng : function, optional
        Function returning energy injected per volume per time, with redshift :math:`(1+z)` as an input. 
    DM_process : {'swave', 'decay'}, optional
        Dark matter process to use. 
        
    model : tensorflow.keras.Model custom subclass
        Predict the log value of (N -> N) transfer functions.
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
    # CTF LEE ICS ICSREL
    
    def __init__(self, model, TF_type):
        
        ####################
        ### model
        self.model = keras.models.load_model(model) if isinstance(model, str) else model
        self.TF_type = TF_type
        
        ####################
        ### data
        binning_data = load_data('binning')
        if self.TF_type == 'CTF': # [in, out]
            self.io_abscs = np.log([binning_data['phot'], binning_data['phot']])
        elif self.TF_type == 'LEE':
            self.io_abscs = np.log([binning_data['phot'], binning_data['elec']])
        elif self.TF_type == 'ICS':
            self.io_abscs = np.log([binning_data['ics_eng'], binning_data['ics_eng']])
        elif self.TF_type == 'ICSREL':
            self.io_abscs = np.log([binning_data['ics_rel_eng'], binning_data['ics_rel_eng']])
        else:
            raise ValueError('Invalid TF_type.')
        
        ####################
        ### TF
        self.raw_TF = None
        self.TF_shape = (len(self.io_abscs[0]), len(self.io_abscs[1]))
            
        ####################
        ### mask (same for all (rs, xH, xHe))
        self.mask = np.full(self.TF_shape, 0)
        if self.TF_type == 'CTF':
            for ii in range(173, 500):
                self.mask[ii,:ii+1] += 1
        elif self.TF_type == 'LEE':
            me = phys.me
            self.mask[223:,:136] += 1 # Eout < 3 keV, Ein > 3 keV
            for ii in range(200, 223):
                Ein = binning_data['phot'][ii]
                Eomax = (2 * Ein**2) / (phys.me + 2*Ein) # max kinetic energy of outgoing electron in compton scattering
                oimax = np.searchsorted(binning_data['elec'], Eomax) # max oi
                self.mask[ii,:oimax+1] += 1
        elif self.TF_type == 'ICS' or self.TF_type == 'ICSREL': ##################### edit mask
            self.mask += 1
                
        ####################
        ### _pred_in_2D
        self._pred_in_2D = []
        for ii in range(self.TF_shape[0]):
            for oi in range(self.TF_shape[1]):
                if self.mask[ii][oi]:
                    self._pred_in_2D.append( [self.io_abscs[0][ii], self.io_abscs[1][oi]] )
        self._pred_in_2D = np.array(self._pred_in_2D, dtype=np.float32)
        
        
    def predict_raw_TF(self, verbose=False, **params):
        
        if self.TF_type == 'CTF' or self.TF_type == 'LEE':
            rs  = params['rs']
            xH  = np.clip(params['xH'], XMIN, XMAX) if 'xH' in params else None
            xHe = np.clip(params['xHe']/0.07894737, XMIN, XMAX) if 'xHe' in params else None # convert to idl code's convention
            rs_in = np.log(rs)
            if rs > RS_NODES[1]:   # regime 2
                xH_in  = 4.0
                xHe_in = 4.0
            elif rs > RS_NODES[0]: # regime 1
                xH_in  = np.arctanh(2*xH-1)
                xHe_in = -5.0
            else:                  # regime 0
                xH_in  = np.arctanh(2*xH-1)
                xHe_in = np.arctanh(2*xHe-1)
        
            pred_in_shape = (len(self._pred_in_2D),)
            pred_in = np.c_[ np.full( pred_in_shape, xH_in , dtype=np.float32 ),
                             np.full( pred_in_shape, xHe_in, dtype=np.float32 ),
                             np.full( pred_in_shape, rs_in , dtype=np.float32 ),
                             self._pred_in_2D ]
        elif self.TF_type == 'ICS' or self.TF_type == 'ICSREL':
            pred_in = self._pred_in_2D
        
        pred_out = np.array(self.model.predict(pred_in, batch_size=len(pred_in), verbose=verbose)).flatten()
        
        # build raw_TF
        self.raw_TF = np.full(self.TF_shape, LOG_EPSILON)
        pred_out_i = 0
        for ii in range(self.TF_shape[0]):
            for oi in range(self.TF_shape[1]):
                if self.mask[ii][oi]:
                    self.raw_TF[ii][oi] = pred_out[pred_out_i]
                    pred_out_i += 1

                            
class NNTF (NNTFRaw):
    """
    Neural Network Transfer Function (NNTF) full predictions.
    
    Parameters
    -----------
    TF : ndarray (2D)
        Stores predicted transfer function (call predict_TF() first).
    lci_interp : :class:`.IonRSInterp`
        Gives lower cutoff index used in highengphot (compounded or not)
        and lowengelec transfer functions
    """
    
    def __init__(self, model, TF_type):
        super().__init__(model, TF_type)
        self.TF = None
        
        if self.TF_type == 'CTF' or self.TF_type == 'LEE':
            tf_helper_data = load_data('tf_helper')
            self.lci_interp = tf_helper_data['lci']
    
    def predict_TF(self, **params):
        
        self.predict_raw_TF(**params)
        self.TF = np.exp(self.raw_TF)
        
        if self.TF_type == 'LEE':
            lci = int(np.round(self.lci_interp.get_val(params['xH'], params['xHe'], params['rs']))) # xHe in lci_interp is DarkHistory convention
            self.TF[:lci,:] = 0
    
    def __call__(self, in_spec, **params):
        #if not np.all(np.log(in_spec.eng) == self.io_abscs[0]):
        #    raise ValueError('Incompatible input abscissa.')
        
        self.predict_TF(**params)
        
        out_spec_N = np.dot(in_spec.N, self.TF)
        return Spectrum(np.exp(self.io_abscs[1]), out_spec_N, rs=params['rs'], spec_type='N')