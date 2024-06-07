""" Configuration and defaults."""

import os
import sys
import logging

import numpy as np
import json
import pickle

import scipy
from scipy.interpolate import PchipInterpolator, pchip_interpolate, RegularGridInterpolator
scipy.interpolate.interpolate.RegularGridInterpolator = scipy.interpolate.RegularGridInterpolator # for compatibility with old data files

logger = logging.getLogger('darkhistory.config')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(name)s: %(message)s'))
handler.setLevel(logging.INFO)
logger.addHandler(handler)


class PchipInterpolator2D: 

    """ 2D interpolation over PPPC4DMID raw data, using the PCHIP method.

    Parameters
    -----------
    coords_data : ndarray, size (M,N)
        
    values_data : ndarray
    pri : string
        Specifies primary annihilation channel. See :func:`.get_pppc_spec` for the full list.
    sec : {'elec', 'phot'}
        Specifies which secondary spectrum to obtain (electrons/positrons or photons).
        
    Attributes
    ----------
    pri : string
        Specifies primary annihilation channel. See :func:`.get_pppc_spec` for the full list.
    sec : {'elec', 'phot'}
        Specifies which secondary spectrum to obtain (electrons/positrons or photons).
    get_val : function
        Returns the interpolation value at (coord, value) based
        
    Notes
    -------
    PCHIP stands for piecewise cubic hermite interpolating polynomial. This class was built to mimic the Mathematica interpolation of the PPPC4DMID data. 
    
    """
    
    def __init__(self, coords_data, values_data, pri, sec):
        if sec == 'elec':
            i = 0
            # fac is used to multiply the raw electron data by 2 to get the
            # e+e- spectrum that we always use in DarkHistory.
            fac = 2.
        elif sec == 'phot':
            i = 1
            fac = 1.
        else:
            raise TypeError('invalid final state.')
            
        self.pri = pri
        self.sec = sec

        # To compute the spectrum of 'e', we average over 'e_L' and 'e_R'.
        # We do the same thing for 'mu', 'tau', 'W' and 'Z'.
        # To avoid thinking too much, all spectra are split into two parts.
        # self._weight gives the weight of each half.
            
        if pri == 'e' or pri == 'mu' or pri == 'tau':
            pri_1 = pri + '_L'
            pri_2 = pri + '_R'
            self._weight = [0.5, 0.5]
        elif pri == 'W' or pri == 'Z':
            # 2 transverse pol., 1 longitudinal.
            pri_1 = pri + '_T'
            pri_2 = pri + '_L'
            self._weight = [2/3, 1/3]
        else:
            pri_1 = pri
            pri_2 = pri
            self._weight = [0.5, 0.5]

        idx_list_data = {
            'e_L': 0, 'e_R': 1, 'mu_L': 2, 'mu_R': 3, 'tau_L': 4, 'tau_R': 5,
            'q': 6, 'c': 7, 'b': 8, 't': 9,
            'W_L': 10, 'W_T': 11, 'Z_L': 12, 'Z_T': 13, 
            'g': 14, 'gamma': 15, 'h': 16,
            'nu_e': 17, 'nu_mu': 18, 'nu_tau': 19,
            'VV_to_4e': 20, 'VV_to_4mu': 21, 'VV_to_4tau': 22
        } 
        
        # Compile the raw data.
        mDM_in_GeV_arr_1 = np.array(
            coords_data[i, idx_list_data[pri_1], 0]
        )
        log10x_arr_1     = np.array(
            coords_data[i, idx_list_data[pri_1], 1]
        )
        values_arr_1     = np.array(values_data[i, idx_list_data[pri_1]])

        mDM_in_GeV_arr_2 = np.array(
            coords_data[i, idx_list_data[pri_2], 0]
        )
        log10x_arr_2     = np.array(
            coords_data[i, idx_list_data[pri_2], 1]
        )
        values_arr_2     = np.array(values_data[i, idx_list_data[pri_2]])

        self._mDM_in_GeV_arrs = [mDM_in_GeV_arr_1, mDM_in_GeV_arr_2] 
        self._log10x_arrs     = [log10x_arr_1,     log10x_arr_2]

        # Save the 1D PCHIP interpolator over mDM_in_GeV. Multiply the 
        # electron spectrum by 2 by adding np.log10(2).  
        self._interpolators = [
            PchipInterpolator(
                mDM_in_GeV_arr_1, values_arr_1 + np.log10(fac), 
                extrapolate=False
            ),
            PchipInterpolator(
                mDM_in_GeV_arr_2, values_arr_2 + np.log10(fac),
                extrapolate=False
            )
        ]
    
    def get_val(self, mDM_in_GeV, log10x):
        
        if (
            mDM_in_GeV < self._mDM_in_GeV_arrs[0][0] 
            or mDM_in_GeV < self._mDM_in_GeV_arrs[1][0]
            or mDM_in_GeV > self._mDM_in_GeV_arrs[0][-1]
            or mDM_in_GeV > self._mDM_in_GeV_arrs[1][-1]
        ):
            raise TypeError('mDM lies outside of the interpolation range.')
        
        # Call the saved interpolator at mDM_in_GeV, 
        # then use PCHIP 1D interpolation at log10x. 
        result1 = pchip_interpolate(
            self._log10x_arrs[0], self._interpolators[0](mDM_in_GeV), log10x
        )
        # Set all values outside of the log10x interpolation range to 
        # (effectively) zero. 
        result1[log10x >= self._log10x_arrs[0][-1]] = -100.
        result1[log10x <= self._log10x_arrs[0][0]]  = -100.
        
        result2 = pchip_interpolate(
            self._log10x_arrs[1], self._interpolators[1](mDM_in_GeV), log10x
        )
        result2[log10x >= self._log10x_arrs[1][-1]] = -100.
        result2[log10x <= self._log10x_arrs[1][0]]  = -100.
        
        # Combine the two spectra.  
        return np.log10(
            self._weight[0]*10**result1 + self._weight[1]*10**result2
        )
    

def load_data(data_type, prefix=None):
    """ Loads data from downloaded files. 

    Parameters
    ----------
    data_type : {'binning', 'dep_tf', 'hed_tf', 'tf_helper', 'ics_tf', 'struct', 'hist', 'f', 'pppc'}
        Type of data to load. The options are: 

        - *'binning'* -- Default binning for all transfer functions;

        - *'dep_tf'* -- Transfer functions for propagating photons and deposition into low-energy photons, low-energy electrons, high-energy deposition and upscattered CMB energy rate;
        
        - *'hed_tf'* -- Transfer functions for high-energy deposition only;
        
        - *'tf_helper'* -- Helper functions used in reconstructing transfer functions (from neural network);

        - *'ics_tf'* -- Transfer functions for ICS for scattered photons in the Thomson regime, relativistic regime, and scattered electron energy-loss spectrum; 

        - *'struct'* -- Structure formation boosts; 

        - *'hist'* -- Baseline ionization and temperature histories;

        - *'f'* -- :math:`f_c(z)` fractions without backreaction; and

        - *'pppc'* -- Data from PPPC4DMID for annihilation spectra. Specify the primary channel in *primary*.
        
    verbose : {0, 1}
        Set verbosity.
        
    Returns
    --------
    dict
        A dictionary of the data requested. 

    See Also
    ---------
    :func:`.get_pppc_spec`

    """

    use_v1_0_data = False
    
    if prefix is not None:
        data_path = prefix
    else:
        data_path = os.environ['DH_DATA_DIR']
    
    if data_type == 'binning':

        logger.info(f'Using data at {data_path}')

        try:
            if use_v1_0_data:
                binning = np.loadtxt(open(data_path+'/default_binning.p', 'rb'))
                binning_data =  {
                    'phot' : binning[0],
                    'elec' : binning[1]
                }
            else: # binning data expanded for v1.1
                binning_data = pickle.load(open(data_path+'/binning.p', 'rb'))
            return binning_data # keys: 'phot', 'elec', 'ics_eng', 'ics_rel_eng'
        except FileNotFoundError as err:
            raise FileNotFoundError('Please make sure to match the data set version with use_v1_0_data value in config.py!')
    
    elif data_type == 'dep_tf':
        tf_dict = {}
        for k, tf_name in zip(
            ['highengphot', 'lowengphot', 'lowengelec', 'highengdep', 'CMB_engloss'],
            ['highengphot_tf', 'lowengphot_tf', 'lowengelec_tf', 'highengdep', 'CMB_engloss']
        ):
            tf_dict[k] = pickle.load(open(f'{data_path}/{tf_name}_interp.raw', 'rb'))
        logger.info('Loaded deposition transfer functions.')

        return tf_dict
    
    elif data_type == 'hed_tf':
        tf_dict = {}
        for k in ['highengdep']:
            tf_dict[k] = pickle.load(open(f'{data_path}/{k}_interp.raw', 'rb'))
        logger.info('Loaded high energy deposition transfer functions.')
        
        return tf_dict
    
    elif data_type == 'tf_helper':
        try:
            tf_dict = {}
            for k in ['tf_E', 'hep_lb', 'lci', 'hci']:
                tf_dict[k] = pickle.load(open(f'{data_path}/{k}_interp.raw', 'rb'))
            logger.info('Loaded transfer function helpers.')
            return tf_dict
        except FileNotFoundError as err:
            print(type(err).__name__, ':', err)
            raise FileNotFoundError('Neural network transfer function functionalities requires v1.1 data set!')

    elif data_type == 'ics_tf':
        tf_dict = {}
        for k, tf_name in zip(['thomson', 'rel', 'engloss'], ['ics_thomson', 'ics_rel', 'engloss']):
            tf_dict[k] = pickle.load(open(f'{data_path}/{tf_name}_ref_tf.raw', 'rb'))
        logger.info('Loaded ICS transfer functions.')

        return tf_dict

    ##################################################
    ### others
    
    elif data_type == 'struct':

        boost_data = np.loadtxt(
            open(data_path+'/boost_data.txt', 'rb')
        )
        # einasto_subs = np.loadtxt(
        #     open(data_path+'/boost_Einasto_subs.txt', 'rb')
        # )

        return {
            'einasto_subs'    : boost_data[:,[0,1]],
            'einasto_no_subs' : boost_data[:,[0,2]],
            'NFW_subs'        : boost_data[:,[0,3]],
            'NFW_no_subs'     : boost_data[:,[0,4]] 
        }

    elif data_type == 'hist':

        soln_baseline = pickle.load(open(data_path+'/std_soln_He.p', 'rb'))

        return {
            'rs'    : soln_baseline[0,:],
            'xHII'  : soln_baseline[2,:],
            'xHeII' : soln_baseline[3,:],
            'Tm'    : soln_baseline[1,:]
        }

    elif data_type == 'f':

        phot_ln_rs = np.array([np.log(3000) - 0.001*i for i in np.arange(6620)])
        phot_ln_rs_noStruct = np.array([np.log(3000) - 0.002*i for i in np.arange(3199)])
        elec_ln_rs = np.array([np.log(3000) - 0.008*i for i in np.arange(828)])

        log10eng0 = 3.6989700794219966
        log10eng = np.array([log10eng0 + 0.23252559*i for i in np.arange(40)])
        log10eng[-1] = 12.601505994846297

        f_phot_decay        = pickle.load(open(data_path+'/f_phot_decay_std.p', 'rb'))
        f_phot_swave        = pickle.load(open(data_path+'/f_phot_swave_std.p', 'rb'))
        f_phot_swave_struct = pickle.load(open(data_path+'/f_phot_swave_std_einasto_subs.p', 'rb'))
        f_elec_decay        = pickle.load(open(data_path+'/f_elec_decay_std.p', 'rb'))
        f_elec_swave        = pickle.load(open(data_path+'/f_elec_swave_std.p', 'rb'))
        f_elec_swave_struct = pickle.load(open(data_path+'/f_elec_swave_std_einasto_subs.p', 'rb'))

        f_phot_decay_interp        = RegularGridInterpolator((log10eng, np.flipud(phot_ln_rs)), np.log(f_phot_decay))
        f_phot_swave_interp        = RegularGridInterpolator((log10eng, np.flipud(phot_ln_rs_noStruct)), np.log(f_phot_swave))
        f_phot_swave_struct_interp = RegularGridInterpolator((log10eng, np.flipud(phot_ln_rs)), np.log(f_phot_swave_struct))
        f_elec_decay_interp        = RegularGridInterpolator((log10eng, np.flipud(elec_ln_rs)), np.log(f_elec_decay))
        f_elec_swave_interp        = RegularGridInterpolator((log10eng, np.flipud(elec_ln_rs)), np.log(f_elec_swave))
        f_elec_swave_struct_interp = RegularGridInterpolator((log10eng, np.flipud(elec_ln_rs)), np.log(f_elec_swave_struct))

        return {
            'phot_decay'        : f_phot_decay_interp,
            'phot_swave'        : f_phot_swave_interp,
            'phot_swave_struct' : f_phot_swave_struct_interp,
            'elec_decay'        : f_elec_decay_interp,
            'elec_swave'        : f_elec_swave_interp,
            'elec_swave_struct' : f_elec_swave_struct_interp
        }

    elif data_type == 'pppc':
        
        coords_file_name = (
            data_path+'/dlNdlxIEW_coords_table.txt'
        )
        values_file_name = (
            data_path+'/dlNdlxIEW_values_table.txt'
        )

        with open(coords_file_name) as data_file:    
            coords_data = np.array(json.load(data_file), dtype=object)
        with open(values_file_name) as data_file:
            values_data = np.array(json.load(data_file), dtype=object)

        # coords_data is a (2, 23, 2) array. 
        # axis 0: stable SM secondaries, {'elec', 'phot'}
        # axis 1: annihilation primary channel.
        # axis 2: {mDM in GeV, np.log10(K/mDM)}, K is the energy of 
        # the secondary. 
        # Each element is a 1D array.

        # values_data is a (2, 23) array, d log_10 N / d log_10 (K/mDM). 
        # axis 0: stable SM secondaries, {'elec', 'phot'}
        # axis 1: annihilation primary channel.
        # Each element is a 2D array indexed by {mDM in GeV, np.log10(K/mDM)}
        # as saved in coords_data. 

        # Compile a dictionary of all of the interpolators.
        dlNdlxIEW_interp = {'elec':{}, 'phot':{}}

        chan_list = [
            'e_L','e_R', 'e', 'mu_L', 'mu_R', 'mu', 
            'tau_L', 'tau_R', 'tau',
            'q',  'c',  'b', 't',
            'W_L', 'W_T', 'W', 'Z_L', 'Z_T', 'Z', 'g',  'gamma', 'h',
            'nu_e', 'nu_mu', 'nu_tau',
            'VV_to_4e', 'VV_to_4mu', 'VV_to_4tau'
        ]

        for pri in chan_list:
            dlNdlxIEW_interp['elec'][pri] = PchipInterpolator2D(
                coords_data, values_data, pri, 'elec'
            )
            dlNdlxIEW_interp['phot'][pri] = PchipInterpolator2D(
                coords_data, values_data, pri, 'phot'
            )

        return dlNdlxIEW_interp

    else:
        raise ValueError('Invalid data_type.')
