""" Configuration and defaults.

"""

import os
import sys

import numpy as np
import json
import pickle

from scipy.interpolate import PchipInterpolator
from scipy.interpolate import pchip_interpolate


data_path = '/Users/hongwan/Dropbox (MIT)/Photon Deposition/DarkHistory_data'

class PchipInterpolator2D: 

    """ 2D interpolation over PPPC4DMID raw data, using PCHIP method.

    Parameters
    -----------
    pri : string
        Specifies primary annihilation channel. See below for list.
    sec : {'elec', 'phot'}
        Specifies stable SM secondary to get the spectrum of.

    Attributes
    ----------
    pri : string
        Specifies primary annihilation channel. See below for list.
    sec : {'elec', 'phot'}
        Specifies stable SM secondary to get the spectrum of.
    get_val : function
        Returns the interpolation value at (mDM_in_GeV, log10(K/mDM)) where K is the kinetic energy of the secondary.

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

def load_data(data_type, primary=None):
    """ Loads data from downloaded files. 

    Parameters
    ----------
    data_type : {'binning', 'dep_tf', 'ics_tf', 'struct', 'hist', 'pppc'}
        Type of data to load. The options are: 
        * *'binning'*: Default binning for all transfer functions; 
        * *'dep_tf'*: Transfer functions for propagating photons and deposition into low-energy photons, low-energy electrons, high-energy deposition and upscattered CMB energy rate;
        * *'ics_tf'*: Transfer functions for ICS for scattered photons in the Thomson regime, relativistic regime, and scattered electron energy-loss spectrum; 
        * *'struct'*: Structure formation boosts; 
        * *'hist'*: Baseline ionization and temperature histories, and
        * *'pppc'*: Data from PPPC4DMID for annihilation spectra. Specify the primary channel in *primary*. 

    primary : str, optional
        The valid options are *{'e_L','e_R', 'e', 'mu_L', 'mu_R', 'mu', 'tau_L', 'tau_R', 'tau', 'q',  'c',  'b', 't', 'W_L', 'W_T', 'W', 'Z_L', 'Z_T', 'Z', 'g',  'gamma', 'h', 'nu_e', 'nu_mu', 'nu_tau', 'VV_to_4e', 'VV_to_4mu', 'VV_to_4tau'}*. See :func:`.get_pppc_spec` for more details.


    Returns
    --------
    dict
        A dictionary of the data requested. 

    See Also
    ---------
    :func:`.get_pppc_spec`

    """

    path = data_path

    if path == '' or not os.path.isdir(path):
        print('NOTE: enter data directory in config.py to avoid this step.')
        path = input('Enter the data directory, e.g. /Users/foo/bar: ')

    if data_type == 'binning':

        binning = np.loadtxt(open(path+'/default_binning.p', 'rb'))

        return {
            'phot' : binning[0],
            'elec' : binning[1]
        }

    elif data_type == 'dep_tf':

        print('****** Loading transfer functions... ******')

        print('    for propagating photons... ', end =' ')
        highengphot_tf_interp = pickle.load(
            open(path+'/highengphot_tf_interp.raw', 'rb')
        )
        print(' Done!')

        print('    for low-energy photons... ', end=' ')
        lowengphot_tf_interp  = pickle.load(
            open(path+'/lowengphot_tf_interp.raw', 'rb')
        )
        print('Done!')

        print('    for low-energy electrons... ', end=' ')
        lowengelec_tf_interp  = pickle.load(
            open(path+"/lowengelec_tf_interp.raw", "rb")
        )
        print('Done!')

        print('    for high-energy deposition... ', end=' ')
        highengdep_interp     = pickle.load(
            open(path+"/highengdep_interp.raw", "rb")
        )
        print('Done!')

        print('    for total upscattered CMB energy rate... ', end=' ')
        CMB_engloss_interp    = pickle.load(
            open(path+"/CMB_engloss_interp.raw", "rb")
        )
        print('Done!')

        print('****** Loading complete! ******')

        return {
            'highengphot' : highengphot_tf_interp,
            'lowengphot'  : lowengphot_tf_interp,
            'lowengelec'  : lowengelec_tf_interp,
            'highengdep'  : highengdep_interp,
            'CMB_engloss' : CMB_engloss_interp
        }

    elif data_type == 'ics_tf':

        print('****** Loading transfer functions... ******')

        print('    for inverse Compton (Thomson)... ', end=' ')
        ics_thomson_ref_tf = pickle.load(
            open(path+"/ics_thomson_ref_tf.raw", "rb")
        )
        print('Done!')

        print('    for inverse Compton (relativistic)... ', end=' ')
        ics_rel_ref_tf     = pickle.load(
            open(path+"/ics_rel_ref_tf.raw",     "rb")
        )
        print('Done!')

        print('    for inverse Compton (energy loss)... ', end=' ')
        engloss_ref_tf     = pickle.load(
            open(path+"/engloss_ref_tf.raw",     "rb")
        )
        print('Done!')

        print('****** Loading complete! ******')

        return {
            'thomson' : ics_thomson_ref_tf,
            'rel'     : ics_rel_ref_tf,
            'engloss' : engloss_ref_tf
        }

    elif data_type == 'struct':

        einasto_subs = np.loadtxt(
            open(path+'/boost_Einasto_subs.txt', 'rb')
        )

        return {
            'einasto_subs' : einasto_subs
        }

    elif data_type == 'hist':

        soln_baseline = pickle.load(open(path+'/std_soln_He.p', 'rb'))

        return {
            'rs'    : soln_baseline[0,:],
            'xHII'  : soln_baseline[2,:],
            'xHeII' : soln_baseline[3,:],
            'Tm'    : soln_baseline[1,:]
        }

    elif data_type == 'pppc':

        coords_file_name = (
            path+'/dlNdlxIEW_coords_table.txt'
        )
        values_file_name = (
            path+'/dlNdlxIEW_values_table.txt'
        )

        with open(coords_file_name) as data_file:    
            coords_data = np.array(json.load(data_file))
        with open(values_file_name) as data_file:
            values_data = np.array(json.load(data_file))

        # Dictionary for the interpolators
        dlNdlxIEW_interp = {}

        dlNdlxIEW_interp['elec'] = PchipInterpolator2D(
            coords_data, values_data, primary, 'elec'
        )
        dlNdlxIEW_interp['phot'] = PchipInterpolator2D(
            coords_data, values_data, primary, 'phot'
        )

        return dlNdlxIEW_interp

    else:

        raise ValueError('invalid data_type.')


# # Location of all data files. 

# if 'pytest' not in sys.modules and 'readthedocs' not in sys.modules:

#     data_path = '/Users/hongwan/Dropbox (MIT)/Photon Deposition/DarkHistory_data'
#     #data_path = '/Users/gridgway/Dropbox (MIT)/Photon Deposition/DarkHistory_data'
#     # data_path = '/Users/gregoryridgway/Dropbox (MIT)/Photon Deposition/DarkHistory_data'

#     if data_path == '' or not os.path.isdir(data_path):
#         print('NOTE: enter data directory in config.py to avoid this step.')
#         data_path = input('Enter the data directory, e.g. /Users/foo/bar: ')

#     # # Default binning for photons and low-energy electrons. 

#     binning = np.loadtxt(open(data_path+'/default_binning.p', 'rb'))

#     photeng = binning[0]
#     eleceng = binning[1]

# else:

#     data_path = ''
#     photeng   = np.zeros(500)
#     eleceng   = np.zeros(500)

