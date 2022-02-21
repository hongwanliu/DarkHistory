""" Configuration and defaults.

"""

import os
import sys

import numpy as np
import json
import pickle

from scipy.interpolate import PchipInterpolator
from scipy.interpolate import pchip_interpolate
from scipy.interpolate import RegularGridInterpolator

# Location of all data files. CHANGE THIS FOR DARKHISTORY TO ALWAYS
# LOOK FOR THESE DATA FILES HERE. 

data_path    = '/zfs/yitians/darkhistory/DHdata_alt_hep/' ##### CP # CSalt data
data_path_v1 = '/zfs/yitians/darkhistory/DHdata/'       ##### CP # DarkHistory v1.0 version data

# Global variables for data.
glob_binning_data = None
glob_dep_tf_data  = None
glob_ics_tf_data  = None
glob_struct_data  = None
glob_hist_data    = None
glob_pppc_data    = None
glob_f_data       = None
glob_dep_ctf_data = None ##### TMP?
glob_tf_helper_data = None

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
    

def load_data(data_type, use_v1_data=False, verbose=1):
    """ Loads data from downloaded files. 

    Parameters
    ----------
    data_type : {'binning', 'dep_tf', 'ics_tf', 'struct', 'hist', 'f', 'pppc'}
        Type of data to load. The options are: 

        - *'binning'* -- Default binning for all transfer functions;

        - *'dep_tf'* -- Transfer functions for propagating photons and deposition into low-energy photons, low-energy electrons, high-energy deposition and upscattered CMB energy rate;
        
        - *'tf_helper'* -- Helper functions used in reconstructing transfer functions (from neural network);

        - *'ics_tf'* -- Transfer functions for ICS for scattered photons in the Thomson regime, relativistic regime, and scattered electron energy-loss spectrum; 

        - *'struct'* -- Structure formation boosts; 

        - *'hist'* -- Baseline ionization and temperature histories;

        - *'f'* -- :math:`f_c(z)` fractions without backreaction; and

        - *'pppc'* -- Data from PPPC4DMID for annihilation spectra. Specify the primary channel in *primary*.
        
    use_v1_data : bool
        If True, use data files from version 1.0 of DarkHistory, which does not include ['tf_helper'], ['binning']['ics_eng'], ['binning']['ics_rel_eng']

    Returns
    --------
    dict
        A dictionary of the data requested. 

    See Also
    ---------
    :func:`.get_pppc_spec`

    """

    global data_path
    global data_path_v1
    
    load_data_path = data_path_v1 if use_v1_data else data_path
    
    global glob_binning_data, glob_dep_tf_data, glob_ics_tf_data
    global glob_dep_ctf_data, glob_tf_helper_data
    global glob_struct_data,  glob_hist_data, glob_f_data, glob_pppc_data

    if load_data_path == '' or not os.path.isdir(load_data_path):
        print('NOTE: enter data directory in config.py to avoid this step.')
        load_data_path = input('Enter the data directory, e.g. /Users/foo/bar: ')

    ##################################################
    ### binning
    
    if data_type == 'binning':

        if glob_binning_data is None:
            glob_binning_data = pickle.load(open(load_data_path+'default_binning.p', 'rb'))

        return glob_binning_data # keys: 'phot', 'elec', 'ics_eng', 'ics_rel_eng'
        # if use_v1_data==True, return keys: 'phot', 'elec'

    ##################################################
    ### transfer functions
    
    elif data_type == 'dep_tf':

        if glob_dep_tf_data is None:

            if verbose >= 1:
                print('****** Loading transfer functions... ******')
                print('Using data at %s' % load_data_path)
                print('    for propagating photons... ', end =' ', flush=True)
            highengphot_tf_interp = pickle.load( open(load_data_path+'highengphot_tf_interp.raw', 'rb') )
            if verbose >= 1:
                print(' Done!')
                print('    for low-energy photons... ', end=' ', flush=True)
            lowengphot_tf_interp  = pickle.load( open(load_data_path+'lowengphot_tf_interp.raw', 'rb') )
            if verbose >= 1:
                print('Done!')
                print('    for low-energy electrons... ', end=' ', flush=True)
            lowengelec_tf_interp  = pickle.load( open(load_data_path+"lowengelec_tf_interp.raw", "rb") )
            if verbose >= 1:
                print('Done!')
                print('    for high-energy deposition... ', end=' ', flush=True)
            highengdep_interp     = pickle.load( open(load_data_path+"highengdep_interp.raw", "rb") )
            if verbose >= 1:
                print('Done!')
                print('    for total upscattered CMB energy rate... ', end=' ', flush=True)
            CMB_engloss_interp    = pickle.load( open(load_data_path+"CMB_engloss_interp.raw", "rb") )
            if verbose >= 1:
                print('Done!')
                print('****** Loading complete! ******', flush=True)

            glob_dep_tf_data = {
                'highengphot' : highengphot_tf_interp,
                'lowengphot'  : lowengphot_tf_interp,
                'lowengelec'  : lowengelec_tf_interp,
                'highengdep'  : highengdep_interp,
                'CMB_engloss' : CMB_engloss_interp
            }

        return glob_dep_tf_data
    
    elif data_type == 'hed_tf':

        if glob_dep_tf_data is None:

            if verbose >= 1:
                print('****** Loading transfer functions... ******')
                print('Using data at %s' % load_data_path)
                print('    for high-energy deposition... ', end=' ', flush=True)
            highengdep_interp     = pickle.load( open(load_data_path+"highengdep_interp.raw", "rb") )
            if verbose >= 1:
                print('Done!', flush=True)

            glob_dep_tf_data = {
                'highengdep'  : highengdep_interp
            }

        return glob_dep_tf_data

    ##################################################
    ### TMP
    elif data_type == 'dep_ctf':

        if glob_dep_ctf_data is None:
            
            if verbose >= 1:
                print('****** Loading compounded transfer function data... ******')
                print('Using data at %s' % load_data_path)
                print('    for propagating photons (compounded)...  ', end='', flush=True)
            hep_ctf_interp = pickle.load( open(load_data_path+'hep_p12_tf_interp.raw', 'rb'))
            if verbose >= 1:
                print('Done!')
                print('    for propagating photons (propagator)...  ', end='', flush=True)
            prp_ctf_interp = pickle.load( open(load_data_path+'hep_s11_tf_interp.raw', 'rb'))
            if verbose >= 1:
                print('Done!')
                print('****** Loading complete! ******', flush=True)
            
            glob_dep_ctf_data = {
                'hep': hep_ctf_interp,
                'prp': prp_ctf_interp
            }

        return glob_dep_ctf_data
    ##################################################
    
    elif data_type == 'tf_helper':
        
        if glob_tf_helper_data is None:
            
            tf_E_interp   = pickle.load( open(load_data_path+'tf_E_interp.raw', 'rb') )
            hep_lb_interp = pickle.load( open(load_data_path+'hep_lb_interp.raw', 'rb') )
            lci_interp    = pickle.load( open(load_data_path+'lci_interp.raw', 'rb') )
            hci_interp    = pickle.load( open(load_data_path+'hci_interp.raw', 'rb') )
            
            glob_tf_helper_data = {
                'tf_E'   : tf_E_interp,
                'hep_lb' : hep_lb_interp,
                'lci'    : lci_interp,
                'hci'    : hci_interp
            }
            
        return glob_tf_helper_data

    elif data_type == 'ics_tf':

        if glob_ics_tf_data is None:

            if verbose >= 1:
                print('****** Loading transfer functions... ******')
                print('    for inverse Compton (Thomson)... ', end=' ', flush=True)
            ics_thomson_ref_tf = pickle.load(
                open(load_data_path+"/ics_thomson_ref_tf.raw", "rb")
            )
            if verbose >= 1:
                print('Done!')
                print('    for inverse Compton (relativistic)... ', end=' ', flush=True)
            ics_rel_ref_tf     = pickle.load(
                open(load_data_path+"/ics_rel_ref_tf.raw",     "rb")
            )
            if verbose >= 1:
                print('Done!')
                print('    for inverse Compton (energy loss)... ', end=' ', flush=True)
            engloss_ref_tf     = pickle.load(
                open(load_data_path+"/engloss_ref_tf.raw",     "rb")
            )
            if verbose >= 1:
                print('Done!')
                print('****** Loading complete! ******', flush=True)

            glob_ics_tf_data = {
                'thomson' : ics_thomson_ref_tf,
                'rel'     : ics_rel_ref_tf,
                'engloss' : engloss_ref_tf
            }

        return glob_ics_tf_data

    ##################################################
    ### others
    
    elif data_type == 'struct':

        if glob_struct_data is None:

            boost_data = np.loadtxt(
                open(load_data_path+'/boost_data.txt', 'rb')
            )
            # einasto_subs = np.loadtxt(
            #     open(load_data_path+'/boost_Einasto_subs.txt', 'rb')
            # )

            glob_struct_data = {
                'einasto_subs'    : boost_data[:,[0,1]],
                'einasto_no_subs' : boost_data[:,[0,2]],
                'NFW_subs'        : boost_data[:,[0,3]],
                'NFW_no_subs'     : boost_data[:,[0,4]] 
            }

        return glob_struct_data

    elif data_type == 'hist':

        if glob_hist_data is None:

            soln_baseline = pickle.load(open(load_data_path+'/std_soln_He.p', 'rb'))

            glob_hist_data = {
                'rs'    : soln_baseline[0,:],
                'xHII'  : soln_baseline[2,:],
                'xHeII' : soln_baseline[3,:],
                'Tm'    : soln_baseline[1,:]
            }

        return glob_hist_data

    elif data_type == 'f':

        if glob_f_data is None:

            phot_ln_rs = np.array([np.log(3000) - 0.001*i for i in np.arange(6620)])
            phot_ln_rs_noStruct = np.array([np.log(3000) - 0.002*i for i in np.arange(3199)])
            elec_ln_rs = np.array([np.log(3000) - 0.008*i for i in np.arange(828)])

            log10eng0 = 3.6989700794219966
            log10eng = np.array([log10eng0 + 0.23252559*i for i in np.arange(40)])
            log10eng[-1] = 12.601505994846297

            f_phot_decay        = pickle.load(open(load_data_path+'/f_phot_decay_std.p', 'rb'))
            f_phot_swave        = pickle.load(open(load_data_path+'/f_phot_swave_std.p', 'rb'))
            f_phot_swave_struct = pickle.load(open(load_data_path+'/f_phot_swave_std_einasto_subs.p', 'rb'))
            f_elec_decay        = pickle.load(open(load_data_path+'/f_elec_decay_std.p', 'rb'))
            f_elec_swave        = pickle.load(open(load_data_path+'/f_elec_swave_std.p', 'rb'))
            f_elec_swave_struct = pickle.load(open(load_data_path+'/f_elec_swave_std_einasto_subs.p', 'rb'))

            f_phot_decay_interp        = RegularGridInterpolator((log10eng, np.flipud(phot_ln_rs)), np.log(f_phot_decay))
            f_phot_swave_interp        = RegularGridInterpolator((log10eng, np.flipud(phot_ln_rs_noStruct)), np.log(f_phot_swave))
            f_phot_swave_struct_interp = RegularGridInterpolator((log10eng, np.flipud(phot_ln_rs)), np.log(f_phot_swave_struct))
            f_elec_decay_interp        = RegularGridInterpolator((log10eng, np.flipud(elec_ln_rs)), np.log(f_elec_decay))
            f_elec_swave_interp        = RegularGridInterpolator((log10eng, np.flipud(elec_ln_rs)), np.log(f_elec_swave))
            f_elec_swave_struct_interp = RegularGridInterpolator((log10eng, np.flipud(elec_ln_rs)), np.log(f_elec_swave_struct))

            glob_f_data = {
                'phot_decay'        : f_phot_decay_interp,
                'phot_swave'        : f_phot_swave_interp,
                'phot_swave_struct' : f_phot_swave_struct_interp,
                'elec_decay'        : f_elec_decay_interp,
                'elec_swave'        : f_elec_swave_interp,
                'elec_swave_struct' : f_elec_swave_struct_interp
            }
        return glob_f_data

    elif data_type == 'pppc':

        if glob_pppc_data is None:

            coords_file_name = (
                load_data_path+'/dlNdlxIEW_coords_table.txt'
            )
            values_file_name = (
                load_data_path+'/dlNdlxIEW_values_table.txt'
            )

            with open(coords_file_name) as data_file:    
                coords_data = np.array(json.load(data_file))
            with open(values_file_name) as data_file:
                values_data = np.array(json.load(data_file))

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

            glob_pppc_data = dlNdlxIEW_interp

        return glob_pppc_data

    else:

        raise ValueError('invalid data_type.')