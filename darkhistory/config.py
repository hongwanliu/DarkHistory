""" Configuration and defaults."""

import os
import numpy as np
import json
import h5py

from scipy.interpolate import PchipInterpolator, pchip_interpolate, RegularGridInterpolator


#===== SET DATA PATH HERE =====#
# or set the environment variable DH_DATA_DIR.
data_path = '/n/holystore01/LABS/iaifi_lab/Users/yitians/darkhistory/DHupdate/DHdata_v1_1_full_new'

if data_path is None and 'DH_DATA_DIR' in os.environ.keys():
    data_path = os.environ['DH_DATA_DIR']
#==============================#

        
# Global variables for data.
glob_binning_data = None
glob_dep_tf_data  = None
glob_ics_tf_data  = None
glob_struct_data  = None
glob_hist_data    = None
glob_pppc_data    = None
glob_f_data       = None
glob_dep_ctf_data = None
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


def load_h5_dict(fn):
    """Load a dictionary from an HDF5 file."""
    d = {}
    with h5py.File(fn, 'r') as hf:
        for k, v in hf.items():
            d[k] = v[()]
    return d
    

def load_data(data_type, verbose=1):
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

    global data_path
    
    global glob_binning_data, glob_dep_tf_data, glob_ics_tf_data
    global glob_dep_ctf_data, glob_tf_helper_data
    global glob_struct_data,  glob_hist_data, glob_f_data, glob_pppc_data
    
    if data_path is None or not os.path.isdir(data_path):
        raise ValueError('Please set data directory in darkhistory.config or to `DH_DATA_DIR` environment variable.')

    ##################################################
    ### binning
    
    if data_type == 'binning':
        if glob_binning_data is None:
            try:
                glob_binning_data = load_h5_dict(data_path+'/binning.h5')
            except FileNotFoundError as err:
                print(type(err).__name__, ':', err)
                raise FileNotFoundError('Please update your dataset! See README.md for instructions.')
        return glob_binning_data

    ##################################################
    ### transfer functions
    
    elif data_type == 'dep_tf':
        from darkhistory.spec.transferfunclist import TransferFuncInterp
        from darkhistory.history.histools import IonRSInterp
        # prevent Spectrum -> physics -> load_data -> TransferFuncInterp -> Spectrum ciruclar import
        if glob_dep_tf_data is None:
            if verbose >= 1:
                print('****** Loading transfer functions... ******')
                print(f'Using data at {data_path}')
                print('    for propagating photons... ', end =' ', flush=True)
            highengphot_tf_interp = TransferFuncInterp(load_h5_dict(data_path+'/highengphot.h5'))
            if verbose >= 1:
                print(' Done!')
                print('    for low-energy photons... ', end=' ', flush=True)
            lowengphot_tf_interp  = TransferFuncInterp(load_h5_dict(data_path+'/lowengphot.h5'))
            if verbose >= 1:
                print('Done!')
                print('    for low-energy electrons... ', end=' ', flush=True)
            lowengelec_tf_interp  = TransferFuncInterp(load_h5_dict(data_path+'/lowengelec.h5'))
            if verbose >= 1:
                print('Done!')
                print('    for high-energy deposition... ', end=' ', flush=True)
            highengdep_interp     = IonRSInterp(load_h5_dict(data_path+'/highengdep.h5'))
            if verbose >= 1:
                print('Done!')
                print('    for total upscattered CMB energy rate... ', end=' ', flush=True)
            CMB_engloss_interp    = IonRSInterp(load_h5_dict(data_path+'/CMB_engloss.h5'))
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
        from darkhistory.history.histools import IonRSInterp
        if glob_dep_tf_data is None:
            if verbose >= 1:
                print('****** Loading transfer functions... ******')
                print(f'Using data at {data_path}')
                print('    for high-energy deposition... ', end=' ', flush=True)
            highengdep_interp     = IonRSInterp(load_h5_dict(data_path+'/highengdep.h5'))
            if verbose >= 1:
                print('Done!', flush=True)
            glob_dep_tf_data = {'highengdep'  : highengdep_interp}
        return glob_dep_tf_data
    

    elif data_type == 'tf_helper':
        from darkhistory.history.histools import IonRSInterp
        if glob_tf_helper_data is None:
            try:
                glob_tf_helper_data = {k : IonRSInterp(load_h5_dict(data_path+f'/{k}.h5')) for k in ['tf_E', 'hep_lb', 'lci', 'hci']}
            except FileNotFoundError as err:
                print(type(err).__name__, ':', err)
                raise FileNotFoundError('Neural network transfer function functionalities requires v1.1 data set!')
        return glob_tf_helper_data


    elif data_type == 'ics_tf':
        from darkhistory.spec.transferfunction import TransFuncAtRedshift
        if glob_ics_tf_data is None:
            if verbose >= 1:
                print('****** Loading transfer functions... ******')
                print('    for inverse Compton (Thomson)... ', end=' ', flush=True)
            ics_thomson_ref_tf = TransFuncAtRedshift(load_h5_dict(data_path+'/ics_thomson_ref.h5'))
            if verbose >= 1:
                print('Done!')
                print('    for inverse Compton (relativistic)... ', end=' ', flush=True)
            ics_rel_ref_tf     = TransFuncAtRedshift(load_h5_dict(data_path+'/ics_rel_ref.h5'))
            if verbose >= 1:
                print('Done!')
                print('    for inverse Compton (energy loss)... ', end=' ', flush=True)
            engloss_ref_tf     = TransFuncAtRedshift(load_h5_dict(data_path+'/ics_engloss_ref.h5'))
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
            boost_data = np.loadtxt(data_path+'/boost_data.txt')
            #einasto_subs = np.loadtxt(open(data_path+'/boost_Einasto_subs.txt', 'rb'))
            glob_struct_data = {
                'einasto_subs'    : boost_data[:,[0,1]],
                'einasto_no_subs' : boost_data[:,[0,2]],
                'NFW_subs'        : boost_data[:,[0,3]],
                'NFW_no_subs'     : boost_data[:,[0,4]] 
            }
        return glob_struct_data

    elif data_type == 'hist':
        if glob_hist_data is None:
            glob_hist_data = load_h5_dict(data_path+'/std_soln_He.h5')
        return glob_hist_data

    elif data_type == 'f':
        if glob_f_data is None:
            phot_ln_rs = np.array([np.log(3000) - 0.001*i for i in np.arange(6620)])
            phot_ln_rs_noStruct = np.array([np.log(3000) - 0.002*i for i in np.arange(3199)])
            elec_ln_rs = np.array([np.log(3000) - 0.008*i for i in np.arange(828)])

            log10eng0 = 3.6989700794219966
            log10eng = np.array([log10eng0 + 0.23252559*i for i in np.arange(40)])
            log10eng[-1] = 12.601505994846297

            f_dict = load_h5_dict(data_path+'/f_std.h5')
            f_phot_decay_interp        = RegularGridInterpolator((log10eng, np.flipud(phot_ln_rs)),          np.log(f_dict['f_phot_decay']))
            f_phot_swave_interp        = RegularGridInterpolator((log10eng, np.flipud(phot_ln_rs_noStruct)), np.log(f_dict['f_phot_swave']))
            f_phot_swave_struct_interp = RegularGridInterpolator((log10eng, np.flipud(phot_ln_rs)),          np.log(f_dict['f_phot_swave_struct']))
            f_elec_decay_interp        = RegularGridInterpolator((log10eng, np.flipud(elec_ln_rs)),          np.log(f_dict['f_elec_decay']))
            f_elec_swave_interp        = RegularGridInterpolator((log10eng, np.flipud(elec_ln_rs)),          np.log(f_dict['f_elec_swave']))
            f_elec_swave_struct_interp = RegularGridInterpolator((log10eng, np.flipud(elec_ln_rs)),          np.log(f_dict['f_elec_swave_struct']))

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

            coords_data = np.array(json.load(open(data_path+'/dlNdlxIEW_coords_table.json')), dtype=object)
            # coords_data is a (2, 23, 2) array. 
            # axis 0: stable SM secondaries, {'elec', 'phot'}
            # axis 1: annihilation primary channel.
            # axis 2: {mDM in GeV, np.log10(K/mDM)}, K is the energy of 
            # the secondary. 
            # Each element is a 1D array.

            values_data = np.array(json.load(open(data_path+'/dlNdlxIEW_values_table.json')), dtype=object)
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
        raise ValueError('Invalid data_type.')
