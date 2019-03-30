""" Configuration and defaults.

"""

import os
import sys

import numpy as np
import json
import pickle

data_path = '/Users/hongwan/Dropbox (MIT)/Photon Deposition/DarkHistory_data'

def load_data(data_type):
    """ Loads data from downloaded files. 

    Parameters
    ----------
    data_type:{'binning', 'dep_tf', 'ics_tf', 'struct', 'hist', 'pppc'}
        Type of data to load. The options are: 
        * *'binning'*: Default binning for all transfer functions; 
        * *'dep_tf'*: Transfer functions for propagating photons and deposition into low-energy photons, low-energy electrons, high-energy deposition and upscattered CMB energy rate;
        * *'ics_tf'*: Transfer functions for ICS for scattered photons in the Thomson regime, relativistic regime, and scattered electron energy-loss spectrum; 
        * *'struct'*: Structure formation boosts; 
        * *'hist'*: Baseline ionization and temperature histories, and
        * *'pppc'*: Data from PPPC4DMID for annihilation spectra. 

    Returns
    --------
    dict
        A dictionary of the data requested. 
    """

    if data_path == '' or not os.path.isdir(data_path):
        print('NOTE: enter data directory in config.py to avoid this step.')
        data_path = input('Enter the data directory, e.g. /Users/foo/bar: ')

    if data_type == 'binning':

        binning = np.loadtxt(open(data_path+'/default_binning.p', 'rb'))

        return {
            'photeng' : binning[0]
            'eleceng' : binning[1]
        }

    elif data_type == 'dep_tf':

        print('****** Loading transfer functions... ******')

        print('    for propagating photons... ', end =' ')
        highengphot_tf_interp = pickle.load(
            open(data_path+'/highengphot_tf_interp.raw', 'rb')
        )
        print(' Done!')

        print('    for low-energy photons... ', end=' ')
        lowengphot_tf_interp  = pickle.load(
            open(data_path+'/lowengphot_tf_interp.raw', 'rb')
        )
        print('Done!')

        print('    for low-energy electrons... ', end=' ')
        lowengelec_tf_interp  = pickle.load(
            open(data_path+"/lowengelec_tf_interp.raw", "rb")
        )
        print('Done!')

        print('    for high-energy deposition... ', end=' ')
        highengdep_interp     = pickle.load(
            open(data_path+"/highengdep_interp.raw", "rb")
        )
        print('Done!')

        print('    for total upscattered CMB energy rate... ', end=' ')
        CMB_engloss_interp    = pickle.load(
            open(data_path+"/CMB_engloss_interp.raw", "rb")
        )
        print('Done!')

        print('****** Loading complete! ******')

        return {
            'highengphot' : highengphot_tf_interp
            'lowengphot'  : lowengphot_tf_interp
            'lowengelec'  : lowengelec_tf_interp
            'highengdep'  : highengdep_interp
            'CMB_engloss' : CMB_engloss_interp
        }

    elif data_type == 'ics_tf':

        print('****** Loading transfer functions... ******')

        print('    for inverse Compton (Thomson)... ', end=' ')
        ics_thomson_ref_tf = pickle.load(
            open(data_path+"/ics_thomson_ref_tf.raw", "rb")
        )
        print('Done!')

        print('    for inverse Compton (relativistic)... ', end=' ')
        ics_rel_ref_tf     = pickle.load(
            open(data_path+"/ics_rel_ref_tf.raw",     "rb")
        )
        print('Done!')

        print('    for inverse Compton (energy loss)... ', end=' ')
        engloss_ref_tf     = pickle.load(
            open(data_path+"/engloss_ref_tf.raw",     "rb")
        )
        print('Done!')

        print('****** Loading complete! ******')

        return {
            'thomson' : ics_thomson_ref_tf
            'rel'     : ics_rel_ref_tf
            'engloss' : engloss_ref_tf
        }

    elif data_type == 'struct':

        einasto_subs = np.loadtxt(
            open(data_path+'/boost_Einasto_subs.txt', 'rb')
        )

        return {
            'einasto_subs' : einasto_subs
        }

    elif data_type == 'hist':

        soln_baseline = pickle.load(open(data_path+'/std_soln_He.p', 'rb'))

        return {
            'rs'    : soln_baseline[0,:]
            'xHII'  : soln_baseline[2,:]
            'xHeII' : soln_baseline[3,:]
            'Tm'    : soln_baseline[1,:]
        }

    elif data_type == 'pppc':

        coords_file_name = (
            data_path+'/dlNdlxIEW_coords_table.txt'
        )
        values_file_name = (
            data_path+'/dlNdlxIEW_values_table.txt'
        )

        with open(coords_file_name) as data_file:    
            coords_data = np.array(json.load(data_file))
        with open(values_file_name) as data_file:
            values_data = np.array(json.load(data_file))



# # Location of all data files. 

if 'pytest' not in sys.modules and 'readthedocs' not in sys.modules:

    data_path = '/Users/hongwan/Dropbox (MIT)/Photon Deposition/DarkHistory_data'
    #data_path = '/Users/gridgway/Dropbox (MIT)/Photon Deposition/DarkHistory_data'
    # data_path = '/Users/gregoryridgway/Dropbox (MIT)/Photon Deposition/DarkHistory_data'

    if data_path == '' or not os.path.isdir(data_path):
        print('NOTE: enter data directory in config.py to avoid this step.')
        data_path = input('Enter the data directory, e.g. /Users/foo/bar: ')

    # # Default binning for photons and low-energy electrons. 

    binning = np.loadtxt(open(data_path+'/default_binning.p', 'rb'))

    photeng = binning[0]
    eleceng = binning[1]

else:

    data_path = ''
    photeng   = np.zeros(500)
    eleceng   = np.zeros(500)

