import sys
sys.path.append("../..")

import numpy as np
import scipy.interpolate as interp
import darkhistory.physics as phys
import darkhistory.utilities as utils

import os
cwd = os.getcwd()
abspath = os.path.abspath(__file__)
dir_path = os.path.dirname(abspath)
#dir_path = os.path.dirname(os.path.realpath(__file__))

def make_interpolator():
    """Creates cubic splines that interpolate the Medea Data.  Stores them in globally defined variables so that these functions are only computed once

    Assumes that the data files are in the same directory as this script.

    Parameters
    ----------

    Returns
    -------
    """

    #engs = np.array([10.2, 13.6, 14, 30, 60, 100, 300, 3000])
    #print('AHHHHHH NOOOOOO!')
    engs = np.array([14, 30, 60, 100, 300, 3000])
    print('AHHHH YEAHHHH!')

    grid_vals = np.zeros((26, len(engs), 5))
    os.chdir(dir_path)
    # load MEDEA files
    for i, eng in enumerate(engs):
        with open('results-'+str(eng)+'ev-xH-xHe_e-10-yp024.dat','r') as f:
            lines_list = f.readlines()

            # load ionization levels only once
            if i==0:
                xes = np.array([float(line.split('\t')[0]) for line in lines_list[2:]])

            # load deposition fractions for each energy
            grid_vals[:,i,:] = np.transpose(np.array([
                [
                    #set 0 to 10^-15 to avoid -\infty
                    max(float(line.split('\t')[k]),1.0e-200)
                    for line in lines_list[2:]
                ] for k in [1,2,3,4,5]
            ]))

    os.chdir(cwd)

    MEDEA_interp = utils.Interpolator2D(xes, 'xes', engs, 'engs', grid_vals, logInterp=True)

    return MEDEA_interp

def compute_fs(MEDEA_interp, spec_elec, xe, dE_dVdt_inj, dt):
    """ Given an electron energy spectrum, calculate how much of that energy splits into
    continuum photons, lyman_alpha transitions, H ionization, He ionization, and heating of the IGM.

    Parameters
    ----------
    spec_elec : Spectrum object
        spectrum of low energy electrons. spec_elec.toteng() should return energy per baryon.
    xe : float
        The ionization fraction ne/nH.
    dE_dVdt_inj : float
        dE/dVdt, i.e. energy injection rate of DM per volume per time
    dt : float
        time in seconds over which these electrons were deposited.

    Returns
    -------
    list of floats
        Ratio of deposited energy to a given channel over energy deposited by DM.
        The order of the channels is heat, lyman, ionH, ionHe, cont
    """
    rs = spec_elec.rs
    #Fractions of energy being split off into each channel
    fracs_grid = MEDEA_interp.get_vals(xe, spec_elec.eng)

    #enforce that all functions sum to 1
    fracs_grid /= np.sum(fracs_grid, axis=1)[:, np.newaxis]

    #compute ratio of deposited divided by injected
    norm_factor = phys.nB * rs**3 / (dt * dE_dVdt_inj)
    totengList = spec_elec.eng * spec_elec.N * norm_factor
    f_elec =  np.array([
        np.sum(totengList * fracs) for fracs in np.transpose(fracs_grid)
    ])

    return np.array([f_elec[4], f_elec[1], f_elec[2], f_elec[3], f_elec[0]])
