""" Deposition of energy from low-energy electrons

    As detailed in section III.F.2 of the paper, low-energy electrons (sub-3keV electrons) deposit their energy into the IGM through hydrogen/helium ionization, hydrogen excitation, heat, and continuum photons.  To calculate how much energy is deposited into each channel we use the MEDEA results [1]_ as described in the paragraph before Eq. (45) of the paper.
  
"""

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

def make_interpolator(interp_type='2D', cross_check=False):
    """Creates cubic splines that interpolate the Medea Data.  Stores them in globally defined variables so that these functions are only computed once

    Assumes that the data files are in the same directory as this script.

    Parameters
    ----------

    interp_type : {'1D', '2D'}, optional
        Returns the type of interpolation over the MEDEA data. 

    Returns
    -------

    Interpolator2D or function
        The interpolating function (takes x_e and electron energy)
    """

    if cross_check:
        engs = np.array([14., 30, 60, 100, 300, 3000])
    else:
        engs = np.array([10.2, 13.6, 14, 30, 60, 100, 300, 3000])
    #print('AHHHHHH NOOOOOO!')

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
                    # HL: changed to 1e-4 for consistency with Tracy
                    max(float(line.split('\t')[k]),1e-4)
                    for line in lines_list[2:]
                ] for k in [1,2,3,4,5]
            ]))

    os.chdir(cwd)

    if interp_type == '2D':

        MEDEA_interp = utils.Interpolator2D(
            xes, 'xes', engs, 'engs', grid_vals, logInterp=True
        )

    elif interp_type == '1D':

        from scipy.interpolate import interp1d

        class Fake_Interpolator2D:

            def __init__(
                self, interp_log_xe_func
            ):

                self.interp_log_xe_func = interp_log_xe_func

            def get_vals(self, xe, eng):

                log_grid_vals = interp_log_xe_func(np.log(xe))
                interp_log_eng_func = interp1d(
                    np.log(engs), log_grid_vals, axis=0,
                    bounds_error=False, 
                    fill_value=(log_grid_vals[0], log_grid_vals[-1])
                )
                return np.exp(interp_log_eng_func(np.log(eng)))
        
        interp_log_xe_func = interp1d(
            np.log(xes), np.log(grid_vals), axis=0
        )

        MEDEA_interp = Fake_Interpolator2D(interp_log_xe_func)

    else:

        raise TypeError('Invalid interp_type.')

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
