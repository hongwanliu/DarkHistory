import sys
sys.path.append("../..")

import numpy as np
import scipy.interpolate as interp
import darkhistory.physics as phys

import os
cwd = os.getcwd()
abspath = os.path.abspath(__file__)
dir_path = os.path.dirname(abspath)
#dir_path = os.path.dirname(os.path.realpath(__file__))

#this code can certainly be optimized to make lists, rather than keeping track of 5 objects at a time
interp_heat, interp_lyman, interp_ionH, interp_ionHe, interp_lowE_photon = [], [], [], [], []

def make_interpolators():
    """Creates cubic splines that interpolate the Medea Data.  Stores them in globally defined variables so that these functions are only computed once

    Assumes that the data files are in the same directory as this script.

    Parameters
    ----------

    Returns
    -------
    """

    engs = [10.2, 13.6, 14, 30, 60, 100, 300, 3000]
    xHII = []
    heat, lyman, ionH, ionHe, lowE_photon = [
        [ [] for i in range(len(engs))] for j in range(5)
    ]
    global interp_heat, interp_lyman, interp_ionH, interp_ionHe, interp_lowE_photon

    os.chdir(dir_path)
    # load ln(data) from MEDEA files, replace ln(0) with -15 to avoid -infinities
    for i, num in enumerate(engs, start=0):
        with open('results-'+str(num)+'ev-xH-xHe_e-10-yp024.dat','r') as f:
            lines_list = f.readlines()
            if i==0:
                xHII = [np.log(float(line.split('\t')[0])) for line in lines_list[2:]]
            heat[i], lyman[i], ionH[i], ionHe[i], lowE_photon[i] = [
                [
                    np.log(max(float(line.split('\t')[k]),1.0e-15))
                    for line in lines_list[2:]
                ] for k in [1,2,3,4,5]
            ]
    os.chdir(cwd)
    engs = np.log(engs)

    heat, lyman, ionH, ionHe, lowE_photon = (
        np.array(heat), np.array(lyman), np.array(ionH), np.array(ionHe), np.array(lowE_photon)
    )

    #interpolate data, use linear interpolation to maintain the condition that all 5 functions sum up to 1
    interp_heat, interp_lyman, interp_ionH, interp_ionHe, interp_lowE_photon = [
        interp.interp2d(engs, xHII, llist.T, kind='linear') for llist in [
            heat, lyman, ionH, ionHe, lowE_photon
        ]
    ]

make_interpolators()
def compute_dep_inj_ratio(e_spectrum, xHII, tot_inj, time_step):
    """ Given an electron energy spectrum, calculate how much of that energy splits into
    heating of the IGM, lyman_alpha transitions, H ionization, He ionization, and continuum photons.

    Parameters
    ----------
    e_spectrum : Spectrum object
        spectrum of primary electrons
    xHII : float
        The ionization fraction nHII/nH.
    tot_inj : float
        dE/dVdt energy injection rate of DM per volume per time

    Returns
    -------
    list of floats
        Ratio of deposited energy to a given channel over energy deposited by DM.
        The order of the channels is heat, lyman, ionH, ionHe, lowE_photon
    """
    #enforce that all functions sum to 1
    tmpList = (heat+lyman+ionH+ionHe+lowE_photon)
    heat, lyman, ionH, ionHe, lowE_photon = (
        heat/tmpList, lyman/tmpList, ionH/tmpList, ionHe/tmpList, lowE_photon/tmpList
    )

    #compute ratio of deposited divided by injected
    norm_factor = 2 / (time_step * tot_inj)
    tmpList = e_spectrum.eng * e_spectrum.N * norm_factor
    return sum(heat*tmpList), sum(lyman*tmpList), sum(ionH*tmpList), sum(ionHe*tmpList), sum(lowE_photon*tmpList)
