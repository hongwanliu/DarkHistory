import sys
sys.path.append("../..")

import numpy as np
import scipy.interpolate as interp

import matplotlib.pyplot as plt
import physics as phys

#this code can certainly be optimized to make lists, rather than keeping track of 5 objects at a time
interp_heat, interp_lyman, interp_ionH, interp_ionHe, interp_lowE_photon = [], [], [], [], []

def make_interpolators():
    """Creates cubic splines that interpolate the Medea Data.  Stores them in globally defined variables so that these functions are only computed once

    Parameters
    ----------

    Returns
    -------
    """

    engs = [14, 30, 60, 100, 300, 3000]
    xHII = []
    heat, lyman, ionH, ionHe, lowE_photon = [ [ [] for i in range(len(engs))]
                                             for j in range(5)]
    global interp_heat, interp_lyman, interp_ionH, interp_ionHe, interp_lowE_photon

    # load ln(data) from MEDEA files, replace ln(0) with -15 to avoid -infinities
    for i, num in enumerate(engs, start=0):
        with open('results-'+str(num)+'ev-xH-xHe_e-10-yp024.dat','r') as f:
            lines_list = f.readlines()
            if i==0:
                xHII = [np.log(float(line.split('\t')[0])) for line in lines_list[2:]]
            heat[i], lyman[i], ionH[i], ionHe[i], lowE_photon[i] = [[np.log(max(float(line.split('\t')[k]),1.0e-15))
                                                                     for line in lines_list[2:]] for k in [1,2,3,4,5]]
    engs = np.log(engs)

    heat, lyman, ionH, ionHe, lowE_photon = np.array(heat), np.array(lyman), np.array(ionH), np.array(ionHe), np.array(lowE_photon)

    #interpolate data, use linear interpolation to maintain the condition that all 5 functions sum up to 1
    interp_heat = interp.interp2d(engs,xHII,heat.T, kind='linear')
    interp_lyman = interp.interp2d(engs,xHII,lyman.T, kind='linear')
    interp_ionH = interp.interp2d(engs,xHII,ionH.T, kind='linear')
    interp_ionHe = interp.interp2d(engs,xHII,ionHe.T, kind='linear')
    interp_lowE_photon = interp.interp2d(engs,xHII,lowE_photon.T, kind='linear')

make_interpolators()
def compute_dep_inj_ratio(e_spectrum, xHII, rs, tot_inj):
    """ Needs a description

    Parameters
    ----------
    e_spectrum : Spectrum object
        spectrum of primary electrons
    xHII : float
        The ionization fraction nHII/nH.
    rs : float
        redshift
    tot_inj : float
        total energy injected by DM

    Returns
    -------
    float, x5
        Ratio of deposited energy to a given channel over energy deposited by DM.
        The order of the channels is heat, lyman, ionH, ionHe, lowE_photon
    """
    global interp_heat, interp_lyman, interp_ionH, interp_ionHe, interp_lowE_photon

    heat = np.exp(interp_heat(np.log(e_spectrum.eng),[np.log(xHII)]))
    lyman = np.exp(interp_lyman(np.log(e_spectrum.eng),[np.log(xHII)]))
    ionH = np.exp(interp_ionH(np.log(e_spectrum.eng),[np.log(xHII)]))
    ionHe = np.exp(interp_ionHe(np.log(e_spectrum.eng),[np.log(xHII)]))
    lowE_photon = np.exp(interp_lowE_photon(np.log(e_spectrum.eng),[np.log(xHII)]))

    #enforce that all functions sum to 1
    tmpList = (heat+lyman+ionH+ionHe+lowE_photon)
    heat, lyman, ionH, ionHe, lowE_photon = heat/tmpList, lyman/tmpList, ionH/tmpList, ionHe/tmpList, lowE_photon/tmpList

    #compute ratio of deposited divided by injected
    tmpList = e_spectrum.eng*e_spectrum.N/tot_inj
    return sum(heat*tmpList), sum(lyman*tmpList), sum(ionH*tmpList), sum(ionHe*tmpList), sum(lowE_photon*tmpList)
