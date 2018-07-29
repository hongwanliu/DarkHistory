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
interp_heat, interp_lyman, interp_ionH, interp_ionHe, interp_cont = [], [], [], [], []

def make_interpolators():
    """Creates cubic splines that interpolate the Medea Data.  Stores them in globally defined variables so that these functions are only computed once

    Assumes that the data files are in the same directory as this script.

    Parameters
    ----------

    Returns
    -------
    """

    engs = [10.2, 13.6, 14, 30, 60, 100, 300, 3000]
    #print('AHHHHHH NOOOOOO!')
    #engs = [10.2, 14, 30, 60, 100, 300, 3000]
    #engs = [14, 30, 60, 100, 300, 3000]
    #print('AHHHH YEAHHHH!')
    xHII = []
    heat, lyman, ionH, ionHe, cont = [
        [ [] for i in range(len(engs))] for j in range(5)
    ]
    global interp_heat, interp_lyman, interp_ionH, interp_ionHe, interp_cont

    os.chdir(dir_path)
    # load MEDEA files
    for i, num in enumerate(engs, start=0):
        with open('results-'+str(num)+'ev-xH-xHe_e-10-yp024.dat','r') as f:
            lines_list = f.readlines()

            # load ionization levels only once
            if i==0:
                xHII = [np.log(float(line.split('\t')[0])) for line in lines_list[2:]]

            # load deposition fractions for each energy
            heat[i], lyman[i], ionH[i], ionHe[i], cont[i] = [
                [
                    #set 0 to 10^-15 to avoid -\infty
                    np.log(max(float(line.split('\t')[k]),1.0e-15))
                    for line in lines_list[2:]
                ] for k in [1,2,3,4,5]
            ]
    os.chdir(cwd)
    engs = np.log(engs)

    heat, lyman, ionH, ionHe, cont = (
        np.array(heat), np.array(lyman), np.array(ionH), np.array(ionHe), np.array(cont)
    )

    #interpolate data, use linear interpolation to maintain the condition that all 5 functions sum up to 1
    interp_heat, interp_lyman, interp_ionH, interp_ionHe, interp_cont = [
        interp.interp2d(engs, xHII, llist.T, kind='linear') for llist in [
            heat, lyman, ionH, ionHe, cont
        ]
    ]

make_interpolators()
def compute_fs(e_spectrum, xHII, dE_dVdt_inj, time_step):
    """ Given an electron energy spectrum, calculate how much of that energy splits into
    heating of the IGM, lyman_alpha transitions, H ionization, He ionization, and continuum photons.

    Parameters
    ----------
    e_spectrum : Spectrum object
        spectrum of primary electrons
    xHII : float
        The ionization fraction nHII/nH.
    dE_dVdt_inj : float
        dE/dVdt, i.e. energy injection rate of DM per volume per time

    Returns
    -------
    list of floats
        Ratio of deposited energy to a given channel over energy deposited by DM.
        The order of the channels is heat, lyman, ionH, ionHe, cont
    """
    global interp_heat, interp_lyman, interp_ionH, interp_ionHe, interp_cont
    rs = e_spectrum.rs

    #Fractions of energy being split off into each channel
    heat, lyman, ionH, ionHe, cont = [
        np.exp(f(np.log(e_spectrum.eng),[np.log(xHII)])) for f in [
            interp_heat, interp_lyman, interp_ionH, interp_ionHe, interp_cont
        ]
    ]

    # print('************ inside lowE_electrons.compute_fs ***********')

    # print('check heat: ', heat[100:110])
    # print('check lyman: ', lyman[100:110])
    # print('check ionH: ', ionH[100:110])

    #enforce that all functions sum to 1
    tmpList = (heat+lyman+ionH+ionHe+cont)
    # print('check tmpList: ', tmpList[100:110])
    heat, lyman, ionH, ionHe, cont = (
        heat/tmpList, lyman/tmpList, ionH/tmpList, ionHe/tmpList, cont/tmpList
    )

    print('Normalized electron heat, lyman, ionH, ionHe, cont: ', 
        np.sum(heat)/heat.size, np.sum(lyman)/lyman.size, np.sum(ionH)/ionH.size, 
        np.sum(ionHe)/ionHe.size, np.sum(cont)/cont.size
    )

    # print('check 2 heat: ', heat[100:110])
    # print('check 2 lyman: ', lyman[100:110])
    # print('check 2 ionH: ', ionH[100:110])


    #compute ratio of deposited divided by injected
    norm_factor = phys.nB * rs**3 / (time_step * dE_dVdt_inj)
    tmpList = e_spectrum.eng * e_spectrum.N * norm_factor
    print('Goods: ', np.sum(tmpList))
    # print('check 2 tmpList: ', tmpList[100:110])
    # print('check e_spectrum.toteng(): ', e_spectrum.toteng())
    f_elec =  np.array([
        np.dot(cont, tmpList), 
        np.dot(lyman, tmpList), 
        np.dot(ionH, tmpList), 
        np.dot(ionHe, tmpList), 
        np.dot(heat,tmpList)
    ])
    # print('f_elec: ', f_elec)

    return f_elec
