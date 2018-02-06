import sys
sys.path.append("../..")

import numpy as np
import scipy.interpolate as interp
from scipy.integrate import quad

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import physics as phys
import spec.spectools as spectools
from spec.spectrum import Spectrum

interp_heat, interp_lyman, interp_ionH, interp_ionHe, interp_lowE_photon = [], [], [], [], []

def make_interpolators():
    """Creates cubic splines that interpolate the Medea Data.  Stores them in globally defined variables so that these functions are only computed once

    Parameters
    ----------

    Returns
    -------
    """

    # 6 energy values: 14, 30, 60, 100, 300, 3000eV
    # 26 x_HII values: ranging from 0 to 1 (non-linearly spaced)
    engs = [14, 30, 60, 100, 300, 3000]
    xHII = []
    heat, lyman, ionH, ionHe, lowE_photon = [ [ [] for i in range(len(engs))]
                                             for j in range(5)]
    # load ln(data) from MEDEA files, replace ln(0) with -15
    for i, num in enumerate(engs, start=0):
        with open('results-'+str(num)+'ev-xH-xHe_e-10-yp024.dat','r') as f:
            lines_list = f.readlines()
            if i==0:
                xHII = [np.log(float(line.split('\t')[0])) for line in lines_list[2:]]
            heat[i], lyman[i], ionH[i], ionHe[i], lowE_photon[i] = [[np.log(max(float(line.split('\t')[k]),1.0e-15)) for line in lines_list[2:]] for k in [1,2,3,4,5]]

    heat, lyman, ionH, ionHe, lowE_photon = np.array(heat), np.array(lyman), np.array(ionH), np.array(ionHe), np.array(lowE_photon)

    global interp_heat, interp_lyman, interp_ionH, interp_ionHe, interp_lowE_photon
    #interpolate data, use linear interpolation to maintain the condition that all 5 functions sum up to 1
    engs = np.log(engs)
    interp_heat = interp.interp2d(engs,xHII,heat.T, kind='linear')
    #interp_heat = interp.RegularGridInterpolator((engs,xHII),heat) #Different interpolator
    interp_lyman = interp.interp2d(engs,xHII,lyman.T, kind='linear')
    interp_ionH = interp.interp2d(engs,xHII,ionH.T, kind='linear')
    interp_ionHe = interp.interp2d(engs,xHII,ionHe.T, kind='linear')
    interp_lowE_photon = interp.interp2d(engs,xHII,lowE_photon.T, kind='linear')

make_interpolators()
def compute_dep_inj_ratio(xHII):
    """ Needs a description

    Parameters
    ----------
    e_spectrum : Spectrum object
        spectrum of primary electrons
    xHII : float
        The ionization fraction nHII/nH.
    rs : float
        redshift

    Returns
    -------
    f_Heat, f_Lyman, f_ionH, f_ionHe, f_lowEnergy_photon : float
        ratio of deposited energy to a given channel over energy deposited by DM
    """
    global interp_heat, interp_lyman, interp_ionH, interp_ionHe, interp_lowE_photon
    T=1.5
    def CMB_dNdE_at_T(eng):
        return phys.CMB_spec(eng,T)
    eng = 10**((np.arange(120)-90)*(1/10))
    discrete_CMB = spectools.discretize(eng,CMB_dNdE_at_T)
    discrete_CMB.rs = T/phys.TCMB(1)
    e_spectrum = Spectrum(discrete_CMB.eng, discrete_CMB.dNdE, discrete_CMB.rs)

    #tmpList = [heat, lyman, ionH, ionHe, lowE_photon]
    depList = [[], [], [], [], []]
    for i, f in enumerate([interp_heat, interp_lyman, interp_ionH, interp_ionHe, interp_lowE_photon]):
        tmpList[i] = np.exp( f(np.log(e_spectrum.eng),[np.log(xHII)]))
    """
    heat = np.exp(interp_heat(np.log(e_spectrum.eng),[np.log(xHII)]))
    lyman = np.exp(interp_lyman(np.log(e_spectrum.eng),[np.log(xHII)]))
    ionH = np.exp(interp_ionH(np.log(e_spectrum.eng),[np.log(xHII)]))
    ionHe = np.exp(interp_ionHe(np.log(e_spectrum.eng),[np.log(xHII)]))
    lowE_photon = np.exp(interp_lowE_photon(np.log(e_spectrum.eng),[np.log(xHII)]))
    """

    #enforce that all functions sum to 1
    tmpList = (heat+lyman+ionH+ionHe+lowE_photon)
    heat, lyman, ionH, ionHe, lowE_photon = heat/tmpList, lyman/tmpList, ionH/tmpList, ionHe/tmpList, lowE_photon/tmpList

    #heat, lyman, ionH, ionHe, lowE_photon = heat*e_spectrum*, lyman/sumList, ionH/sumList, ionHe/sumList, lowE_photon/sumList

def troubleshoot():
    engs = [14, 30, 60, 100, 300, 3000]
    xHII = []
    heat, lyman, ionH, ionHe, lowE_photon = [ [ [] for i in range(len(engs))]
                                             for j in range(5)]

    for i, num in enumerate(engs, start=0):
        with open('results-'+str(num)+'ev-xH-xHe_e-10-yp024.dat','r') as f:
            lines_list = f.readlines()
            if i==0:
                xHII = [np.log(float(line.split('\t')[0])) for line in lines_list[2:]]
            heat[i], lyman[i], ionH[i], ionHe[i], lowE_photon[i] = [[np.log(max(float(line.split('\t')[k]),10e-15)) for line in lines_list[2:]] for k in [1,2,3,4,5]]

    heat, lyman, ionH, ionHe, lowE_photon = np.array(heat), np.array(lyman), np.array(ionH), np.array(ionHe), np.array(lowE_photon)
    print(np.exp(heat)+np.exp(lyman)+np.exp(ionH)+np.exp(ionHe)+np.exp(lowE_photon))
    global interp_heat, interp_lyman, interp_ionH, interp_ionHe, interp_lowE_photon
    engs = np.log(engs)

    x = np.linspace(engs[0],engs[-1],100)
    y = np.linspace(xHII[0],xHII[-1],100)
    #points = np.array( [[[a,b] for a in x] for b in y ])
    #points = np.array([x,y]).T
    #z = interp_heat(points) #When using RegularGridInterpolator
    z = interp_heat(x,y)
    x, y = np.meshgrid(x,y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x,y,z)

    engs, xHII = np.meshgrid(engs,xHII)
    engs = engs.flatten()
    xHII = xHII.flatten()
    heat = heat.T.flatten()
    ax.scatter(engs, xHII, heat, c='red', marker='.', lw=0)

    ax.set_xlabel("ln(E) (eV)")
    ax.set_ylabel("ln(x_HII)")
    ax.set_zlabel("ln(Deposition) (%)")
    plt.show()

    #Do the interpolants sum up to 1?
    T=1.5
    def CMB_dNdE_at_T(eng):
        return phys.CMB_spec(eng,T)
    eng = 10**((np.arange(120)-90)*(1/10))
    discrete_CMB = spectools.discretize(eng,CMB_dNdE_at_T)
    discrete_CMB.rs = T/phys.TCMB(1)
    test_CMB = Spectrum(discrete_CMB.eng, discrete_CMB.dNdE, discrete_CMB.rs)

    heat = np.exp(interp_heat(np.log(test_CMB.eng),[np.log(xHII)]))
    lyman = np.exp(interp_lyman(np.log(test_CMB.eng),[np.log(xHII)]))
    ionH = np.exp(interp_ionH(np.log(test_CMB.eng),[np.log(xHII)]))
    ionHe = np.exp(interp_ionHe(np.log(test_CMB.eng),[np.log(xHII)]))
    lowE_photon = np.exp(interp_lowE_photon(np.log(test_CMB.eng),[np.log(xHII)]))
    sumList = (heat+lyman+ionH+ionHe+lowE_photon)
    heat, lyman, ionH, ionHe, lowE_photon = heat/sumList, lyman/sumList, ionH/sumList, ionHe/sumList, lowE_photon/sumList
    #print(heat+lyman+ionH+ionHe+lowE_photon)
    #print(heat,lyman,ionH,ionHe,lowE_photon)

"""
    print(len(engs),len(xHII),len(heat))
    print(np.exp(engs),"\n")
    print(np.exp(xHII),"\n")
    print(np.exp(heat),"\n")
    return interp_heat, interp_lyman, interp_ionH, interp_ionHe, interp_lowE_photon
    print("xHII: ", xHII,"\n")
    print("heat: ", heat[5],"\n")
    print("lyman: ", lyman[5], "\n")
    print("ionH: ", ionH[5], "\n")
    print("ionHe: ", ionHe[5], "\n")
    print("lowE_photon: ", lowE_photon[5], "\n")
"""
#Should I be worried that there are fractions of particles in some bins?
#To avoid -inf, I replace ln(0) with ln(1e-15)
#The sum of fractions (given by the interpolators) should add up to 1, but they're a little off
