""" Utilities for Neural Network transfer functions.
"""

import numpy as np


####################
### CONSTANTS

EPSILON = 1e-100
LOG_EPSILON = np.log(EPSILON)

XMAX = (np.tanh(+4)+1)/2
XMIN = (np.tanh(-5)+1)/2

RS_NODES = [40, 1600]


####################
### UTILITIES

def scale_to_E(eng, N, i_st, i_ed, E_target):
    """ Rescales part of N spectrum such that the total energy is E_target. """
    E_part = np.dot(eng[i_st:i_ed+1], N[i_st:i_ed+1])
    E_tot  = np.dot(eng, N)
    E_part_target = E_part + E_target - E_tot
    N[i_st:i_ed+1] *= (E_part_target/E_part)

def ics_pred_Eout_max(Ein, TF_type): # Eout(Ein)
    """ Estimate max output energy in the ICS transfer functions as a function of input energy. """
    x = np.log10(Ein)
    if TF_type == 'ics_thomson':
        p = [1.07789781e-03, 1.32714060e+00, 9.95665255e-01, 7.48920711e-04, 1.13092342e+00]
        y = (1+p[0]*np.exp(p[1]*x))/(p[2]+p[3]*np.exp(p[4]*x))
    elif TF_type == 'ics_engloss':
        p = [5.72857939e-08, -1.53672737e-06, -1.01678957e-05, 2.00363041e-04, 1.09665195e-03, -3.42969564e-03, -1.64671173e-02,  5.13922029e-01, -1.38282978e+00]
        y = np.poly1d(p)(x)
    elif TF_type == 'ics_rel':
        p = [1.97808661, -9.73260386]
        y = np.minimum(np.poly1d(p)(x), x)
    else:
        raise ValueError('Invalid TF_type.')
    return 10**y

def distortion_zero_est(rs):
    """ Estimate location of distortion zero in the highengphot transfer functions (hep_p12) as a function of redshift rs. """
    p = np.array([-0.0082342 ,  0.21588987,  3.68529367])
    return int(np.round(np.exp(np.polyval(p, np.log(rs)))))

def distortion_zero(a, iz):
    """ Find the distortion zero in highengphot transfer functions (hep_p12) based on estimate. """
    if a[iz] < a[iz-1]:
        return iz + next((offset for offset in range(5) if a[iz+offset] < a[iz+offset+1]), 0)
    else:
        return iz - next((offset for offset in range(1,5) if a[iz-offset] < a[iz-offset-1]), 0)