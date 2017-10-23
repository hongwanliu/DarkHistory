"""Series for computation of ICS spectrum without quadrature."""

import numpy as np 
import scipy.special as sp 

class SeriesError:


def F1(a,b,epsrel=1e-10):
    """Definite integral of x/[(exp(x) - 1)]. 

    Parameters
    ----------
    a : ndarray
        Lower limit of integration. 
    b : ndarray
        Upper limit of integration.
    err : float
        Error associated with series expansion. 

    Returns
    -------
    float
        The resulting integral.

    Note
    ----
    For a or b > 0.01, the exact analytic expression is used, whereas below that we use a series expansion. This avoids numerical errors due to computation of log(1 - exp(-x)) and likewise in the `spence` function. Note that `scipy.special.spence` can only take `float64` numbers, so downcasting is necessary in the x > 0.01 limit.
    """

    if x < 0.01: 
        integral = (
            (b-a) - (b**2 - a**2)/4 + (b**3 - a**3)/36 
            - (b**5 - a**5)/3600 + (b**7 - a**7)/211680 
            - (b**9 - a**9)/10886400
        )
        err = (b**11 - a**11)/526901760
        if err/integral > epsrel:
        raise RuntimeError('Relative error in series too large.')
        print('Series error is: ', err)
        print('Relative error required is: ', epsrel)
    else:
        integral = (
            b*np.log(1. - np.exp(-b)) - a*np.log(1. - np.exp(-a))
            - sp.spence(np.array(1. - np.exp(-b), dtype='float64'))
            + sp.spence(np.array(1. - np.exp(-a), dtype='float64'))
        )
    return integral

    


