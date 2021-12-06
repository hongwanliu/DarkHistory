import numpy as np
from math import factorial as fac
from scipy.special import loggamma 

from scipy.interpolate import interp1d 

import darkhistory.physics as phys
from darkhistory.spec.spectrum import Spectrum

from config import load_data


def g(l, lp, n, kappa=None):
    """
    Matrix element for bound-free transition. 

    Parameters
    ----------
    l : int
        The initial l-state in the hydrogen atom. 
    lp : int
        The final l-state in the continuum. 
    n : int
        The initial energy level of the hydrogen atom. 
    kappa : ndarray, optional
        The values of kappa to evaluate the matrix element at. 
    

    Returns
    -------
    float or ndarray
        Matrix element. 

    Notes
    -----
    See arXiv:0911.1359 Eq. (31) and (32) for definitions.
    """

    if np.abs(l - lp) != 1:

        raise ValueError('|l - lp| must be 1.')

    if l > n-1:

        raise ValueError('l must be less than n')

    if kappa is None:
        # If no abscissa specified, using the default kappa^2 given above. Return the table values.

        return load_data('bnd_free')['g_table_dict'][n][l][lp]

    else:

        if l+1==lp and lp==n:

            # kappa can be vectorized. 
            try: 

                _ = iter(kappa)

                g_vec = np.zeros_like(kappa)

                # Transforming product to sum of logs to avoid overflow. 

                log_one_plus_s2_k2_prod = np.sum(np.array([np.log(1 + s**2 * kappa**2) for s in np.arange(lp+1)]), axis=0)
                
                log_prefac_0 = (l - n) * np.log(2*n) + 0.5 * (loggamma(n+l+1) - loggamma(n-l) + log_one_plus_s2_k2_prod)
                log_prefac_1 = np.log(np.sqrt(np.pi/2) * 8 * n) - loggamma(2*n) + n*np.log(4*n) - 2*n 
                log_prefac_2 = -np.log(np.sqrt(1 - np.exp(-2*np.pi/kappa[kappa != 0]))) + (
                    2*n - 2 / kappa[kappa != 0] * np.arctan(n*kappa[kappa != 0])
                    - (n+2) * np.log(1. + n**2 * kappa[kappa != 0]**2)
                )

                g_vec[kappa == 0] = np.exp(log_prefac_0[kappa == 0] + log_prefac_1)
                g_vec[kappa != 0] = np.exp(log_prefac_0[kappa != 0] + log_prefac_1 + log_prefac_2)


                return g_vec

            except:

                log_one_plus_s2_k2_prod = np.sum(np.array([np.log(1 + s**2 * kappa**2) for s in np.arange(lp+1)]), axis=0)
                log_prefac_0 = (l - n) * np.log(2*n) + 0.5 * (loggamma(n+l+1) - loggamma(n-l) + log_one_plus_s2_k2_prod)
                log_prefac_1 = np.log(np.sqrt(np.pi/2) * 8 * n) - loggamma(2*n) + n*np.log(4*n) - 2*n 

                if kappa == 0:

                    return np.exp(log_prefac_0 + log_prefac_1)
                
                else:

                    log_prefac_2 = -np.log(np.sqrt(1 - np.exp(-2*np.pi/kappa))) + (
                        2*n - 2 / kappa * np.arctan(n*kappa)
                        - (n+2) * np.log(1. + n**2 * kappa**2)
                    )

                    return np.exp(log_prefac_0 + log_prefac_1 + log_prefac_2)

                

        elif l == n-2 and lp == n-1:

            fac = 1. / (2*n) * np.sqrt(1. / (n+l+1) / (n-l-1) / (1 + (lp+1)**2 * kappa**2))     

            return (2*n - 1) * (1 + n**2 * kappa**2) * n * g(n-1,n,n,kappa) * fac

        elif l == n-1 and lp == n-2:

            fac = 1. / np.sqrt((1 + (lp+1)**2 * kappa**2) * (1 + (lp+2)**2 * kappa**2))
             
            return (1 + n**2 * kappa**2) / (2*n) * g(n-1,n,n,kappa) * fac

        elif l == n-2 and lp == n-3:

            fac = 1. / (2*n) * np.sqrt(1. / (n+l+1) / (n-l-1) / (1 + (lp+1)**2 * kappa**2))
             
            return (2*n - 1) * (4 + (n-1) * (1 + n**2 * kappa**2)) * g(n-1,n-2,n,kappa) * fac

        else:

            if l < lp:

                fac_1 = 1. / (2*n) * np.sqrt(1. / (n+l+1) / (n-l-1) / (1 + (lp+1)**2 * kappa**2))
                fac_2 = 1. / (2*n)**2 * np.sqrt(
                    1. / (n+l+1) / (n+l+2) / (n-l-1) / (n-l-2) / (1 + (lp+1)**2 * kappa**2) / (1 + (lp+2)**2 * kappa**2)
                )
                

                return (
                    (4*(n**2 - (l+2)**2) + (l+2)*(2*(l+2) - 1)*(1 + n**2*kappa**2)) 
                    * g(l+1,lp+1,n,kappa=kappa) * fac_1
                    - (4*n**2 * (n**2 - (l+2)**2) * (1 + ((l+2)+1)**2 * kappa**2)) 
                    * g(l+2,lp+2,n,kappa=kappa) * fac_2
                )

            else:

                fac_1 = 1. / (2*n) * np.sqrt(1. / (n+l+1) / (n-l-1) / (1 + (lp+1)**2 * kappa**2))
                fac_2 = 1. / (2*n)**2 * np.sqrt(
                    1. / (n+l+1) / (n+l+2) / (n-l-1) / (n-l-2) / (1 + (lp+1)**2 * kappa**2) / (1 + (lp+2)**2 * kappa**2)
                )
            

                return (
                    (4*(n**2 - (l+1)**2) + (l+1)*(2*(l+1) + 1) * (1 + n**2*kappa**2)) 
                    * g(l+1,lp+1,n,kappa=kappa) * fac_1
                    - 4*n**2 * (n**2 - ((l+1)+1)**2) * (1 + (l+1)**2*kappa**2) 
                    * g(l+2,lp+2,n,kappa=kappa) * fac_2
                )

def Theta(l, lp, n, kappa=None):
    """
    Related matrix element for bound-free transition. 

    Parameters
    ----------
    l : int
        The initial l-state in the hydrogen atom. 
    lp : int
        The final l-state in the continuum. 
    n : int
        The initial energy level of the hydrogen atom. 
    kappa : ndarray, optional
        The values of kappa to evaluate the matrix element at. 
    

    Returns
    -------
    float or ndarray
        Related matrix element. 

    Notes
    -----
    See using notation in Burgess MNRAS 69, 1 (1965) Eq. (2) for definition.
    Compare these results in with Table 1 in the same paper. 
    """

    if kappa is None:

        # Use default kappa^2 binning. 

        kappa = np.sqrt(load_data('bnd_free')['kappa2_bin_edges_ary'][n])

        return (1 + n**2 * kappa**2) * load_data('bnd_free')['g_table_dict'][n][l][lp]**2

    else:

        return (1 + n**2 * kappa**2) * g(l, lp, n, kappa=kappa)**2

# Set-up the Newton-Cotes weights. We use an 11-point Newton-Cotes integration
# over each of the 50 bins. 

newton_cotes_11_weights = 5. / 299376. * np.ones(load_data('bnd_free')['n_kap']*11)
newton_cotes_11_weights[0::11] *= 16067.
newton_cotes_11_weights[1::11] *= 106300. 
newton_cotes_11_weights[2::11] *= -48525. 
newton_cotes_11_weights[3::11] *= 272400. 
newton_cotes_11_weights[4::11] *= -260550. 
newton_cotes_11_weights[5::11] *= 427368. 
newton_cotes_11_weights[6::11] *= -260550. 
newton_cotes_11_weights[7::11] *= 272400. 
newton_cotes_11_weights[8::11] *= -48525. 
newton_cotes_11_weights[9::11] *= 106300. 
newton_cotes_11_weights[10::11] *= 16067.


def I_Burgess(n, l, lp, T_m, T_r=None, f_gamma=None, stimulated_emission=True, old_rydberg=False):
    """
    Thermally averaged matrix element. 

    Parameters
    ----------
    n : int
        The initial energy level of the hydrogen atom. 
    l : int
        The initial l-state in the hydrogen atom. 
    lp : int
        The final l-state in the continuum. 
    T_m : float
        The matter temperature in eV. 
    T_r : float, optional
        The radiation temperature in eV for a blackbody distribution.
    f_gamma : function, optional
        Photon occupation number as a function of energy. 
    stimulated_emission : boolean
        If True, includes stimulated emission for a blackbody distribution. 
    old_rydberg : boolean 
        If True, sets the hydrogen ionization potential to alpha^2 me/2 instead of
        using the proton-electron reduced mass.  

    Returns
    -------
    ndarray
        Thermally averaged matrix element. 

    Notes
    -----
    See Burgess MNRAS 69, 1 (1965) Eq. (10) for definition.
    Compare these results in with Table 2 in the same paper. Currently only 
    doing the integral with default kappa values. 
    """

    if stimulated_emission:
        if T_r is not None and f_gamma is not None: 

            raise ValueError('Please use either T_r or f_gamma, not both.')

        if T_r is None and f_gamma is None:

            raise ValueError('Please use either T_r or f_gamma.')

    else:

        if f_gamma is not None: 

            raise ValueError('Please use f_gamma with stimulated emission only.')


    # For comparison with Burgess, we need to use the ionization potential assuming
    # the electron mass and not the electron-proton reduced mass. 
    if old_rydberg:

        rydb = phys.alpha**2 * phys.me / 2 

    else:
    
        rydb = phys.rydberg

    y = rydb / T_m

    # Define the integral (as a function of kappa^2).
    def integ(kappa2):

        if stimulated_emission:
            # With stimulated emission, we integrate over 1+f_gamma. 

            E_gamma = (1. / n**2 + kappa2) * rydb

            if f_gamma is not None: 

                fac = 1. + f_gamma(E_gamma)

            else:

                # Blackbody occupation number. 
                fac = 1. + np.exp(-E_gamma/T_r) / (1. - np.exp(-E_gamma/T_r))

        else:

            fac = 1. 
        
        return fac * (1. + n**2 * kappa2)**2 * np.exp(-kappa2 * y) * (
            Theta(l, lp, n, kappa=None)
        )
    
    # Multiply the computed Newton-Cotes weights above by the size of the interval. 
    weights = newton_cotes_11_weights * load_data('bnd_free')['h_ary'][n]

    # Perform the Newton-Cotes integral. 
    integral = np.dot(weights, integ(load_data('bnd_free')['kappa2_bin_edges_ary'][n]))

    return np.max((l, lp)) * y * integral


def alpha_nl(n, l, T_m, T_r=None, f_gamma=None, stimulated_emission=True):
    """
    Recombination coefficient to the nl state.  

    Parameters
    ----------
    n : int
        The final energy level of the hydrogen atom. 
    l : int
        The final l-state in the hydrogen atom. 
    T_m : float
        The matter temperature in eV. 
    T_r : float, optional
        The radiation temperature in eV for a blackbody distribution.
    f_gamma : function, optional
        Photon occupation number as a function of energy. 
    stimulated_emission : boolean
        If True, includes stimulated emission for a blackbody distribution. 

    Returns
    -------
    ndarray
        The recombination coefficient in cm^3 s^-1. 

    Notes
    -----
    See Burgess MNRAS 69, 1 (1965) Eq. (9) for definition.
    Currently only doing the integral with default kappa values. 
    """

    if stimulated_emission:

        if T_r is not None and f_gamma is not None: 

            raise ValueError('Please use either T_r or f_gamma, not both.')

        if T_r is None and f_gamma is None: 

            raise ValueError('Please use either T_r or f_gamma.')

    else:

        if f_gamma is not None: 

            raise ValueError('Please use f_gamma with stimulated emission only.')

    prefac = 2 * np.sqrt(np.pi) * phys.alpha**4 * phys.bohr_rad**2 * phys.c / 3. 

    y = phys.rydberg / T_m

    return prefac * 2 * np.sqrt(y) / n**2 * (
        I_Burgess(n, l, l-1, T_m, T_r=T_r, f_gamma=f_gamma, stimulated_emission=stimulated_emission) * (l > 0) 
        + I_Burgess(n, l, l+1, T_m, T_r=T_r, f_gamma=f_gamma, stimulated_emission=stimulated_emission)
    )

def beta_nl(n, l, T_r=None, f_gamma=None):
    """
    Photoionization rate for the nl state.  

    Parameters
    ----------
    n : int
        The final energy level of the hydrogen atom. 
    l : int
        The final l-state in the hydrogen atom. 
    T_r : float, optional
        The radiation temperature in eV for a blackbody distribution.
    f_gamma : function, optional
        Photon occupation number as a function of energy. 

    Returns
    -------
    ndarray
        The photoionization rate in s^-1. 

    Notes
    -----
    See Burgess MNRAS 69, 1 (1965) Eq. (1) for definition.
    Currently only doing the integral with default kappa values. 
    """

    if T_r is not None and f_gamma is not None: 

        raise ValueError('Please use either T_r or f_gamma, not both.')

    if T_r is None and f_gamma is None: 

        raise ValueError('Please use either T_r or f_gamma.')

    prefac = 4 * np.pi * phys.alpha * phys.bohr_rad**2 / 3 * n**2 / (2*l + 1)

    # Define the integral (as a function of kappa2 and f_gamma)
    def integ(kappa2, f_gamma):

        # The photon energy to produce an electron with energy kappa^2. 
        E_gamma = (1. / n**2 + kappa2) * phys.rydberg
        
        if f_gamma is None:

            # Blackbody occupation number. 
            f_gam_fac = np.exp(-E_gamma/T_r) / (1. - np.exp(-E_gamma/T_r)) 

        else:

            f_gam_fac = f_gamma(E_gamma)

        # Using the fact that dn_gamma / dE_gamma = (1/n^2 + K^2)^2 I_H^3 / pi^2 f
        # times a conversion to get into natural units. 

        return (kappa2 + 1/n**2)**2 * phys.rydberg**2 / np.pi**2 * f_gam_fac * (
            (l+1)*Theta(l, l+1, n, kappa=None)
            + l*Theta(l, l-1, n, kappa=None)
        ) * phys.rydberg / phys.hbar**3 / phys.c**2
    
    # Multiply the computed Newton-Cotes weights above by the size of the interval. 
    weights = newton_cotes_11_weights * load_data('bnd_free')['h_ary'][n]

    # Perform the Newton-Cotes integral. 
    integral = np.dot(weights, integ(load_data('bnd_free')['kappa2_bin_edges_ary'][n], f_gamma))

    return prefac*integral 


def alpha_B(T_m, T_r=None, f_gamma=None, stimulated_emission=True, n=100):
    """
    Case-B recombination coefficient. This is the sum of alpha_nl, n>=2.  

    Parameters
    ----------
    T_m : float
        The matter temperature in eV.  
    T_r : float, optional
        The radiation temperature in eV for a blackbody distribution.
    f_gamma : function, optional
        Photon occupation number as a function of energy. 
    stimulated_emission : boolean
        If True, includes stimulated emission for a blackbody distribution. 
    n : int
        The maximum n to include. 

    Returns
    -------
    ndarray
        The case-B recombination coefficient in cm^-3 s^-1. 

    Notes
    -----
    Currently only doing the integral with default kappa values. 
    """

    if stimulated_emission:

        if T_r is not None and f_gamma is not None: 

            raise ValueError('Please use either T_r or f_gamma, not both.')

        if T_r is None and f_gamma is None: 

            raise ValueError('Please use either T_r or f_gamma.')

    coeff = 0. 

    # Sum from nn = 2 to n. 
    for nn in 2 + np.arange(n-1):

        contrib_nn = 0

        # Sum from l=0 to nn-1. 
        for ll in np.arange(nn):

            contrib_ll = alpha_nl(nn, ll, T_m, T_r=T_r, f_gamma=f_gamma, stimulated_emission=stimulated_emission)
            contrib_nn += contrib_ll
            
        coeff += contrib_nn

        # print(nn, contrib_nn)
            
    return coeff

def beta_B(T_r, n=100):
    """
    Case-B photoionization coefficient.  

    Parameters
    ----------
    T_r : float, optional
        The radiation temperature in eV for a blackbody distribution.
    n : int
        The maximum n to include. 

    Returns
    -------
    ndarray
        The case-B photoionization rate in s^-1. 

    Notes
    -----
    See Peebles ApJ 153, 1 (1968) Eq. (26) for definition.
    Currently only doing the integral with default kappa values. 
    Assumes that the nl-states are equilibrium distributed. 
    """

    coeff = 0. 
    
    # Sum from nn = 2 to n. 
    for nn in 2 + np.arange(n-1):

        contrib_nn = 0

        # Sum from l=0 to nn-1.
        for ll in np.arange(nn):

            # 2*ll+1 comes from the sum over m substates
            contrib_ll = beta_nl(nn, ll, T_r=T_r, f_gamma=None) * (2*ll + 1) * np.exp(-phys.rydberg * (1./4. - 1./nn**2) / T_r)
            contrib_nn += contrib_ll
            
        coeff += contrib_nn

        # print(nn, contrib_nn)
            
    #hc=2*np.pi*phys.hbar*phys.c
    #lam_T = hc/(2*np.pi * phys.mu_ep * T_r)**(1/2)
    return coeff 
    # return coeff/4 #* lam_T**3 * np.exp(phys.rydberg/4 / T_r)

def gamma_nl(n, l, T_m, T_r=None, f_gamma=None, stimulated_emission=True):
    """
    Recombination photon spectrum coefficient in cm^3 eV^-1 sec^-1. 

    Parameters
    ----------
    n : int
        The initial energy level of the hydrogen atom. 
    l : int
        The initial l-state in the hydrogen atom. 
    T_m : float
        The matter temperature in eV. 
    T_r : float, optional
        The radiation temperature in eV for a blackbody distribution.
    f_gamma : function, optional
        Photon occupation number as a function of energy. 
    stimulated_emission : boolean
        If True, includes stimulated emission for a blackbody distribution. 

    Returns
    -------
    Spectrum
        Recombination photon spectrum. 

    Notes
    -----
    The number density of photons per energy per time produced by recombination to
    the nl level is n_e * n_HII * gamma_nl. 
    The photon energy abscissa is fixed to be (kappa2 + 1/n**2) * phys.rydberg, where 
    kappa2 is our precomputed kappa2 values. 
    """

    if f_gamma is not None and not stimulated_emission: 

        raise ValueError('Please use f_gamma with stimulated emission only.')

    if T_r is None and f_gamma is None:

        raise ValueError('Please use either T_r or f_gamma.')

    rydb = phys.rydberg

    y = rydb / T_m

    kappa2 = load_data('bnd_free')['kappa2_bin_edges_ary'][n]

    E_gamma = (kappa2 + 1./n**2) * rydb

    # Define the integral (as a function of kappa^2).
    if stimulated_emission:
        # With stimulated emission, we integrate over 1+f_gamma. 

        if f_gamma is not None: 

            fac_gamma = 1. + f_gamma(E_gamma)

        else:

            # Blackbody occupation number. 
            fac_gamma = 1. + np.exp(-E_gamma/T_r) / (1. - np.exp(-E_gamma/T_r))

    else:

        fac_gamma = 1.

    prefac = (
        2 * np.sqrt(np.pi) * phys.alpha**4 * phys.bohr_rad**2 * phys.c / 3. 
        * 2 * np.sqrt(y) / n**2 
        * y * (1 + n**2 * kappa2)**2 * np.exp(-kappa2 * y)
    ) / phys.rydberg

    # Remove the first few entries, which are repeated because kappa2 << 1/n^2.
    eng, ind = np.unique(E_gamma, return_index=True)
    res = prefac * fac_gamma * (
        l * Theta(l, l-1, n, kappa=None) * (l > 0) 
        + (l+1) * Theta(l, l+1, n, kappa=None)
    )

    return Spectrum(eng, res[ind],spec_type='dNdE')

def xi_nl(n, l, T_r=None, f_gamma=None):
    """
    Photoionization photon spectrum coefficient in eV^-1 sec^-1. 

    Parameters
    ----------
    n : int
        The initial energy level of the hydrogen atom. 
    l : int
        The initial l-state in the hydrogen atom.  
    T_r : float, optional
        The radiation temperature in eV for a blackbody distribution.
    f_gamma : function, optional
        Photon occupation number as a function of energy. 

    Returns
    -------
    Spectrum
        Energy abscissa and photoionization photon spectrum coefficient. 

    Notes
    -----
    The number density of photons per energy per time absorbed by photoionization from
    the nl level is n_nl * xi_nl. 
    The photon energy abscissa is fixed to be (kappa2 + 1/n**2) * phys.rydberg, where 
    kappa2 is our precomputed kappa2 values. 
    """

    if T_r is not None and f_gamma is not None: 

        raise ValueError('Please use either T_r or f_gamma, not both.')

    if T_r is None and f_gamma is None: 

        raise ValueError('Please use either T_r or f_gamma.')

    prefac = 4 * np.pi * phys.alpha * phys.bohr_rad**2 / 3 * n**2 / (2*l + 1)

    kappa2 = load_data('bnd_free')['kappa2_bin_edges_ary'][n]

    E_gamma = (1. / n**2 + kappa2) * phys.rydberg 

    if f_gamma is None:

        # Blackbody occupation number. 
        f_gam_fac = np.exp(-E_gamma/T_r) / (1. - np.exp(-E_gamma/T_r)) 

    else:

        f_gam_fac = f_gamma(E_gamma)

    d_beta_d_kappa2 = prefac * (kappa2 + 1./n**2)**2 * phys.rydberg**2 / np.pi**2 * f_gam_fac * (
        (l+1)*Theta(l, l+1, n, kappa=None)
        + l*Theta(l, l-1, n, kappa=None)
    ) * phys.rydberg / phys.hbar**3 / phys.c**2

    # Remove the first few entries, which are repeated because kappa2 << 1/n^2.
    eng, ind = np.unique(E_gamma, return_index=True)
    res = d_beta_d_kappa2 / phys.rydberg 

    return Spectrum(eng, res[ind],spec_type='dNdE')

def gamma_B(eng, T_m, T_r=None, f_gamma=None, stimulated_emission=True, n=100):
    """
    Case-B recombination spectrum coefficient, including transitions to n>=2.  

    Parameters
    ----------
    eng : ndarray
        The energy abscissa for the coefficients in eV.
    T_m : float
        The matter temperature in eV.  
    T_r : float, optional
        The radiation temperature in eV for a blackbody distribution.
    f_gamma : function, optional
        Photon occupation number as a function of energy. 
    stimulated_emission : boolean
        If True, includes stimulated emission for a blackbody distribution. 
    n : int
        The maximum n to include. 

    Returns
    -------
    Spectrum
        The case-B recombination spectrum coefficient in eV cm^-3 s^-1. 

    Notes
    -----
    Currently only doing the integral with default kappa values. 
    """

    if T_r is not None and f_gamma is not None: 

        raise ValueError('Please use either T_r or f_gamma, not both.')

    if T_r is None and f_gamma is None: 

        raise ValueError('Please use either T_r or f_gamma.')

    coeff = None 

    # Sum from nn = 2 to n. 
    for nn in 2 + np.arange(n-1):

        contrib_nn = None

        # Sum from l=0 to nn-1. 
        for ll in np.arange(nn):

            contrib_ll = gamma_nl(nn, ll, T_m, T_r, f_gamma=f_gamma, stimulated_emission=stimulated_emission)
            if contrib_nn is None: 
                contrib_nn = contrib_ll 
            else:
                contrib_nn += contrib_ll

        contrib_nn.rebin(eng)

        if coeff is None:
            
            coeff = contrib_nn 

        else:

            coeff += contrib_nn

            
    return coeff

def xi_B(eng, T_r, n = 100):
    """
    Case-B Photoionization photon spectrum coefficient in eV^-1 sec^-1, including transitions to n>=2.

    Parameters
    ----------
    eng: ndarray
        The energy abscissa for the coefficients in eV. 
    T_r : float, optional
        The radiation temperature in eV for a blackbody distribution.
    n : int
        The maximum n to include

    Returns
    -------
    Spectrum
        Energy abscissa and photoionization photon spectrum coefficient in eV^-1 sec^-1.

    Notes
    -----
    Currently only doing the integral with default kappa values. 
    Assumes that the nl-states are equilibrium distributed. 
    """

    coeff = None

    # Sum from nn = 2 to n. 
    for nn in 2 + np.arange(n-1): 

        contrib_nn = None
        
        # Sum from l=0 to nn-1. 
        for ll in np.arange(nn): 

            contrib_ll = xi_nl(nn, ll, T_r) * (2*ll + 1) * np.exp(-phys.rydberg * (1./4. - 1./nn**2) / T_r)
            if contrib_nn is None: 
                contrib_nn = contrib_ll 
            else:
                contrib_nn += contrib_ll 

        contrib_nn.rebin(eng)

        if coeff is None:
            
            coeff = contrib_nn 

        else:

            coeff += contrib_nn

    return coeff 

def generate_g_table_dict():
    """
    Generates a table of g-values from default kappa2.  

    Returns
    -------
    dict
        Dictionary indexed by [n][l][lp] of g. 

    Notes
    -----
    See arXiv:0911.1359 Eq. (31) and (32) for definitions. 
    """

    g_table_dict = {}

    for n in 1 + np.arange(300):

        g_table_dict[n] = {}

        abscissa = load_data('bnd_free')['kappa2_bin_edges_ary'][n]

        for l in np.arange(n-1, -1, step=-1):

            g_table_dict[n][l] = {}

            if l == n-1:

                g_table_dict[n][l][l+1] = g(l, l+1, n, kappa=np.sqrt(abscissa), interp=False)

                fac = 1. / np.sqrt((1 + l**2 * abscissa) * (1 + (l+1)**2 * abscissa))
             
                g_table_dict[n][l][l-1] = (1 + n**2 * abscissa) / (2*n) * g_table_dict[n][l][l+1] * fac

            elif l == n-2:


                fac_down = 1. / (2*n) * np.sqrt(1. / (n+l+1) / (n-l-1) / (1 + l**2 * abscissa))

                g_table_dict[n][l][l-1] = (2*n - 1)*(4 + (n-1)*(1 + n**2 * abscissa)) * g_table_dict[n][l+1][l] * fac_down

                fac_up = 1. / (2*n) * np.sqrt(1. / (n+l+1) / (n-l-1) / (1 + (l+2)**2 * abscissa))

                g_table_dict[n][l][l+1] = (2*n - 1) * (1 + n**2 * abscissa) * n * g_table_dict[n][l+1][l+2] * fac_up

            else:

                fac_1_down = 1. / (2*n) * np.sqrt(1. / (n+l+1) / (n-l-1) / (1 + l**2 * abscissa))
                fac_2_down = 1. / (2*n)**2 * np.sqrt(
                    1. / (n+l+1) / (n+l+2) / (n-l-1) / (n-l-2) / (1 + l**2 * abscissa) / (1 + (l+1)**2 * abscissa)
                )

                g_table_dict[n][l][l-1] = (
                    (4 * (n**2 - (l+1)**2) + (l+1)*(2*(l+1) + 1)*(1 + n**2*abscissa)) * g_table_dict[n][l+1][l] * fac_1_down
                    - 4*n**2 * (n**2 - (l+2)**2) * (1 + (l+1)**2 * abscissa) * g_table_dict[n][l+2][l+1] * fac_2_down
                )

                fac_1_up = 1. / (2*n) * np.sqrt(1. / (n+l+1) / (n-l-1) / (1 + (l+2)**2 * abscissa))
                fac_2_up = 1. / (2*n)**2 * np.sqrt(
                    1. / (n+l+1) / (n+l+2) / (n-l-1) / (n-l-2) / (1 + (l+2)**2 * abscissa) / (1 + (l+3)**2 * abscissa)
                )

                g_table_dict[n][l][l+1] = (
                    (4 * (n**2 - (l+2)**2) + (l+2)*(2*(l+2) - 1)*(1 + n**2*abscissa)) * g_table_dict[n][l+1][l+2] * fac_1_up
                    - 4*n**2 * (n**2 - (l+2)**2) * (1 + (l+3)**2 * abscissa) * g_table_dict[n][l+2][l+3] * fac_2_up
                )

        print(n)

    return g_table_dict



    
