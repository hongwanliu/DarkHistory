""" Physics functions as well as constants.

Throughout DarkHistory, we choose cm, s and eV as our system of units. Masses and temperatures are also given in eV. Particle Data Group central values [1]_ are used for constants, while cosmological parameters are set to the central values of the Planck 2018 baseline TT,TE,EE+lowE+lensing [2]_. 

"""

import pickle
import sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import zeta

# from config import data_path
from config import load_data

#############################################################################
#############################################################################
# Fundamental Particles and Constants                                       #
#############################################################################
#############################################################################

mp          = 0.938272081e9
"""Proton mass in eV."""
me          = 510998.9461
"""Electron mass in eV."""
mHe         = 3.97107*mp
"""Helium nucleus mass in eV."""
hbar        = 6.58211951e-16
"""hbar in eV s."""
c           = 299792458e2
"""Speed of light in cm s\ :sup:`-1`\ ."""
kB          = 8.6173324e-5
"""Boltzmann constant in eV K\ :sup:`-1`\ ."""
alpha       = 1/137.035999139
"""Fine structure constant."""
ele         = 1.60217662e-19
"""Electron charge in coulombs."""
G = 6.6730e-8
"""Newton's Gravitational Constant in cm\ :sup:`3` g\ :sup:`-1` s\ :sup:`-2`\ ."""

mass = {
    'e': me,         'mu': 105.6583745e6, 'tau': 1776.86e6,
    'c': 1.275e9  ,  'b' :   4.18e9,        't': 173.1e9,
    'W': 80.379e9  , 'Z' :  91.1876e9,      'h': 125.18e9
}
"""Masses of Standard Model particles."""
thomson_xsec = 6.652458734e-25
"""Thomson cross section in cm\ :sup:`2`\ ."""
stefboltz    = np.pi**2 / (60 * (hbar**3) * (c**2))
"""Stefan-Boltzmann constant in eV\ :sup:`-3` cm\ :sup:`-2` s\ :sup:`-1`\ .
"""
ele_rad      = hbar * c * alpha / me
"""Classical electron radius in cm."""
ele_compton  = 2*np.pi*hbar * c / me
"""Electron Compton wavelength in cm."""

#############################################################################
#############################################################################
# Cosmology                                                                 #
#############################################################################
#############################################################################

#########################################
# Densities and Hubble                  #
#########################################

h    = 0.6736
""" h parameter."""
H0   = 100*h*3.241e-20
""" Hubble parameter today in s\ :sup:`-1`\ ."""

omega_m      = 0.3153
""" Omega of all matter today."""
omega_rad    = 8e-5
""" Omega of radiation today."""
omega_lambda = 0.6847
""" Omega of dark energy today."""
omega_baryon = 0.02237/(h**2)
""" Omega of baryons today."""
omega_DM      = 0.1200/(h**2)
""" Omega of dark matter today."""
rho_crit     = 1.05371e4*(h**2)
""" Critical density of the universe in eV cm\ :sup:`-3`\ . 

See [1] for the definition. This is a mass density, with mass measured in eV.
"""
rho_DM       = rho_crit*omega_DM
""" DM density in eV cm\ :sup:`-3`\ ."""
rho_baryon   = rho_crit*omega_baryon
""" Baryon density in eV cm\ :sup:`-3`\ ."""
nB          = rho_baryon/mp
""" Baryon number density in eV cm\ :sup:`-3`\ ."""
s0          = 2891.2
""" Entropy density of CMB in cm\ :sup:`-3`\ ."""

YHe         = 0.245
"""Helium abundance by mass."""
nH          = (1-YHe)*nB
""" Atomic hydrogen number density in cm\ :sup:`-3`\ ."""
nHe         = (YHe/4)*nB
""" Atomic helium number density in cm\ :sup:`-3`\ ."""
nA          = nH + nHe
""" Hydrogen and helium number density in cm\ :sup:`-3`\ .""" 
chi         = nHe/nH
"""Ratio of helium to hydrogen nuclei."""


def hubble(rs, H0=H0, omega_m=omega_m, omega_rad=omega_rad, omega_lambda=omega_lambda):
    """ Hubble parameter in s\ :sup:`-1`\ .

    Assumes a flat universe.

    Parameters
    ----------
    rs : float
        The redshift of interest.
    H0 : float
        The Hubble parameter today, default value `H0`.
    omega_m : float, optional
        Omega matter today, default value `omega_m`.
    omega_rad : float, optional
        Omega radiation today, default value `omega_rad`.
    omega_lambda : float, optional
        Omega dark energy today, default value `omega_lambda`.

    Returns
    -------
    float
    """


    return H0*np.sqrt(omega_rad*rs**4 + omega_m*rs**3 + omega_lambda)

def dtdz(rs, H0=H0, omega_m=omega_m, omega_rad=omega_rad, omega_lambda=omega_lambda):
    """ dt/dz in s.

    Parameters
    ----------
    rs : float
        The redshift of interest.
    H0 : float
        The Hubble parameter today, default value `H0`.
    omega_m : float, optional
        Omega matter today, default value `omega_m`.
    omega_rad : float, optional
        Omega radiation today, default value `omega_rad`.
    omega_lambda : float, optional
        Omega dark energy today, default value `omega_lambda`.

    Returns
    -------
    float
    """

    return -1./(rs*hubble(rs, H0, omega_m, omega_rad, omega_lambda))

#########################################
# CMB                                   #
#########################################

def TCMB(rs):
    """ CMB temperature in eV.

    Parameters
    ----------
    rs : float
        Redshift (1+z).

    Returns
    -------
    float
    """

    fac = 2.7255
    return fac * kB * rs

def CMB_spec(eng, temp):
    """CMB spectrum in number of photons cm\ :sup:`-3` eV\ :sup:`-1`\ .

    The normalization used here is

    .. math::

        \\frac{dN}{dE \\, dV} = \\frac{E^2}{\\pi^2 (\\hbar c)^3} \\frac{1}{e^{E/T_\\text{CMB}} - 1} \\,,

    with T\ :sub:`CMB` given in eV. Returns zero if E > 500 T\ :sub:`CMB`.

    Parameters
    ----------
    temp : float
        Temperature of the CMB in eV.
    eng : float or ndarray
        Energy of the photon in eV.

    Returns
    -------
    ndarray
        The number density of photons per unit energy in eV.

    """
    prefactor = 8*np.pi*(eng**2)/((ele_compton*me)**3)
    if isinstance(eng, (list, np.ndarray)):
        small = eng/temp < 1e-10
        expr = np.zeros_like(eng)
        if np.any(small):
            expr[small] = prefactor[small]*1/(
                eng[small]/temp + (1/2)*(eng[small]/temp)**2
                + (1/6)*(eng[small]/temp)**3
            )
        if np.any(~small):
            expr[~small] = (
                prefactor[~small]*np.exp(-eng[~small]/temp)
                /(1 - np.exp(-eng[~small]/temp))
            )
    else:
        expr = 0
        if eng/temp < 1e-10:
            expr = prefactor*1/(
                eng/temp + (1/2)*(eng/temp)**2 + (1/6)*(eng/temp)**3
            )
        else:
            expr = prefactor*np.exp(-eng/temp)/(1 - np.exp(-eng/temp))

    return expr

def CMB_N_density(T):
    """ CMB number density in cm\ :sup:`-3`\ .

    Parameters
    ----------
    T : float or ndarray
        CMB temperature.

    Returns
    -------
    float or ndarray
        The number density of the CMB.

    """
    zeta_4 = np.pi**4/90

    return 4*stefboltz/c*T**3*(zeta(3)/(3*zeta_4))

def CMB_eng_density(T):
    """CMB energy density in eV cm\ :sup:`-3`\ .

    Parameters
    ----------
    T : float or ndarray
        CMB temperature.

    Returns
    -------
    float or ndarray
        The energy density of the CMB.
    """

    return 4*stefboltz/c*T**4

#########################################
# Dark Matter                           #
#########################################

def inj_rate(inj_type, rs, mDM=None, sigmav=None, lifetime=None):
    """ Dark matter annihilation/decay energy injection rate.

    Parameters
    ----------
    inj_type : {'swave', 'pwave', 'decay'}
        Type of injection.
    rs : float
        The redshift of injection.
    mDM : float, optional
        DM mass in eV.
    sigmav : float, optional
        Annihilation cross section in cm\ :sup:`-3`\ s\ :sup:`-1`\ . in the case of inj_type='pwave', sigmav = (sigma v)_ref, as in 1604.02457.
    lifetime : float, optional
        Decay lifetime in s.

    Returns
    -------
    float
        The dE/dV_dt injection rate in eV cm\ :sup:`-3`\ s\ :sup:`-1`\ .

    """

    if inj_type == 'swave' or inj_type == 'pwave':

        if sigmav is None:
            raise ValueError('sigmav must be specified for inj_type \'swave\' or \'pwave\'')

    if inj_type == 'decay':

        if lifetime is None:
            raise ValueError('lifetime must be specified for inj_type \'decay\'')

    if mDM is None:

        raise ValueError('Must specify dark matter mass.')

    if inj_type == 'swave':
        return rho_DM**2*rs**6*sigmav/mDM
    elif inj_type == 'decay':
        return rho_DM*rs**3/lifetime
    elif inj_type == 'pwave':
        sigmav_ref = sigmav

        #A reference 1D dispersion used to fix the prefactor in <sigma v>, 
        # conventionally set to dispersion of WIMPS in Milky Way sized halo
        sigma_1D_ref_over_c = 1e7/c

        #1D dispersion of unclustered DM today
        sigma_1D_B_over_c = 1e-11*(1/100)**0.5

        return rho_DM**2*rs**8*sigmav_ref/mDM*(sigma_1D_B_over_c/sigma_1D_ref_over_c)**2

# Create interpolation with structure formation data. 
# if 'pytest' not in sys.modules and 'readthedocs' not in sys.modules:
#     struct_data = np.loadtxt(open(data_path+'/boost_Einasto_subs.txt', 'rb'))

#     log_struct_interp = interp1d(
#         np.log(struct_data[:,0]), np.log(struct_data[:,1]),
#         bounds_error=False, fill_value=(np.nan, 0.)
#     )

def struct_boost_func(model='einasto_subs', model_params=None):
    """Structure formation boost factor 1+B(z).

    Parameters
    ----------
    model : {'einasto_subs', 'einasto_no_subs', 'NFW_subs', 'NFW_no_subs', 'erfc', 'pwave_NFW_no_subs'}
        Model to use. See 1604.02457. 
    model_params : tuple of floats
        Model parameters (b_h, delta, z_h) for 'erfc' option. 

    Returns
    -------
    float or ndarray
        Boost factor. 

    Notes
    -----
    Refer to 1408.1109 for erfc model, 1604.02457 for all other model
    descriptions and parameters.

    """

    if model == 'erfc':

        from scipy.special import erfc

        if model_params is None:
            # Smallest boost in 1408.1109. 
            b_h   = 1.6e5
            delta = 1.54
            z_h   = 19.5
        else:
            b_h   = model_params[0]
            delta = model_params[1]
            z_h   = model_params[2]

        def func(rs):

            return 1. + b_h / rs**delta * erfc( rs/(1+z_h) )

    else:

        struct_data = load_data('struct')[model]
        log_struct_interp = interp1d(
            np.log(struct_data[:,0]), np.log(struct_data[:,1]),
            bounds_error=False, fill_value=(np.nan, 0.)
        )

        def func(rs):

            return np.exp(log_struct_interp(np.log(rs)))

    return func



#########################################
# Other Cosmology Functions             #
#########################################

def get_optical_depth(rs_vec, xe_vec):
    """Computes the optical depth given an ionization history.

    Parameters
    ----------
    rs_vec : ndarray
        Redshift (1+z).
    xe_vec : ndarray
        Free electron fraction xe = ne/nH.

    Returns
    -------
    float
        The optical depth.
    """
    from darkhistory.spec.spectools import get_log_bin_width

    rs_log_bin_width = get_log_bin_width(rs_vec)
    abs_dtdz_vec = -dtdz(rs_vec)

    return np.dot(
        xe_vec*nH*thomson_xsec*c*abs_dtdz_vec,
        rs_vec**4*rs_log_bin_width
    )

#############################################################################
#############################################################################
# Atomic and Optical Physics                                                #
#############################################################################
#############################################################################

#########################################
# Hydrogen                              #
#########################################

rydberg      = 13.60569253
"""Ionization potential of ground state hydrogen in eV."""
lya_eng      = rydberg*3/4
"""Lyman alpha transition energy in eV."""
lya_freq     = lya_eng / (2*np.pi*hbar)
"""Lyman alpha transition frequency in Hz."""
width_2s1s_H = 8.22458
"""Hydrogen 2s to 1s decay width in s\ :sup:`-1`\ ."""
bohr_rad     = (hbar*c) / (me*alpha)
"""Bohr radius in cm."""
HI_exc_eng = {'2s' : lya_eng, '2p' : lya_eng, 
        '3p' : 12.094, '4p' : 12.755, '5p' : 13.061, 
        '6p' : 13.228, '7p' : 13.328, '8p' : 13.393, 
        '9p' : 13.438, '10p' : 13.470}
#See Stone, Kim, and Desclaux, [J. Res. Natl. Inst. Stand. Technol. 107, 327–337 (2002)]
"""HI n=1 to n=2s and 2p through 10p excitation energies in eV"""

#########################################
# Helium                                #
#########################################

He_ion_eng   = 24.5873891
"""Energy needed to singly ionize neutral He in eV."""

He_exc_lambda = {
    '23s': 1./159855.9745,
    '21s': 1./166277.4403,
    '23p': 1./169087.,                  # Approximate for J=0,1,2
    '21p': 1./171134.8970
}
"""HeI n=1 to n=2 excitation wavelength in cm."""

He_exc_eng = {
    #n^(2S+1)L
    '23s': 2*np.pi*hbar*c*159855.9745,
    '21s': 2*np.pi*hbar*c*166277.4403,
    '23p': 2*np.pi*hbar*c*169087.,      # Approximate for J=0,1,2
    '21p': 2*np.pi*hbar*c*171134.8970, '2p' : 2*np.pi*hbar*c*171134.8970,
    #nl^(2S+1)L, n is the excited state of the excited electron, 
    #l is its orbital angular momentum, 
    #(2S+1) is the total spin multiplicity, 
    #and L is the total angular momentum of the two electrons.
    '2p1P' : 21.218,
    '3p1P' : 23.087,
    '4p1P' : 23.742,
    '5p1P' : 24.046,
    '6p1P' : 24.211,
    '7p1P' : 24.311,
    '8p1P' : 24.375,
    '9p1P' : 24.420,
    '10p1P': 24.452
}
"""HeI n=1 to n=2 excitation energies in eV, and n=1 to 3p through 10p"""

A_He_21p = 1.798287e9    
"""Einstein coefficient for 2\ :sup:`1`\ p :math:`\\to` 1s decay in s\ :sup:`-1`\ ."""
A_He_23P1 = 177.58        
"""Einstein coefficient for 2\ :sup:`3`\ P\ :sub:`1` :math:`\\to` 1s decay in s\ :sup:`-1`\ ."""
width_21s_1s_He = 51.3 
"""Width of He 2\ :sup:`1`\ s :math:`\\to` 1s decay in s\ :sup:`-1`\ ."""

#########################################
# Recombination/Ionization              #
#########################################

def alpha_recomb(T_m, species):
    """Case-B recombination coefficient.

    Parameters
    ----------
    T_m : float
        The matter temperature (eV).
    species : {'HI', 'HeI_21s', 'HeI_23s'}
        The species of interest. 

    Returns
    -------
    float
        Case-B recombination coefficient in cm\ :sup:`3`\ s\ :sup:`-1`\ .

    Notes
    -----
    For HeI, returns beta with respect to the 2s state,
    in agreement with convention in RECFAST.
    """

    if species == 'HI':

        # Fudge factor recommended in 1011.3758
        fudge_fac = 1.125
        # fudge_fac = 1.14

        conv_fac = 1.0e-4/kB

        return (
            fudge_fac * 1.0e-13 * 4.309 * (conv_fac*T_m)**(-0.6166)
            / (1 + 0.6703 * (conv_fac*T_m)**0.5300)
        )

    else:

        if species == 'HeI_21s':

            q = 10**-16.744
            p = 0.711
            T_1 = 10**5.114
            T_2 = 3.

        elif species == 'HeI_23s':

            q = 10**-16.306
            p = 0.761
            T_2 = 3 # in K
            T_1 = 10**5.114

        else:

            raise TypeError('invalid species.')

        T_in_K = T_m/kB

        return 1e6 * q / (
            np.sqrt(T_in_K/T_2) 
            * (1 + T_in_K/T_2)**(1-p) 
            * (1 + T_in_K/T_1)**(1+p)
        )

    

def beta_ion(T_rad, species):
    """Case-B photoionization coefficient.

    Parameters
    ----------
    T_rad : float
        The radiation temperature.
    species : {'HI', 'HeI_21s', 'HeI_23s'}
        The relevant species.

    Returns
    -------
    float
        Case-B photoionization coefficient in s\ :sup:`-1`\ .

    Notes
    -----
    For HeI, returns beta with respect to the 2s state,
    in agreement with convention in RECFAST.

    """
    de_broglie_wavelength = (
        c * 2*np.pi*hbar
        / np.sqrt(2 * np.pi * me * T_rad)
    )
    
    if species == 'HI':
        return (
            (1/de_broglie_wavelength)**3
            * np.exp(-rydberg/4/T_rad) * alpha_recomb(T_rad, 'HI')
        )/4

    elif species == 'HeI_21s':
        E_21s_inf = He_ion_eng - He_exc_eng['21s']
        # print(E_21s_inf)
        # print(de_broglie_wavelength)
        return 4*(
            (1/de_broglie_wavelength)**3
            * np.exp(-E_21s_inf/T_rad) * alpha_recomb(T_rad, 'HeI_21s')
        )

    elif species == 'HeI_23s':
        E_23s_inf = He_ion_eng - He_exc_eng['23s']
        return (4/3)*(
            (1/de_broglie_wavelength)**3
            * np.exp(-E_23s_inf/T_rad) * alpha_recomb(T_rad, 'HeI_23s')
        )

    else:

        return TypeError('invalid species.')

def peebles_C(xHII, rs):
    """Hydrogen Peebles C coefficient.

    This is the ratio of the total rate for transitions from n = 2 to the ground state to the total rate of all transitions, including ionization.

    Parameters
    ----------
    xHII : float
        The ionization fraction nHII/nH.
    rs : float
        The redshift in 1+z.

    Returns
    -------
    float
        The Peebles C factor.
    """
    # Rate 2s to 1s transition.
    rate_2s1s = width_2s1s_H

    # Rate of 2p to 1s transition, times (1 - xHII). 
    rate_2p1s_times_x1s = (
        8 * np.pi * hubble(rs)/
        (3*(nH * rs**3 * (c/lya_freq)**3))
    )

    # Gaussian corrections. 
    gauss_corr_1 = -0.14*np.exp(-((np.log(rs) - 7.28)/0.18)**2)
    gauss_corr_2 = 0.079*np.exp(-((np.log(rs) - 6.73)/0.33)**2)

    rate_2p1s_times_x1s /= (1 + gauss_corr_1 + gauss_corr_2)

    rate_exc = 3 * rate_2p1s_times_x1s/4 + (1-xHII) * rate_2s1s/4

    rate_ion = (1-xHII) * beta_ion(TCMB(rs), 'HI')

    return rate_exc/(rate_exc + rate_ion)

def C_He(xHII, xHeII, rs, species):
    """Helium C coefficients. 

    These coefficients play a similar role to the Peebles C factor.

    Parameters
    ----------
    xHII : float
        The HI ionization fraction nHII/nH.
    xHeII : float
        The HeI ionization fraction nHeII/nH.
    rs : float
        The redshift in 1+z.
    species : {'singlet', 'triplet'}
        The relevant helium C coefficient to return.

    Returns
    -------
    float
        The C coefficient.
    """

    T = TCMB(rs)

    if species == 'singlet':

        # Energy difference between 21p and 21s states 
        E_ps = He_exc_eng['21p'] - He_exc_eng['21s']

        # Sobolev optical depth
        tau = (
            3 * A_He_21p * nH*rs**3 * (chi - xHeII)
            * He_exc_lambda['21p']**3
            / (8 * np.pi * hubble(rs))
        )

        # Escape probability
        p_He = (1 - np.exp(-tau))/tau

        # Taking into account hydrogen. The probability
        # of interaction is 1/(1 + a*gamma**b)
        a = 0.36
        b = 0.86

        sigma_H_photo_ion = photo_ion_xsec(He_exc_eng['21p'], 'HI')
        Delta_nu = (c/He_exc_lambda['21p'])*np.sqrt(2 * T / mHe)

        gamma_numer = 3*A_He_21p*(chi - xHeII)*He_exc_lambda['21p']**2
        gamma_denom = 8*np.pi**(3/2)*sigma_H_photo_ion*Delta_nu*(1 - xHII)

        if xHII >= 1:
            # Avoid dividing by zero.
            gamma = np.inf
        else:
            gamma = gamma_numer/gamma_denom

        p_H = 1/(1 + a*gamma**b)

        # rate for excitation to 21p
        K = (1/3)/(A_He_21p * (p_He + p_H) * (nH*rs**3) * (chi - xHeII))

        numer = (
            1 + K * width_21s_1s_He 
            * (nH*rs**3) * (chi - xHeII) * np.exp(E_ps/T)
        )
        denom = (
            1 + K * (width_21s_1s_He + beta_ion(T, 'HeI_21s')) 
            * (nH*rs**3) * (chi - xHeII) * np.exp(E_ps/T)
        )

        return numer/denom

    elif species == 'triplet':

        E_ps = He_exc_eng['23p'] - He_exc_eng['23s']

        tau = 3 * A_He_23P1 * nH*rs**3 * (chi - xHeII) * (
            He_exc_lambda['23p']**3/(8* np.pi *hubble(rs))
        )

        # Escape probability
        p_He = (1 - np.exp(-tau))/tau

        # Taking into account hydrogen. The probability of
        # interaction is 1/(1 + a*gamma**b)

        a = 0.66
        b = 0.9

        sigma_H_photo_ion = photo_ion_xsec(He_exc_eng['23p'], 'HI')
        Delta_nu = (c/He_exc_lambda['23p'])*np.sqrt(2 * T / mHe)

        gamma_numer = 3*A_He_23P1*(chi - xHeII)*He_exc_lambda['23p']**2
        gamma_denom = 8*np.pi**(3/2)*sigma_H_photo_ion*Delta_nu*(1 - xHII)

        if xHII >= 1:
            # Avoid dividing by zero. 
            gamma = np.inf
        else:
            gamma = gamma_numer/gamma_denom

        p_H = 1/(1 + a*gamma**b)

        if beta_ion(TCMB(rs), 'HeI_23s') == 0.:
            return 1.
        else:
            # Numerator agrees with astro-ph/0703438, but not RECFAST.
            C_He_triplet = A_He_23P1*(p_He + p_H)*np.exp(-E_ps/T)
            C_He_triplet /= beta_ion(TCMB(rs), 'HeI_23s') + C_He_triplet

        return C_He_triplet

    else:

        return TypeError('invalid species.')

def xe_Saha(rs, species):
    """Saha equilibrium ionization value for H and He. 

    Parameters
    ----------
    rs : float
        The redshift in 1+z.
    species : {'HI', 'HeI'}
        The relevant species.

    Returns
    -------
    float
        The Saha equilibrium xe.

    Notes
    -----
    See astro-ph/9909275 and 1011.3758 for details.
    """

    T = TCMB(rs)

    de_broglie_wavelength = c * 2*np.pi*hbar / np.sqrt(2 * np.pi * me * T)

    if species == 'HI':

        rhs = (1/de_broglie_wavelength)**3 / (nH*rs**3) * np.exp(-rydberg/T)
        a  = 1. 
        b  = rhs
        q  = -rhs

        if rhs < 1e8:
            
            xe = (-b + np.sqrt(b**2 - 4*a*q))/(2*a)

        else:

            xe = 1. - a/rhs

    elif species == 'HeI':
        rhs = (
            4 * (1/de_broglie_wavelength)**3 
            / (nH*rs**3) * np.exp(-He_ion_eng/T)
        )

        if rhs < 1e9:
            a   = 1. 
            b   = rhs - 1. 
            q   = -(1. + chi)*rhs

            xe = (-b + np.sqrt(b**2 - 4*a*q))/(2*a)
        else:
            xe = (1 + chi)*(1 - (1 + chi)/rhs)
    else:
        raise TypeError('invalid species.')

    return xe

def d_xe_Saha_dz(rs, species):
    """`z`-derivative of the Saha equilibrium ionization value.

    Parameters
    ----------
    rs : float
        The redshift in 1+z.
    species : {'singlet', 'triplet'}

    Returns
    -------
    float
        The derivative of the Saha equilibrium d xe/dz. 

    Notes
    -----
    See astro-ph/9909275 and 1011.3758 for details.
    """
    
    xe    = xe_Saha(rs, species)

    if species == 'HI':

        numer = (rydberg/TCMB(rs) - 3/2)*xe**2*(1-xe)
        denom = rs*(2*xe*(1-xe) + xe**2)

    elif species == 'HeI':

        numer = (He_ion_eng/TCMB(rs) - 3/2) * (xe - 1) * xe * (1 + chi - xe)
        denom = rs * (chi * (2*xe - 1) - (xe - 1)**2)

    else:

        return TypeError('invalid species.')

    return numer/denom

# Standard ionization and thermal histories

# if 'pytest' not in sys.modules and 'readthedocs' not in sys.modules:
#     soln_baseline = pickle.load(open(data_path+'/std_soln_He.p', 'rb'))

#     _xHII_std  = interp1d(soln_baseline[0,:], soln_baseline[2,:])
#     _xHeII_std = interp1d(soln_baseline[0,:], soln_baseline[3,:])
#     _Tm_std    = interp1d(soln_baseline[0,:], soln_baseline[1,:])

_xHII_std  = None
_xHeII_std = None
_Tm_std    = None

def xHII_std(rs):
    """Baseline nHII/nH value.

    Parameters
    ----------
    rs : float
        The redshift (1+z). 

    Returns
    -------
    float
        nHII/nH. 
    """

    global _xHII_std

    if _xHII_std is None:

        rs_vec   = load_data('hist')['rs']
        xHII_vec = load_data('hist')['xHII']

        _xHII_std = interp1d(rs_vec, xHII_vec)

    return _xHII_std(rs)

def xHeII_std(rs):
    """Baseline nHeII/nH value.

    Parameters
    ----------
    rs : float
        The redshift (1+z). 

    Returns
    -------
    float
        nHeII/nH. 
    """

    global _xHeII_std

    if _xHeII_std is None:

        rs_vec    = load_data('hist')['rs']
        xHeII_vec = load_data('hist')['xHeII']

        _xHeII_std = interp1d(rs_vec, xHeII_vec)

    return _xHeII_std(rs)

def Tm_std(rs):
    """Baseline Tm value.

    Parameters
    ----------
    rs : float
        The redshift (1+z). 

    Returns
    -------
    float
        Tm in eV. 
    """

    global _Tm_std

    if _Tm_std is None:

        rs_vec  = load_data('hist')['rs']
        Tm_vec  = load_data('hist')['Tm']

        _Tm_std = interp1d(rs_vec, Tm_vec)

    return _Tm_std(rs)

    # For redshifts above 3000, assume full coupling to the CMB temperature
    #if isinstance(rs,np.float):
    #    if rs > 3e3:
    #        return TCMB(rs)
    #    else:
    #        return _Tm_std(rs)
    #else:
    #    Tm_list = TCMB(rs)
    #    Tm_list[rs<3e3] = _Tm_std(rs)
    #    return Tm_list


# Atomic Cross-Sections

def photo_ion_xsec(eng, species):
    """Photoionization cross section in cm\ :sup:`2`\ .

    Cross sections for hydrogen, neutral helium and singly-ionized helium are available.

    Parameters
    ----------
    eng : ndarray
        Energy to evaluate the cross section at.
    species : {'HI', 'HeI', 'HeII'}
        Species of interest.

    Returns
    -------
    xsec : ndarray
        Photoionization cross section.
    """

    eng_thres = {'HI':rydberg, 'HeI':He_ion_eng, 'HeII':4*rydberg}

    if not isinstance(eng, np.ndarray):
        if isinstance(eng, list):
            return photo_ion_xsec(np.array(eng), species)
        else:
            if eng > eng_thres[species]:
                return photo_ion_xsec(np.array([eng]), species)[0]
            else:
                return 0

    ind_above = np.where(eng > eng_thres[species])
    
    xsec = np.zeros(eng.size)

    if species == 'HI' or species =='HeII':
        eta = np.zeros(eng.size)
        eta[ind_above] = 1./np.sqrt(eng[ind_above]/eng_thres[species] - 1.)
        xsec[ind_above] = (2.**9*np.pi**2*ele_rad**2/(3.*alpha**3)
            * (eng_thres[species]/eng[ind_above])**4
            * np.exp(-4*eta[ind_above]*np.arctan(1./eta[ind_above]))
            / (1.-np.exp(-2*np.pi*eta[ind_above]))
            )
    elif species == 'HeI':
        x = np.zeros(eng.size)
        y = np.zeros(eng.size)

        sigma0 = 9.492e2*1e-18      # in cm^2
        E0     = 13.61              # in eV
        ya     = 1.469
        P      = 3.188
        yw     = 2.039
        y0     = 4.434e-1
        y1     = 2.136

        x[ind_above]    = (eng[ind_above]/E0) - y0
        y[ind_above]    = np.sqrt(x[ind_above]**2 + y1**2)
        xsec[ind_above] = (sigma0*((x[ind_above] - 1)**2 + yw**2)
            *y[ind_above]**(0.5*P - 5.5)
            *(1 + np.sqrt(y[ind_above]/ya))**(-P)
            )

    return xsec

def photo_ion_rate(rs, eng, xH, xe, atom=None):
    """Photoionization rate in cm\ :sup:`-3` s\ :sup:`-1`\ .

    Parameters
    ----------
    rs : float
        Redshift (1+z).
    eng : ndarray
        Energies to evaluate the cross section.
    xH : float
        Ionization fraction nH+/nH.
    xe : float
        Ionization fraction ne/nH = nH+/nH + nHe+/nH.
    atom : {None,'HI','HeI','HeII'}, optional
        Determines which photoionization rate is returned. The default value is ``None``, which returns the total rate.

    Returns
    -------
    float
        The photoionization rate of the particular species or the total ionization rate.

    """
    atoms = ['HI', 'HeI', 'HeII']

    xHe = xe - xH
    atom_densities = {
        'HI':nH*(1-xH)*rs**3, 'HeI':(nHe - xHe*nH)*rs**3,
        'HeII':xHe*nH*rs**3
    }

    ion_rate = {
        atom: photo_ion_xsec(eng,atom) * atom_densities[atom] * c
        for atom in atoms
    }

    if atom is not None:
        return ion_rate[atom]
    else:
        return sum([ion_rate[atom] for atom in atoms])

def coll_exc_xsec(eng, species=None, method = 'old', state=None):
    """ e-e collisional excitation cross section in cm\ :sup:`2`\ . 

    see under method for references

    Parameters
    ----------
    eng : float or ndarray
        Abscissa of *kinetic* energies.
    species : {'HI', 'HeI', 'HeII'}
        Species of interest.
    method : {'old', 'MEDEA', 'new'}
        if method == 'old', see 0906.1197; 
        if method == 'MEDEA', see Mon. Not. R. Astron. Soc. 422, 420–433 (2012); 
        if method == 'new', nothing yet

    Returns
    -------
    float or ndarray
        e-e collisional excitation cross section.
    """
    if method == 'old':
        if species == 'HI' or species == 'HeI':

            if species == 'HI':
                A_coeff = 0.5555
                B_coeff = 0.2718
                C_coeff = 0.0001
                E_bind = rydberg
                E_exc = lya_eng
            elif species == 'HeI':
                A_coeff = 0.1771
                B_coeff = -0.0822
                C_coeff = 0.0356
                E_bind = He_ion_eng
                E_exc  = He_exc_eng['23s']

            prefac = 4*np.pi*bohr_rad**2*rydberg/(eng + E_bind + E_exc)

            xsec = prefac*(
                A_coeff*np.log(eng/rydberg) + B_coeff + C_coeff*rydberg/eng
            )

            try:
                xsec[eng <= E_exc] *= 0
            except:
                if eng <= E_exc:
                    return 0

            return xsec

        elif species == 'HeII':

            alpha = 3.22
            beta = 0.357
            gamma = 0.00157
            delta = 1.59
            eta = 0.764
            E_exc = 4*lya_eng

            x = eng/E_exc

            prefac = np.pi*bohr_rad**2/(16*x)
            xsec = prefac*(
                alpha*np.log(x) + beta*np.log(x)/x
                + gamma + delta/x + eta/x**2
            )

            try:
                xsec[eng <= E_exc] *= 0
            except:
                if eng <= E_exc:
                    return 0

            return xsec

        else:
            raise TypeError('invalid species.')
    elif (method == 'MEDEA') | (method == 'AcharyaKhatri'):
        if (method == 'AcharyaKhatri') | (method == 'MEDEA'):
            if (species == 'HeI'):
                state = '2p'
            elif species == 'HeII':
                return 0


        if (species == 'HI') and ((state != '2s') and ((state[-1] != 'p') or (int(state[:-1]) not in np.arange(2,11)))):

            TypeError("Must specify 2s, 2p, 3p, ..., or 10p, for HI excitation") 

        elif (species == 'HeI') and ((state[1:] != 'p1P') and (int(state[0]) not in np.arange(2,11))):

            TypeError("Must specify 2p1P, 3p1P, ..., or 10p1P for HeI excitation") 

        else:

            # If eng is a number, make it an np.ndarray
            if isinstance(eng, float):
                eng = np.array([eng])

            # parameters for 1s-np, see Stone, Kim, Desclaux (2002). No resonance  at threshold is included.
            # threshold energy
            exc_eng  = {'HI': HI_exc_eng, 'HeI': He_exc_eng}
            bind_eng = {'HI': rydberg,    'HeI': He_ion_eng}

            #!!! CHECK must include relativistic corrections at E>10keV
            #Parameters for high energy limit, ordered from 2p to 10p
            a_params = {
                'HI':  [ .555512,  .089083,  .030956,  .014534,  .008031,  .004919,  .003237,  .002246,  .001623],
                'HeI': [ .165601,  .041611,  .016111,  .008298,  .004740,  .002963,  .001975,  .001383,  .001006]
            } 
            b_params = {
                'HI':  [ .271785,  .060202,  .022984,  .011243,  .006348,  .003939,  .002550,  .001824,  .001323],
                'HeI': [-.076942, -.018087, -.007040, -.003475, -.001972, -.001227, -.000816, -.000570, -.000414]
            }
            c_params = {
                'HI':  [ .000112, -.019775, -.009279, -.004880, -.002853, -.001806, -.001213, -.000854, -.000623],
                'HeI': [ .033306,  .002104, -.000045, -.000228, -.000194, -.000146, -.000108, -.000080, -.000061]
            }

            fsc_HeI  = [ .2583,    .07061,   .02899,   .01466,   .00844,   .00529,   .00354,   .00248,   .00181]
            facc_HeI = [ .2762,    .07343,   .02986,   .01504,   .00863,   .00541,   .00361,   .00253,   .00184]

            # Eqn (5) of Kim, Stone, Desclaux (2002)
            def xsec_asympt(species, state, KE):
                ind = int(state[0])-2
                if species == 'HeI':
                    f_ratio = facc_HeI[ind]/fsc_HeI[ind]
                else:
                    if state == '2s':
                        return np.zeros_like(KE)
                    f_ratio = 1.

                factor = (a_params[species][ind] * np.log(KE/rydberg) 
                        + b_params[species][ind]
                        + c_params[species][ind] * rydberg/KE)*f_ratio

                return 4*np.pi*bohr_rad**2*rydberg/(KE + bind_eng[species] + exc_eng[species][state]) * factor


            if method == 'MEDEA':
                exc_xsec = load_data('exc')[species][state](eng)*1e-16 # in units of cm^-2
                exc_xsec[eng<exc_eng[species][state]] = 0
                exc_xsec[eng>3e3] = xsec_asympt(species, state, eng[eng>3e3])


            elif method == 'AcharyaKhatri':
                exc_xsec = load_data('exc_AcharyaKhatri')[species][state](eng) # in units of cm^2
                exc_xsec[eng<exc_eng[species][state]] = 0
                # CCC cross-sections end around 1 keV
                if species == 'HI':
                    exc_xsec[eng>999] = xsec_asympt(species, state, eng[eng>999])
                else:
                    exc_xsec[eng>900] = xsec_asympt(species, state, eng[eng>900])

                # !!! bad extrapolation
                exc_xsec[eng<14] = load_data('exc_AcharyaKhatri')[species][state](14)

            return exc_xsec
    
    elif method == 'new':
        raise TypeError('new method has not yet been implemented')
    else:
        raise TypeError("Must pick method = {'old', 'MEDEA', 'AcharyaKhatri', or 'new'}")

def coll_ion_xsec(eng, species=None, method='old'):
    """ e-e collisional ionization cross section in cm\ :sup:`2`\ . 

    For the 'old' method, see 0906.1197.
    For the 'MEDEA' method, see Mon. Not. R. Astron. Soc. 422, 420–433 (2012).
    For the 'new' method, TBD.


    Parameters
    ----------
    eng : float or ndarray
        Abscissa of *kinetic* energies.
    species : {'HI', 'HeI', 'HeII'}
        Species of interest.
    method : {'old', 'MEDEA', 'new'}
        if method == 'old', see 0906.1197; if method == 'MEDEA', see Mon. Not. R. Astron. Soc. 422, 420–433 (2012); if method == 'new', nothing yet

    Returns
    -------
    float or ndarray
        e-e collisional ionization cross section.

    Notes
    -----
    Returns the Arnaud and Rothenflug rate if method == 'old'.

    """
    if method == 'AcharyaKhatri':
        if species != 'HI':
            method='old'

    if method == 'old':
        if species == 'HI':
            A_coeff = 22.8
            B_coeff = -12.0
            C_coeff = 1.9
            D_coeff = -22.6
            ion_pot = rydberg
        elif species == 'HeI':
            A_coeff = 17.8
            B_coeff = -11.0
            C_coeff = 7.0
            D_coeff = -23.2
            ion_pot = He_ion_eng
        elif species == 'HeII':
            A_coeff = 14.4
            B_coeff = -5.6
            C_coeff = 1.9
            D_coeff = -13.3
            ion_pot = 4*rydberg
        else:
            raise TypeError('invalid species.')

        u = eng/ion_pot

        prefac = 1e-14/(u*ion_pot**2)

        xsec = prefac*(
            A_coeff*(1 - 1/u) + B_coeff*(1 - 1/u)**2
            + C_coeff*np.log(u) + D_coeff*np.log(u)/u
        )

        try:
            xsec[eng <= ion_pot] *= 0
        except:
            if eng <= ion_pot:
                return 0

    elif method == 'MEDEA':
        if (species == 'HI') or (species == 'HeI'):
            if species == 'HI':
                # Binding Energy
                B = rydberg

                # Average kinetic energy of electron in the atom's potential
                U = rydberg

                # Number of electrons in valence shell
                N = 1
                Ni = .4343
                def D(t):
                    return (-.022473/2*(1-((t+1)/2)**-2) + 1.1775/3*(1-((t+1)/2)**-3) + (
                        -0.46264/4*(1-((t+1)/2)**-4) + 0.089064/5*(1-((t+1)/2)**-5)
                    ))/N

            elif species == 'HeI':
                B  = He_ion_eng
                U  = 39.51
                N  = 2
                Ni = 1.605
                def D(t):
                    return (12.178/3*(1-((t+1)/2)**-3) - 29.585/4*(1-((t+1)/2)**-4) + (
                        31.251/5*(1-((t+1)/2)**-5) - 12.175/6*(1-((t+1)/2)**-6)
                    ))/N
            S = 4 * np.pi * bohr_rad**2 * N * (rydberg/B)**2
            t = eng/B
            u = U/B
            xsec = S/(t+u+1) * (D(t)*np.log(t) + (2 - Ni/N)*((t-1)/t - np.log(t)/(t+1)))

        elif species == 'HeII':
            B = 4*rydberg
            Z = 2
            def F1(tt):
                return -1.4332/(tt+1)**2
            def F2(tt): 
                return 1.4332/(tt+1)
            def F3(tt):
                return 0.5668 * np.log(tt)/(tt+1)
            tt = eng/Z**2/rydberg

            xsec = 4 * np.pi * bohr_rad**2/Z**4 * (F1(tt)*np.log(tt) + F2(tt)*(1-tt**-1) + F3(tt)*(1-tt**-2)/2)

        else:
            raise TypeError('invalid species.')


        try:
            xsec[eng <= B] *= 0
        except:
            if eng <= B:
                return 0
    elif method=='AcharyaKhatri':
        if isinstance(eng, float):
            eng = np.array([eng])
        ionHI=np.array([[14.00,1.69960e-18],[14.50,3.05083e-18],[15.00,5.65916e-18],
            [15.60,8.92024e-18],[17.60,1.99166e-17],[20.00,2.99277e-17],[25.00,4.46188e-17],
            [30.00,5.27297e-17],[35.00,5.63541e-17],[45.00,6.24417e-17],[54.42,6.32510e-17],
            [70.00,6.11397e-17],[100.00,5.35302e-17],[150.00,4.58328e-17],[200.00,3.85752e-17],
            [500.00,1.94152e-17],[999.00,1.08292e-17]]
        )
        ionHI = interp1d(ionHI[:,0], ionHI[:,1], bounds_error=False, fill_value=(1.69960e-18,1.08292e-17))
        xsec = ionHI(eng)
        
        # Low Values
        xsec[eng<rydberg] = 0

        # High Values
        neng = eng[eng>999]/rydberg
        C_i = np.exp(3.048)
        gamma_i = -1.63 - np.log(neng)
        xsec[eng>999.0] = 4*np.pi * bohr_rad**2/neng * (
                0.28*np.log(4*C_i*neng) + gamma_i/neng
        )
        return xsec

    else:
        raise TypeError('method = new not developed yet')

    return xsec

def coll_ion_sec_elec_spec(in_eng, eng, species=None, method='old'):
    """ Secondary electron spectrum after collisional ionization. 

    For 'old' method, See 0910.4410 --> Shull, Astrophysical Journal, 234:761-764, 1979 December 1
    For 'MEDEA' method, see Kim Y., Rudd M. E., 1994, Phys. Rev. A, 50, 3954 OR Mon. Not. R. Astron. Soc. 422, 420–433 (2012)
    For 'new' method, TBD

    Parameters
    ----------
    in_eng : float
        The incoming electron energy.
    eng : ndarray
        Abscissa of *kinetic* energies.
    species : {'HI', 'HeI', 'HeII'}
        Species of interest.
    method : {'old', 'MEDEA', 'new'}
        if method == 'old', see Shull, Astrophysical Journal, 234:761-764, 1979 December 1; 
        if method == 'MEDEA', we follow the method used in the MEDEA code, which is the 'old' method; 
        if method == 'new', nothing yet

    Returns
    -------
    ndarray
        Secondary electron spectrum. Total number of electrons = 2.

    Notes
    -----
    Includes both the freed and initial electrons. Conservation of energy
    is not enforced, but number of electrons is.

    """

    from darkhistory.spec.spectrum import Spectrum
    from darkhistory.spec import spectools

    if (method == 'old') | (method == 'MEDEA'):
        if species == 'HI':
            eps_i = 8.
            ion_pot = rydberg
        elif species == 'HeI':
            eps_i = 12.8 #15.8
            ion_pot = He_ion_eng
        elif species == 'HeII':
            eps_i = 27 #32.6
            ion_pot = 4*rydberg
        else:
            raise TypeError('invalid species.')

        if np.isscalar(in_eng):
            # If the input energy is below threshold return zeros.
            if in_eng < ion_pot:
                return np.zeros_like(eng)

            # See Shull (1979) eqn A1
            low_eng_elec_dNdE = 1/(1 + (eng/eps_i)**2.0) #2.1
            # This spectrum describes the lower energy electron only.
            low_eng_elec_dNdE[eng >= (in_eng - ion_pot)/2] = 0

            if np.sum(low_eng_elec_dNdE) == 0:
                # The lowest bin lies above the halfway point, (in_eng - ion_pot)/2.
                # All electrons are now counted in the lowest bin.
                #!!!
                tot_elec_N = np.zeros_like(eng)
                tot_elec_N[0] = (in_eng-ion_pot)/eng[0]
                return np.outer(np.ones_like(in_eng), tot_elec_N)

            low_eng_elec_spec = Spectrum(eng, low_eng_elec_dNdE)

            # Normalize the spectrum to one electron.
            low_eng_elec_spec /= low_eng_elec_spec.totN()

            in_eng = np.array([in_eng])

            low_eng_elec_N = np.outer(
                np.ones_like(in_eng), low_eng_elec_spec.N)

            high_eng_elec_N = spectools.engloss_rebin_fast(
                in_eng, eng + ion_pot, low_eng_elec_N, eng
            )

            return np.squeeze(low_eng_elec_N + high_eng_elec_N)

        else:

            from darkhistory.spec.spectra import Spectra

            in_eng_mask = np.outer(in_eng, np.ones_like(eng))
            eng_mask    = np.outer(np.ones_like(in_eng), eng)

            # See Shull (1979) eqn A1
            low_eng_elec_dNdE = np.outer(
                np.ones_like(in_eng), 1/(1 + (eng/eps_i)**2.0) #2.1
            )

            low_eng_elec_dNdE[eng_mask >= (in_eng_mask - ion_pot)/2] = 0

            # These input energies lie above the halfway point, (in_eng - ion_pot)/2.
            # We will treat the lowest bin as an underflow bin (see end of this if statement block)
            #!!!
            zero_mask = (np.sum(low_eng_elec_dNdE, axis=1) == 0) & (in_eng>ion_pot)

            # Normalize the spectrum to one electron.
            low_eng_elec_spec = Spectra(
                low_eng_elec_dNdE, eng=eng, in_eng=in_eng
            )

            totN_arr = low_eng_elec_spec.totN()
            # Avoids divide by zero errors.
            totN_arr[totN_arr == 0] = np.inf

            low_eng_elec_spec /= totN_arr

            if low_eng_elec_spec.spec_type == 'dNdE':
                low_eng_elec_spec.switch_spec_type()

            low_eng_elec_N = low_eng_elec_spec.grid_vals

            high_eng_elec_N = spectools.engloss_rebin_fast(
                in_eng, eng + ion_pot, low_eng_elec_N, eng
            )

            #underflow bin
            low_eng_elec_N[zero_mask,0]= (in_eng[zero_mask]-ion_pot)/eng[0]
            #high_eng_elec_N[zero_mask,0]=1.

    elif method == 'MEDEA':
        # See Kim Y., Rudd M. E., 1994, Phys. Rev. A, 50, 3954
        if species == 'HI':
            # Binding Energy
            B = rydberg

            # Average kinetic energy of electron in the atom's potential
            U = rydberg
            u = U/B

            # Number of electrons in valence shell
            N = 1
            S = 4 * np.pi * bohr_rad**2

            # Taylor coefficients of differential dipole oscillator strength for 
            def dfdw(w):
                return -.022473*(w+1)**-2 + 1.1775*(w+1)**-3 - 0.46264*(w+1)**-4 + 0.089064*(w+1)**-5
            #a =  0
            #b = -0.022473
            #c =  1.1775
            #d = -0.46264
            #e =  0.089064
            #f =  0
            #g =  0
            Ni=  0.4343

        elif species == 'HeI':
            B = He_ion_eng
            U = 39.51
            u = U/B
            N = 2
            S = 4 * np.pi * bohr_rad**2 * N * (rydberg/B)**2
            #a =  0
            #b =  0
            #c =  12.178
            #d = -29.585
            #e =  31.251
            #f = -12.175
            #g =  0
            def dfdw(w):
                return 12.178*(w+1)**-3 - 29.585*(w+1)**-4 + 31.251*(w+1)**-5 - 12.175*(w+1)**-6
            Ni=  1.605

        elif species == 'HeII':
            Z = 2
            B = Z**2 * rydberg
            S = 4 * np.pi * bohr_rad**2 * (rydberg/B)**2
            def F(tt):
                return np.array([-1.4332/(tt+1)**2, 1.4332/(tt+1), 0.5668 * np.log(tt)/(tt+1)])
            def f(n,w):
                return (w+1)**-n
            def ft(n,tt,w):
                return (tt-w)**-n

        else:
            raise TypeError('invalid species.')


        w = eng/B

        if np.isscalar(in_eng):
            # dsigma / dW, where W is the kinetic energy of the secondary (lower energy) electron
            if species != 'HeII':
                t = in_eng/B

                # When t = w, this factor blows up, but w < t.
                fac = np.divide(1,(t-w), out=np.zeros_like(w), where = w<=(t-1)/2)

                low_eng_elec_dNdE = (
                    S/(t + u + 1)/B*(
                        (Ni/N-2)/(t+1)*(1/(w+1) + fac)+
                        (2-(Ni/N))*(1/(w+1)**2  + fac**2)+
                        np.log(t)/(N*(w+1)) * dfdw(w)
                    )
                )
            else:
                tt = in_eng/Z**2/rydberg
                F_array = F(tt)
                summation = sum(
                        np.array([(f(n,w) + ft(n,tt,w)) * F_array[n-1] for n in [1,2,3]])
                        )
                low_eng_elec_dNdE = S*summation/55

            # This spectrum describes the lower energy electron only.
            low_eng_elec_dNdE[eng >= (in_eng - B)/2] = 0
            # Normalize the spectrum to one electron.

            low_eng_elec_spec = Spectrum(eng, low_eng_elec_dNdE)
            if np.sum(low_eng_elec_dNdE) == 0:
                # Either in_eng < in_pot, or the lowest bin lies
                # above the halfway point, (in_eng - B)/2.
                # Add to the lowest bin.
                return np.zeros_like(eng)

            #low_eng_elec_spec /= low_eng_elec_spec.totN()

            in_eng = np.array([in_eng])

            low_eng_elec_N = np.outer(
                np.ones_like(in_eng), low_eng_elec_spec.N)

            high_eng_elec_N = spectools.engloss_rebin_fast(
                in_eng, eng + B, low_eng_elec_N, eng
            )

            return np.squeeze(low_eng_elec_N + high_eng_elec_N)

        else:

            from darkhistory.spec.spectra import Spectra

            in_eng_mask = np.outer(in_eng, np.ones_like(eng))
            eng_mask    = np.outer(np.ones_like(in_eng), eng)

            if species != 'HeII':
                #t = in_eng/B
                #fac = np.transpose(np.array([
                #        np.divide(1,_t-w, out=np.zeros_like(w), where = w<=(_t-1)/2)
                #        for _t in t]))

                #prefac = S/(t + u + 1)/B
                #a1 = prefac * (Ni/N-2)/(t+1)
                #b1 = 1/(w+1)
                #low_eng_elec_dNdE = np.outer(b1, a1)
                #low_eng_elec_dNdE += a1 * fac

                #low_eng_elec_dNdE += np.outer(b1**2, prefac * (2-(Ni/N)))
                #low_eng_elec_dNdE += prefac * (2-(Ni/N)) * fac**2

                #low_eng_elec_dNdE += np.outer(dfdw(w)/(N*(w+1)), prefac * np.log(t))
                #low_eng_elec_dNdE = low_eng_elec_dNdE.transpose()
                low_eng_elec_dNdE = np.array([
                    S/(t*B + U + B)*(
                        (Ni/N-2)/(t+1)*(1/(w+1) + 1/(t-w))+
                        (2-(Ni/N))*(1/(w+1)**2  + 1/(t-w)**2)+
                        np.log(t)/(N*(w+1)) * dfdw(w)
                    ) for t in in_eng/B]
                )
            else:
                tt_list = in_eng/Z**2/rydberg
                low_eng_elec_dNdE = np.zeros((in_eng.size, eng.size))
                for i,tt in enumerate(tt_list):
                    F_array = F(tt)
                    summation = sum(
                            np.array([(f(n,w) + ft(n,tt,w)) * F_array[n-1] for n in [1,2,3]])
                            )
                    low_eng_elec_dNdE[i] = S*summation

            low_eng_elec_dNdE[eng_mask >= (in_eng_mask - B)/2] = 0
            # Normalize the spectrum to one electron.
            low_eng_elec_spec = Spectra(
                low_eng_elec_dNdE, eng=eng, in_eng=in_eng
            )

            totN_arr = low_eng_elec_spec.totN()
            # Avoids divide by zero errors.
            totN_arr[totN_arr == 0] = np.inf

            low_eng_elec_spec /= totN_arr

            if low_eng_elec_spec.spec_type == 'dNdE':
                low_eng_elec_spec.switch_spec_type()

            low_eng_elec_N = low_eng_elec_spec.grid_vals

            high_eng_elec_N = spectools.engloss_rebin_fast(
                in_eng, eng + B, low_eng_elec_N, eng
            )

    elif method == 'new':
        raise TypeError('We have not developed the new method yet')
        #if np.isscalar(in_eng):
        #else:

    return low_eng_elec_N + high_eng_elec_N


def elec_heating_engloss_rate(eng, xe, rs, method='old', Te = 0):
    """Electron energy loss rate of electrons due to Coulomb heating in eV s\ :sup:`-1`\ .

    Parameters
    ----------
    eng : ndarray
        Abscissa of electron *kinetic* energy.
    xe : float
        The free electron fraction.
    rs : float
        The redshift.

    Returns
    -------
    ndarray
        The energy loss rate due to heating (positive).

    Notes
    -------
    See 0910.4410 for the expression. 
    The units are orinally from Spitzer's 1962 textbook, cgs-emu,
    so that e^2/r is in units of energy (statcoulombs), 
    so to convert to SI, we insert 1/(4*pi*eps_0)^2 and use that e^2/(4*pi*eps_0) = alpha
    """

    if method == 'MEDEA':
        method = 'old'
    if method == 'old':
        w = np.sqrt(1 - 1/(1 + eng/me)**2)
        ne = xe*nH*rs**3 * (hbar*c)**3

        prefac = 4*np.pi*alpha**2
        #zeta_e = 7.40e-11*ne/(hbar*c)**3 # Must be in units of cm^-3)
        #coulomb_log = np.log(4*eng/zeta_e)
        coulomb_log = np.log(4*eng * (4*np.pi*alpha*ne/me)**(-1/2))

        # must use the mass of the electron in eV m^2 s^-2.
        return prefac*ne*coulomb_log/(me*w) / hbar #*xe**.05
    elif method == 'AcharyaKhatri':
        ne = xe*nH*rs**3
        return 2e-4 * ne**0.97/eng**0.44 * ( (eng-Te)/(eng-0.53*Te) )**(2.36)

def f_std(mDM, rs, inj_particle=None, inj_type=None, struct=False, channel=None):
    """energy deposition fraction into channel c, f_c(z), as a function of dark matter mass and redshift.

    Parameters
    ----------
    mDM : float
        Dark matter mass
    inj_particle : string
        Injected particle, either set to 'phot' for photons, or 'elec' for electrons.
    inj_type : string
        Type of energy injection, either 'swave' or 'decay
    struct : bool
        If *True*, include structure formation, if *False* assume no structure formation. Default is *False*. This option makes no difference for decays.
    """

    if (inj_particle != 'phot') and (inj_particle != 'elec'):
        raise ValueError("inj_particle must either be 'phot' or 'elec'")

    if (inj_type != 'decay') and (inj_type != 'swave') and (inj_type != 'pwave'):
        raise ValueError("inj_type must either be 'swave' or 'decay' or 'pwave'")

    if channel not in ['H ion', 'cont', 'exc', 'heat', 'He ion']:
        raise ValueError(
            "channel must be in ['H ion', 'He ion', 'exc', 'heat', 'cont']"
        )

    if isinstance(mDM, (int, float)):
        mDM = np.array([mDM])*1.
    else:
        mDM = np.array(mDM)*1.

    if isinstance(rs, (int, float)):
        rs = np.array([rs])*1.
    else:
        rs = np.array(rs)*1.

    struct_str = ''
    if (inj_type == 'swave') or (inj_type == 'pwave'):
        if inj_particle == 'phot':
            Einj = mDM
        else:
            Einj = mDM - me

        if inj_type == 'pwave':
            struct_str = '_NFW'
        else:
            if struct:
                struct_str = '_einasto'
            else:
                struct_str = '_noStruct'
    else:
        if inj_particle == 'phot':
            Einj = mDM/2
        else:
            Einj = mDM/2 - me


    ind_dict = {'H ion' : 0, 'He ion' : 1, 'exc' : 2, 'heat' : 3, 'cont' : 4}
    ind = ind_dict[channel]
    f_data_baseline = load_data('f')[inj_particle+'_'+inj_type+struct_str]

    Einj[Einj<5.001e3] = 5.001e3
    Einj[Einj>10**12.6015] = 10**12.6015

    if inj_particle != 'phot' or inj_type != 'swave':
        rs[rs<4.017] = 4.017
    else:
        rs[rs<5.2] = 5.2
    rs[rs>3000] = 3000


    return np.exp(
            f_data_baseline((np.log10(Einj), np.log(rs)))[:,ind]
    )

# Unused for now.



# def tau_sobolev(rs):
#     """Sobolev optical depth.

#     Parameters
#     ----------
#     rs : float
#         Redshift (1+z).
#     Returns
#     -------
#     float
#     """
#     xsec = 2 * np.pi * 0.416 * np.pi * alpha * hbar * c ** 2 / me
#     lya_omega = lya_eng / hbar

#     return nH * rs ** 3 * xsec * c / (hubble(rs) * lya_omega)

def dLam2s_dnu(eng):
    """Hydrogen 2s to 1s two-photon decay rate per nu as a function of nu (unitless).

    nu is the frequency of the more energetic photon.
    To find the total decay rate (8.22 s^-1), integrate from 5.1eV/h to 10.2eV/h

    Parameters
    ----------

    Returns
    -------
    Lam : ndarray
     Decay rate per nu.
    """
    #coeff = 9 * alpha**6 * rydberg /(
    # 2**10 * 2 * np.pi * hbar
    #) * width_2s1s_H/8.26548398114 / lya_freq
    # Fix the normalization
    coeff = 1/19.317043357174253/2
    #     #print(coeff)

    #     # coeff * psi(y) * dy = probability of emitting a photon in the window nu_alpha * [y, y+dy)
    #     # interpolating points come from Spitzer and Greenstein, 1951
    y = np.arange(0, 1.05, .05)
    psi = np.array([0, 1.725, 2.783, 3.481, 3.961, 4.306, 4.546, 4.711, 4.824, 4.889, 4.907,
                4.889, 4.824, 4.711, 4.546, 4.306, 3.961, 3.481, 2.783, 1.725, 0])

    #     # evaluation outside of interpolation window yields 0.
    f = interp1d(y, psi, kind='cubic', bounds_error=False, fill_value=(0,0))
    #def dLam2s_dnu(nu):
    #nu = eng/(hbar*2*np.pi)
    return coeff * f(eng/lya_eng)

     #return dLam2s_dnu


# # CMB



# def A_2s(y):
#     """2s to 1s two-photon decay probability density.

#     A_2s(y) * dy = probability that a photon is emitted with a frequency in the range nu_lya * [y, y+dy)
#     with the other photon constrained by h*nu_1 + h*nu_2 = 10.2 eV.

#     Parameters
#     ----------
#     y : float
#         nu / nu_lya, frequency of one of the photons divided by lyman-alpha frequency.

#     Returns
#     -------
#     float or ndarray
#         probability density of emitting those two photons.
#     """
#     return 0


