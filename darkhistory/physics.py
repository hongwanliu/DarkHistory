""" Physics functions as well as constants.

"""

import numpy as np

# Fundamental constants
mp          = 0.938272081e9                     
"""Proton mass in eV."""
me          = 510998.9461
"""Electron mass in eV."""
hbar        = 6.58211951e-16
"""hbar in eV s."""
c           = 299792458e2
"""Speed of light in cm/s."""
kB          = 8.6173324e-5
"""Boltzmann constant in eV/K."""
alpha       = 1/137.035999139
"""Fine structure constant."""
ele         = 1.60217662e-19
"""Electron charge in C."""

# Atomic and optical physics

thomson_xsec = 6.652458734e-25
"""Thomson cross section in cm^2."""
stefboltz    = np.pi**2 / (60 * (hbar**3) * (c**2))
"""Stefan-Boltzmann constant in eV^-3 cm^-2 s^2."""
rydberg      = 13.60569253
"""Ionization potential of ground state hydrogen in eV."""
lya_eng      = rydberg*3/4
"""Lyman alpha transition energy in eV."""
lya_freq     = lya_eng / (2*np.pi*hbar)
"""Lyman alpha transition frequency in Hz."""
width_2s1s    = 8.23
"""Hydrogen 2s to 1s decay width in s^-1."""
bohr_rad     = (hbar*c) / (me*alpha)
"""Bohr radius in cm."""
ele_rad      = bohr_rad * (alpha**2)
"""Classical electron radius in cm."""
ele_compton  = 2*np.pi*hbar*c/me
"""Electron Compton wavelength in cm."""

# Hubble

h    = 0.6727
""" h parameter."""
H0   = 100*h*3.241e-20
""" Hubble parameter today in s^-1."""

# Omegas

omega_m      = 0.3156 
""" Omega of all matter today."""
omega_rad    = 8e-5
""" Omega of radiation today."""
omega_lambda = 0.6844
""" Omega of dark energy today."""
omega_baryon = 0.02225/(h**2)
""" Omega of baryons today."""
omega_DM      = 0.1198/(h**2)
""" Omega of dark matter today."""

# Densities

rho_crit     = 1.05375e4*(h**2)
""" Critical density of the universe in eV/cm^3."""
rho_DM       = rho_crit*omega_DM
""" DM density in eV/cm^3."""
rho_baryon   = rho_crit*omega_baryon
""" Baryon density in eV/cm^3."""
nB          = rho_baryon/mp
""" Baryon number density in cm^-3."""
YHe         = 0.250                       
"""Helium abundance by mass."""
nH          = (1-YHe)*nB
""" Atomic hydrogen number density in cm^-3."""
nHe         = (YHe/4)*nB
""" Atomic helium number density in cm^-3."""
nA          = nH + nHe
""" Hydrogen and helium number density in cm^-3."""

# Cosmology functions

def hubble(rs, H0=H0, omega_m=omega_m, omega_rad=omega_rad, omega_lambda=omega_lambda): 
    """ Hubble parameter in s^-1.

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

    return 1/(rs*hubble(rs, H0, omega_m, omega_rad, omega_lambda))

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

    return 2.7255 * 8.61733e-5 * rs

def get_inj_rate(inj_type, inj_fac):
    """Dark matter injection rate function.

    Parameters
    ----------
    inj_type : {'sWave', 'decay'}
        The type of injection. 
    inj_fac : float
        The prefactor for the injection rate, consisting of everything other than the redshift dependence. 

    Returns
    -------
    function
        The function takes redshift as an input, and outputs the injection rate. 
    """

    def inj_rate(rs):
        if inj_type == 'sWave':
            return inj_fac*(rs**6)
        elif inj_type == 'decay': 
            return inj_fac*(rs**3)

    return inj_rate




# Atomic Cross-Sections

def photo_ion_xsec(eng, species):
    """Photoionization cross section in cm^2. 

    Cross sections for hydrogen, neutral helium and singly-ionized helium are available. 

    Parameters
    ----------
    eng : ndarray
        Energy to evaluate the cross section at. 
    species : {'H0', 'He0', 'He1'}
        Species of interest.

    Returns
    -------
    xsec : ndarray
        Cross section in cm^2. 
    """

    eng_thres = {'H0':rydberg, 'He0':24.6, 'He1':4*rydberg}

    ind_above = np.where(eng > eng_thres[species])
    xsec = np.zeros(eng.size)

    if species == 'H0' or species =='He1': 
        eta = np.zeros(eng.size)
        eta[ind_above] = 1./np.sqrt(eng[ind_above]/eng_thres[species] - 1.)
        xsec[ind_above] = (2.**9*np.pi**2*ele_rad**2/(3.*alpha**3)
            * (eng_thres[species]/eng[ind_above])**4 
            * np.exp(-4*eta[ind_above]*np.arctan(1./eta[ind_above]))
            / (1.-np.exp(-2*np.pi*eta[ind_above]))
            )
    elif species == 'He0':
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
    """Photoionization rate in cm^3 s^-1.
 
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
    atom : {None,'H0','He0','He1'}, optional
        Determines which photoionization rate is returned. The default value is ``None``, which returns the total rate.
     
    Returns
    -------
    ionrate : float
        The ionization rate of the particular species or the total ionization rate.
 
    """
    atoms = ['H0', 'He0', 'He1']
 
    xHe = xe - xH
    atom_densities = {'H0':nH*(1-xH)*rs**3, 'He0':(nHe - xHe*nH)*rs**3, 
        'He1':xHe*nH*rs**3}
 
    ion_rate = {atom: photo_ion_xsec(eng,atom) * atom_densities[atom] * c 
        for atom in atoms}
 
    if atom is not None:
        return ion_rate[atom]
    else:
        return sum([ion_rate[atom] for atom in atoms])

def tau_sobolev(rs):
    """Sobolev optical depth. 
    
    Parameters
    ----------
    rs : float
        Redshift (1+z). 
    Returns
    -------
    float
    """
    xsec = 2 * np.pi * 0.416 * np.pi * alpha * hbar * c ** 2 / me
    lya_omega = lya_eng / hbar

    return nH * rs ** 3 * xsec * c / (hubble(rs) * lya_omega)

# CMB

def CMB_spec(eng, temp):
    """CMB spectrum in number of photons/cm^3/eV. 

    Returns zero if the energy exceeds 500 times the temperature.

    Parameters
    ----------
    temp : float
        Temperature of the CMB in eV.
    eng : float or ndarray
        Energy of the photon in eV. 
    
    Returns
    -------
    phot_spec_density : ndarray
        Returns the number of photons/cm^3/eV. 

    """
    prefactor = 8*np.pi*(eng**2)/((ele_compton*me)**3)
    if eng/temp < 500:
        return prefactor/(np.exp(eng/temp) - 1)
    else:
        return 0