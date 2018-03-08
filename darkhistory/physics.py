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
"""Stefan-Boltzmann constant in eV^-3 cm^-2 s^-1."""
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

    dtdz_vec = dtdz(rs_vec)
    integrand = xe_vec * nH*rs_vec**3 * thomson_xsec * c * dtdz_vec

    return np.trapz(integrand, rs_vec)


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

def alpha_recomb(T_matter):
	"""Case-B recombination coefficient.

	Parameters
	----------
	T_matter : float
		The matter temperature.

	Returns
	-------
	float
		Case-B recombination coefficient in cm^3/s.
	"""

	# Fudge factor recommended in 1011.3758
	fudge_fac = 1.126

	return (
		fudge_fac * 1e-13 * 4.309 * (1.16405*T_matter)**(-0.6166)
		/ (1 + 0.6703 * (1.16405*T_matter)**0.5300)
	)

def beta_ion(T_rad):
	"""Case-B photoionization coefficient.

	Parameters
	----------
	T_rad : float
		The radiation temperature.

	Returns
	-------
	float
		Case-B photoionization coefficient in s^-1.

	"""
	reduced_mass = mp*me/(mp + me)
	de_broglie_wavelength = (
		c * 2*np.pi*hbar
		/ np.sqrt(2 * np.pi * reduced_mass * T_rad)
	)
	return (
		(1/de_broglie_wavelength)**3/4
		* np.exp(-rydberg/4/T_rad) * alpha_recomb(T_rad)
	)

# def betae(Tr):
# 	# Case-B photoionization coefficient
# 	thermlambda = c*(2*pi*hbar)/sqrt(2*pi*(mp*me/(me+mp))*Tr)
# 	return alphae(Tr) * exp(-(rydberg/4)/Tr)/(thermlambda**3)

def rate_factor(xe, rs)
    """returns numerator of the Peebles C coefficient

    Parameters
    ----------
    xe : float
        the ionization fraction ne/nH.
    rs : float
        the redshift in 1+z.

    Returns
    -------
    float
        Numerator of the Peebles C coefficient.
    """

	# Net rate for 2p to 1s transition.
	rate_2p1s = (
		8 * np.pi * hubble(rs)
		/(3*(nH*rs**3 * (1-xe) * (c/lya_freq)**3))
	)

	# Net rate for 2s to 1s transition.
	rate_2s1s = width_2s1s

    return (3*rate_2p1s/4 + rate_2s1s/4)


def peebles_C(xe, rs):
	"""Returns the Peebles C coefficient.

	This is the ratio of the total rate for transitions from n = 2 to the ground state to the total rate of all transitions, including ionization.

	Parameters
	----------
	xe : float
		The ionization fraction ne/nH.
	Tm : float
		The matter temperature.
	rs : float
		The redshift in 1+z.

	Returns
	-------
	float
		The Peebles C factor.
	"""

    rate_exc = rate_factor(xe, rs)

	# Net rate for ionization.
	rate_ion = beta_ion(TCMB(rs))

	# Rate is averaged over 3/4 of excited state being in 2p, 1/4 in 2s.
	return rate_exc/(rate_exc + rate_ion)



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

def CMB_eng_density(T):
    """CMB energy density in eV/cm^3.

    Parameters
    ----------
    T : float or ndarray
        CMB temperature

    Returns
    -------
    float or ndarray
        The energy density of the CMB.
    """

    return 4*stefboltz/c*T**4
