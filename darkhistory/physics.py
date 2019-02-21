""" Physics functions as well as constants.

"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import zeta

cross_check = True
# Fundamental constants
mp          = 0.938272081e9
"""Proton mass in eV."""
me          = 510998.9461
"""Electron mass in eV."""
hbar        = 6.58211951e-16
"""hbar in eV s."""
c           = 299792458e2
"""Speed of light in cm/s."""
if not cross_check:
    kB          = 8.6173324e-5
    """Boltzmann constant in eV/K."""
else:
    kB = 8.6173423e-5
alpha       = 1/137.035999139
"""Fine structure constant."""
ele         = 1.60217662e-19
"""Electron charge in C."""
if not cross_check:
    G = 6.70711 * 10**-39 * hbar * c**5 * 10**-18
    """Newton's constant in cm^5 s^-4 eV^-1"""
else:
    G = 6.6730e-8


# Atomic and optical physics

thomson_xsec = 6.652458734e-25
"""Thomson cross section in cm^2."""
stefboltz    = np.pi**2 / (60 * (hbar**3) * (c**2))
"""Stefan-Boltzmann constant in eV^-3 cm^-2 s^-1."""

## Hydrogen Atomic Properties

rydberg      = 13.60569253
"""Ionization potential of ground state hydrogen in eV."""
lya_eng      = rydberg*3/4
"""Lyman alpha transition energy in eV."""
lya_freq     = lya_eng / (2*np.pi*hbar)
"""Lyman alpha transition frequency in Hz."""
width_2s1s    = 8.22458
"""Hydrogen 2s to 1s decay width in s^-1."""
bohr_rad     = (hbar*c) / (me*alpha)
"""Bohr radius in cm."""

## Electron

ele_rad      = bohr_rad * (alpha**2)
"""Classical electron radius in cm."""
ele_compton  = 2*np.pi*hbar*c/me
"""Electron Compton wavelength in cm."""

## Helium Atomic Properties

He_ion_eng   = 24.5873891
"""Energy needed to singly ionize neutral He in eV."""


def He_exc_lambda(level):
    """n=2 excitation wavelength from ground state of neutral He.

    Parameters
    ----------
    level : {'23s', '21s', '23p', '21p'}
        The n = 2 levels of He. 

    Returns
    -------
    float
        Excitation wavelength in cm. 
    """
    # Taken from NIST Basic Atomic Spectroscopic Data
    
    # For 23p, there are J=0, 1, 2 states. Doesn't
    # matter for our purposes which one we take.
    exc_lambda_dict = {
        '23s': 1./159855.9745,
        '21s': 1./166277.4403,
        '23p': 1./169087.,
        '21p': 1./171134.8970
    }

    return exc_lambda_dict[level]

def He_exc_eng(level):
    """n=2 excitation energy from ground state of neutral He.

    Parameters
    ----------
    level : {'23s', '21s', '23p', '21p'}
        The n = 2 levels of He. 

    Returns
    -------
    float
        Excitation energy in eV. 
    """

    return 2*np.pi*hbar*c/He_exc_lambda(level)

# Hubble

h    = 0.6727
""" h parameter."""
if not cross_check:
    H0   = 100*h*3.241e-20
else:
    H0 = 1/(4.5979401e17)
""" Hubble parameter today in s^-1."""

# Omegas

if not cross_check:
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
else:
    kmperMpc = 3.08568025e19
    amuperg = 6.0221415e23
    eVperg = 5.60958921e32
    nh0 = 0.022068*(1e2/kmperMpc)**2*amuperg*3/(8*np.pi*G)

    omega_m      = 0.3175
    """ Omega of all matter today."""
    omega_rad    = 8e-5
    """ Omega of radiation today."""
    omega_lambda = 0.6825
    """ Omega of dark energy today."""
    omega_baryon = 8*np.pi*G/(3*H0**2) * (nh0/amuperg)
    """ Omega of baryons today."""
    omega_DM      = omega_m-omega_baryon
    #""" Omega of dark matter today."""

# Densities

rho_crit     = 1.05375e4*(h**2)
""" Critical density of the universe in eV/cm^3."""
if not cross_check:
    rho_DM       = rho_crit*omega_DM
    """ DM density in eV/cm^3."""
else:
    #rho_DM = omega_DM/(8*np.pi*G/(3*H0**2))*eVperg
    rho_DM=1274.4140
if not cross_check:
    rho_baryon   = rho_crit*omega_baryon
    """ Baryon density in eV/cm^3."""
    nB          = rho_baryon/mp
    """ Baryon number density in cm^-3."""
else:
    nB = nh0
    rho_baryon = nh0*mp

if not cross_check:
    YHe         = 0.250
    """Helium abundance by mass."""
else:
    YHe = .24
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

    rs_log_bin_width = get_log_bin_width(rs_vec)
    dtdz_vec = dtdz(rs_vec)

    return np.dot(
        xe_vec*nH*thomson_xsec*c*dtdz_vec,
        rs_vec**4*rs_log_bin_width
    )


# def get_inj_rate(inj_type, inj_fac):
#     """Dark matter injection rate function.

#     Parameters
#     ----------
#     inj_type : {'sWave', 'decay'}
#         The type of injection.
#     inj_fac : float
#         The prefactor for the injection rate, consisting of everything other than the redshift dependence.

#     Returns
#     -------
#     function
#         The function takes redshift as an input, and outputs the injection rate.
#     """

#     def inj_rate(rs):
#         if inj_type == 'sWave':
#             return inj_fac*(rs**6)
#         elif inj_type == 'decay':
#             return inj_fac*(rs**3)

#     return inj_rate

def inj_rate(inj_type, rs, mDM=None, sigmav=None, tau=None):
    """ Dark matter annihilation/decay energy injection rate.

    Parameters
    ----------
    inj_type : {'swave', 'decay'}
        Type of injection.
    rs : float
        The redshift of injection.
    mDM : float, optional
        DM mass in eV.
    sigmav : float, optional
        Annihilation cross section in cm^3 s^-1.
    tau : float, optional
        Decay lifetime in s.

    Returns
    -------
    float
        The dE/dV_dt injection rate in eV cm^-3 s^-1.

    """
    if inj_type == 'swave':
        return rho_DM**2*rs**6*sigmav/mDM
    elif inj_type == 'decay':
        return rho_DM*rs**3/tau

def alpha_recomb(T_m, species):
    """Case-B recombination coefficient.

    Parameters
    ----------
    T_m : float
        The matter temperature.
    species : {'HI', 'HeI_21s', 'HeI_23s'}
        The species of interest. 

    Returns
    -------
    float
        Case-B recombination coefficient in cm^3/s.

    Note
    ----
    For HeI, returns beta with respect to the 2s state,
    in agreement with convention in RECFAST.
    """

    if species == 'HI':

        # Fudge factor recommended in 1011.3758
        fudge_fac = 1.126
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
        Case-B photoionization coefficient in s^-1.

    Note
    ----
    For HeI, returns beta with respect to the 2s state,
    in agreement with convention in RECFAST.

    """
    de_broglie_wavelength = (
        c * 2*np.pi*hbar
        / np.sqrt(2 * np.pi * phys.me * T_rad)
    )
    
    if species == 'HI':
        return (
            (1/de_broglie_wavelength)**3
            * np.exp(-rydberg/4/T_rad) * alpha_recomb(T_rad, 'HI')
        )/4

    elif species == 'HeI_21s':
        E_21s_inf = He_ion_eng - He_exc_eng('21s')
        # print(E_21s_inf)
        # print(de_broglie_wavelength)
        return 4*(
            (1/de_broglie_wavelength)**3
            * np.exp(-E_21s_inf/T_rad) * alpha_recomb(T_rad, 'HeI_21s')
        )

    elif species == 'HeI_23s':
        E_23s_inf = He_ion_eng - He_exc_eng('23s')
        return (4/3)*(
            (1/de_broglie_wavelength)**3
            * np.exp(-E_23s_inf/T_rad) * alpha_recomb(T_rad, 'HeI_23s')
        )




# def betae(Tr):
#   # Case-B photoionization coefficient
#   thermlambda = c*(2*pi*hbar)/sqrt(2*pi*(mp*me/(me+mp))*Tr)
#   return alphae(Tr) * exp(-(rydberg/4)/Tr)/(thermlambda**3)

def rate_2p1s_times_x1s(rs):
    """returns the rate at which Ly_a photons are emitted without being immediately absorbed times (1-xe)

    Parameters
    ----------
    xe : float
        the ionization fraction ne/nH.
    rs : float
        the redshift in 1+z.

    Returns
    -------
    float
        R_Ly_alpha * (1-xe)
    """
    num = (
        8 * np.pi * hubble(rs)/
        (3*(nH * rs**3 * (c/lya_freq)**3))
    )
    #print(xe, num)
    return num

def peebles_C(xHII, rs):
    """Returns the Peebles C coefficient.

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
    # Net rate for 2s to 1s transition.
    rate_2s1s = width_2s1s

    rate_exc = 3 * rate_2p1s_times_x1s(rs)/4 + (1-xHII) * rate_2s1s/4

    rate_ion = (1-xHII) * beta_ion(TCMB(rs), 'HI')
    # rate_ion = beta_ion(TCMB(rs))
    
    # if rate_exc/(rate_exc + rate_ion) < 0.1:
    #     print(rate_exc/(rate_exc + rate_ion), xe, rs)

    return rate_exc/(rate_exc + rate_ion)



# Atomic Cross-Sections

def photo_ion_xsec(eng, species):
    """Photoionization cross section in cm^2.

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
        Cross section in cm^2.
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
    atom : {None,'HI','HeI','HeII'}, optional
        Determines which photoionization rate is returned. The default value is ``None``, which returns the total rate.

    Returns
    -------
    ionrate : float
        The ionization rate of the particular species or the total ionization rate.

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

def coll_exc_xsec(eng, species=None):
    """ Returns the collisional excitation rate. See 0906.1197.

    Parameters
    ----------
    eng : float or ndarray
        Abscissa of *kinetic* energies.
    species : {'HI', 'HeI', 'HeII'}
        Species of interest.

    Returns
    -------
    float or ndarray
        Collisional excitation cross section in cm^2.
    """
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
            E_exc  = He_exc_eng('23s')

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

def coll_ion_xsec(eng, species=None):
    """ Returns the collisional ionization rate. See 0906.1197.

    Parameters
    ----------
    eng : float or ndarray
        Abscissa of *kinetic* energies.
    species : {'HI', 'HeI', 'HeII'}
        Species of interest.

    Returns
    -------
    float or ndarray
        Collisional ionization cross section in cm^2.

    Note
    ----
    Returns the Arnaud and Rothenflug rate.

    """
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

    return xsec

def coll_ion_sec_elec_spec(in_eng, eng, species=None):
    """ Returns the secondary electron spectrum after collisional ionization. See 0910.4410.

    Parameters
    ----------
    in_eng : float
        The incoming electron energy.
    eng : ndarray
        Abscissa of *kinetic* energies.
    species : {'HI', 'HeI', 'HeII'}
        Species of interest.

    Returns
    -------
    ndarray
        Secondary electron spectrum. Total number of electrons = 2.

    Note
    ----
    Includes both the freed and initial electrons. Conservation of energy
    is not enforced, but number of electrons is.

    """

    from darkhistory.spec.spectrum import Spectrum
    from darkhistory.spec import spectools

    if species == 'HI':
        eps_i = 8.
        ion_pot = rydberg
    elif species == 'HeI':
        eps_i = 15.8
        ion_pot = He_ion_eng
    elif species == 'HeII':
        eps_i = 32.6
        ion_pot = 4*rydberg
    else:
        raise TypeError('invalid species.')

    if np.isscalar(in_eng):
        low_eng_elec_dNdE = 1/(1 + (eng/eps_i)**2.1)
        # This spectrum describes the lower energy electron only.
        low_eng_elec_dNdE[eng >= (in_eng - ion_pot)/2] = 0
        # Normalize the spectrum to one electron.

        low_eng_elec_spec = Spectrum(eng, low_eng_elec_dNdE)
        if np.sum(low_eng_elec_dNdE) == 0:
            # Either in_eng < in_pot, or the lowest bin lies
            # above the halfway point, (in_eng - ion_pot)/2.
            # Add to the lowest bin.
            return np.zeros_like(eng)

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

        low_eng_elec_dNdE = np.outer(
            np.ones_like(in_eng), 1/(1 + (eng/eps_i)**2.1)
        )

        low_eng_elec_dNdE[eng_mask >= (in_eng_mask - ion_pot)/2] = 0

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

        return low_eng_elec_N + high_eng_elec_N



        # This spectrum describes the lower energy electron only.

def elec_heating_engloss_rate(eng, xe, rs):
    """Returns the electron energy loss rate due to heating of the gas.

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

    Note
    -------
    See 0910.4410 for the expression. The units have e^2/r being in units of energy, so to convert to SI, we insert 1/(4*pi*eps_0)^2.
    """

    w = c*np.sqrt(1 - 1/(1 + eng/me)**2)
    ne = xe*nH*rs**3

    eps_0 = 8.85418782e-12 # in SI units

    prefac = 4*np.pi*ele**4/(4*np.pi*eps_0)**2
    # prefac is now in SI units (J^2 m^2). Convert to (eV^2 cm^2).
    prefac *= 100**2/ele**2

    # Comparison with Tracy's numfac: numfac == phys.ele**4/((4*np.pi*eps_0)**2*phys.me/phys.c**2)*(100**2/phys.ele**2)/phys.c
    # Because she factored out beta, and left m_e c in numfac.

    # zeta_e = 7.40e-11*nB*rs**3
    zeta_e = 7.40e-11*ne
    coulomb_log = np.log(4*eng/zeta_e)

    # must use the mass of the electron in eV m^2 s^-2.
    return prefac*ne*coulomb_log/(me/c**2*w)


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

def get_dLam2s_dnu():
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
    coeff = 9 * alpha**6 * rydberg /(
        2**10 * 2 * np.pi * hbar
    )
    #print(coeff)

    # coeff * psi(y) * dy = probability of emitting a photon in the window nu_alpha * [y, y+dy)
    # interpolating points come from Spitzer and Greenstein, 1951
    y = np.arange(0, 1.05, .05)
    psi = np.array([0, 1.725, 2.783, 3.481, 3.961, 4.306, 4.546, 4.711, 4.824, 4.889, 4.907,
                   4.889, 4.824, 4.711, 4.546, 4.306, 3.961, 3.481, 2.783, 1.725, 0])

    # evaluation outside of interpolation window yields 0.
    f = interp1d(y, psi, kind='cubic', fill_value=(0,0))
    def dLam2s_dnu(nu):
        return coeff * f(nu/lya_freq) * width_2s1s/8.26548398114 / lya_freq

    return dLam2s_dnu


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

    if not cross_check:
        fac = 2.7255
    else:
        fac = 2.725
    return fac * kB * rs

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

def CMB_N_density(T):
    """ CMB number density in cm^-3.

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
    """CMB energy density in eV/cm^3.

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

def A_2s(y):
    """2s to 1s two-photon decay probability density.

    A_2s(y) * dy = probability that a photon is emitted with a frequency in the range nu_lya * [y, y+dy)
    with the other photon constrained by h*nu_1 + h*nu_2 = 10.2 eV.

    Parameters
    ----------
    y : float
        nu / nu_lya, frequency of one of the photons divided by lyman-alpha frequency.

    Returns
    -------
    float or ndarray
        probability density of emitting those two photons.
    """
    return 0
