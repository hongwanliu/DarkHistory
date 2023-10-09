import numpy as np
from scipy.interpolate import interp1d, interp2d, UnivariateSpline
from scipy.integrate import solve_ivp
from scipy.misc import derivative
from numba import jit, njit
from tqdm import tqdm_notebook as tqdm

import darkhistory.physics as phys

#####################################################
#####################################################
# Units are energy and temperature in eV,           #
# density in cm$^{-3}$, and halo mass in $M_\odot$. #
#####################################################
#####################################################

BHmin = 0.754 # photodetachment energy for H- in eV
kB = phys.kB
hbar = phys.hbar
hplanck = hbar * 2*np.pi
c = phys.c
m_e = phys.me
nH = phys.nH
g_per_Msun = 1.989e33
cm_per_kpc = 3.086e21  

#############################################
#############################################
# Evolution equations for H_2, x_e, and T_m #
# These are all taken from                  #
# Sec. 3 of of astro-ph/0606437             #
#############################################
#############################################

@njit
def photo_det_xsec(eng): # cross-section for H- photodetachment in cm^2
    x = eng / BHmin
    return 3.486e-16 * (x-1)**(3/2) / x**3.11

### Rates
@njit
def k1(T): # Rate for H + e- -> H- + photon [cm^3 s^-1]
    Tm = T / kB
    return 3e-16 * (Tm / 300)**0.95 * np.exp(-Tm / 9320) 

@njit
def k2(T): # Rate for H + H- -> H2 + e- [cm^3 s^-1]
    Tm = T / kB
    return 1.5e-9 * (Tm / 300)**(-0.1)

@njit
def k3(T): # Rate for H+ + H- -> 2H [cm^3 s^-1]
    Tm = T / kB
    return 4e-8 * (Tm / 300)**(-0.5) 

@njit
def kn1_th(T): # Rate for photodetachment by CMB photons [s^-1]
    return 4 * (m_e * T / 2 / np.pi / hbar**2 / c**2)**(3/2) * np.exp(-BHmin / T) * k1(T)

def kn1_nt(distortion_specs, rs): # Rate for photodetachment by spectral distortions [s^-1]
    # distortion_specs should be the spectral distortion at rs
    n = nH * rs**3
    engs = distortion_specs.eng
    integrand = distortion_specs.dNdE * photo_det_xsec(engs)
    return n * c * np.trapz(integrand[engs>BHmin], x=engs[engs>BHmin])

def kn1_nt_alt(distortion_specs, rs): # Alternative calculation that uses Spectra's grid_vals
    n = phys.nH * rs**3
    engs = distortion_specs.eng
    integrand = distortion_specs.grid_vals * photo_det_xsec(engs)
    return n * phys.c * np.trapz(integrand[:, engs>BHmin], x=engs[engs>BHmin]) # integration from BHmin to infinity

def kn1(T, dists, rs): # Rate for photodetachment [s^-1]
    if dists == 0:
        return kn1_th(T)
    else:
        return kn1_th(T) + kn1_nt(dists, rs)

# Approximate rate for photodissociation of H2 [s^-1]
# in the optically thin regime (low density).
# Taken from 2212.11100
def kH2(dists, rs): 
    # Average flux over LW band
    LW_inds = np.argwhere((dists.eng > 11.2)*(dists.eng < 13.6))
    dNdlnE = np.mean(dists.eng[LW_inds] * dists.dNdE[LW_inds])

    # Normalize correctly
    convert = (phys.nB * rs**3) * hplanck * phys.c / (4*np.pi) * phys.ele * 1e7
    J_LW = convert * dNdlnE # erg s^-1 cm^-2 Hz^-1 sr^-1
    return 1.39e-12 * J_LW / 1e-21

# Alternative calculation that uses Spectra's grid_vals
def kH2_alt(dists, rs): 
    # Average flux over LW band
    LW_inds = np.argwhere((dists.eng > 11.2)*(dists.eng < 13.6))
    if dists.spec_type == 'N':
        dists.switch_spec_type()
    dNdlnE = np.transpose(np.mean(dists.eng[LW_inds] * dists.grid_vals[:,LW_inds], axis=1))

    # Normalize correctly
    convert = (phys.nB * rs**3) * hplanck * phys.c / (4*np.pi) * phys.ele * 1e7
    J_LW = convert * dNdlnE.flatten() # erg s^-1 cm^-2 Hz^-1 sr^-1
    return 1.39e-12 * J_LW / 1e-21

def dxH2_dt_IGM(xHII, xHeII, xHeIII, T_m, rs, dists=0): # s^-1
    """Rate of xH2 formation from the H- pathway in s\ :sup:`-1`\ .
    This function is meant for IGM calculations.
 
    Parameters
    ----------
    xHII : float
        Fraction of ionized hydrogen
    xHeII : float
        Fraction of singly ionized helium (relative to hydrogen)
    xHeIII : float
        Fraction of doubly ionized helium (relative to hydrogen)
    T_m : float
        Matter temperature in eV
    rs : float
        redshift
    dists : float or Spectra
        dNdE spectra of nonthermal distortion to CMB blackbody at each redshift; 
        0 if there is no distortion
 
    Returns
    -------
    float
        H2 formation rate in s\ :sup:`-1`\
    """
    xe = xHII + xHeII + 2*xHeIII
    xHI = 1-xHII
    n = phys.nH * rs**3

    kn1_IGM = kn1_th(phys.TCMB(rs)) + kn1_nt_alt(dists, rs)
    
    rate_form = k1(T_m) * k2(T_m) * xe * xHI**2 * n**2 / (k2(T_m)*xHI*n + kn1_IGM + k3(T_m)*xHeII*n)
    rate_dest = 0
    return rate_form + rate_dest

def dxH2_dz(xe, xH2, Tm, n, rs, dists=0, LW=False): # s^-1
    """Rate of xH2 formation from the H- pathway in terms of redshift.
 
    Parameters
    ----------
    xe : float
        Free electron fraction
    Tm : float
        Matter temperature in eV
    n   : float
        Hydrogen number density in cm^-3
    rs : float
        redshift
    dists : float or Spectra
        dNdE spectrum of nonthermal distortion to CMB blackbody at rs; 
        0 if there is no distortion
    LW : bool
        If True, includes LW photodissociation of H2
 
    Returns
    -------
    float
        H2 formation rate in s\ :sup:`-1`\
    """
    xHI = 1-xe
    xHeII = 0
    TCMB = phys.TCMB(rs)
    rate_form = k1(Tm) * k2(Tm) * xe * xHI**2 * n**2 / (k2(Tm)*xHI*n + kn1(TCMB, dists, rs) + k3(Tm)*xHeII*n)
    if LW:
        rate_dest = kH2(dists, rs) * xH2
    else:
        rate_dest = 0
    return (rate_form - rate_dest) * phys.dtdz(rs)

# H2 formation rate from Tegmark et al
def dxH2_dz2(xe, xH2, T, n, rs): # s^-1
    Tm = T / phys.kB
    TCMB = phys.TCMB(rs) / phys.kB
    k2=1.83e-18*Tm**0.88
    k3=1.3e-9
    k4=0.114*TCMB**2.13*np.exp(-8650/TCMB)
    k5=1.85e-23*Tm**1.8
    k6=6.4e-10
    k7=6.36e5*np.exp(-71600/TCMB)
    km = (
        k2 * k3 / (k3 + k4/(1-xe)/n) + k5 * k6 / (k6 + k7/(1-xe)/n)
    )
    return km * n*(1-xe-2*xH2)*xe * phys.dtdz(rs)

# Ionization equation from Tegmark et al
# def dxe_dz(xe, T, n, rs):
#     Tm = T / phys.kB
#     return (
#         -(1.88e-10 * (Tm)**-0.64)
#         * n * xe**2
#     ) * phys.dtdz(rs)

# Ionization equation taken from DarkHistory
def dxe_dz(xe, T, n, rs, DM_switch=False, DM_args=None, f_suppress=False):
    peebC = phys.peebles_C(xe, rs)
    alpha = phys.alpha_recomb(T, 'HI')
    beta_ion = phys.beta_ion(T, 'HI')
    
    if DM_switch:
        n_IGM = phys.nH * rs**3
        mDM, inj_param, inj_type, inj_particle, f_data = DM_args
        f_H_ion = f_data['H ion'] #phys.f_std(mDM, rs, inj_particle=inj_particle, inj_type=inj_type, channel='H ion')
        f_H_exc = f_data['He ion'] #phys.f_std(mDM, rs, inj_particle=inj_particle, inj_type=inj_type, channel='exc')
        inj_rate = phys.inj_rate(inj_type, rs, mDM=mDM, sigmav=inj_param, lifetime=inj_param)
        dm_term = (
            f_H_ion(rs) * inj_rate / (phys.rydberg * n_IGM)
            + (1. - peebC) * (f_H_exc(rs) * inj_rate / (phys.lya_eng * n_IGM))
        )
        if f_suppress:
            dm_term *= n_IGM / n
    else:
        dm_term = 0
    
    return (
     - peebC * (
         alpha * xe*xe*n
         - 4 * beta_ion * (1-xe)
         * np.exp(-phys.lya_eng/phys.TCMB(rs))
     ) + dm_term
    ) * phys.dtdz(rs)

def dTm_dz(xe, xH2, T, n, dndt, rs, vir_switch=False, H2_cool_rate='new', DM_switch=False, DM_args=None, f_suppress=False):
    Tm = T / phys.kB
    T3 = Tm / 1e3
    TCMB = phys.TCMB(rs) / phys.kB
    
    adiabatic = (2/3) * Tm * dndt / n # K/s
    compton = xe * (4.91e-22 * TCMB**4) * (TCMB - Tm) # K/s
    line = - (
        7.5e-19 * n**2 * xe * (1-xe) * (np.exp(-118348/Tm) - np.exp(-118348/TCMB)) # erg / cm^3 / s
        * 1e-7 / phys.ele / phys.kB # K / erg
        / (3/2 * n * (1 + phys.chi + xe)) # cm^3
    ) 
    
    if DM_switch:
        mDM, inj_param, inj_type, inj_particle, f_data = DM_args
        n_IGM = phys.nH * rs**3
        f_heat = f_data['heat'] #phys.f_std(mDM, rs, inj_particle=inj_particle, inj_type=inj_type, channel='heat')
        inj_rate = phys.inj_rate(inj_type, rs, mDM=mDM, sigmav=inj_param, lifetime=inj_param)
        dm_term = f_heat(rs) * inj_rate / (3/2 * n_IGM * (1 + phys.chi + xe))
        if f_suppress:
            dm_term *= n_IGM / n
    else:
        dm_term = 0
    
    # OPTIONS FOR MOLECULAR COOLING RATE
    # From Tegmark et al
    # H2 = - Tm * (1 + 10*T3**(7/2)/(60+T3**4)) * (xH2/n) * np.exp(-512/Tm) / (48200 * 365 * 24 * 3600) # K/s
    Lr = (
        9.5e-22 * T3**3.76 * np.exp(-(0.13 / T3)**3) / (1 + 0.12 * T3**2.1)
        + 3e-24 * np.exp(-0.51/(Tm/1e3))
    ) / n
    Lv = (
        6.7e-19 * np.exp(-5.86/T3)
        + 1.6e-18 * np.exp(-11.7/T3)
    ) / n
    
    if H2_cool_rate=='new':
        # from Galli and Palla 1998
        L_LTE = Lr + Lv
        L_nH0 = 10**(
            - 103 + 97.59*np.log10(Tm) - 48.05*np.log10(Tm)**2 
            + 10.80*np.log10(Tm)**3 - 0.9032*np.log10(Tm)**4
        )
        LH2 = L_LTE / (1 + L_LTE / L_nH0) # erg / cm^3 / s
        H2 = - (
            LH2 * xH2 * n # erg / s
            * 1e-7 / phys.ele / phys.kB # K / erg
        )
    elif H2_cool_rate=='old':
        # Alternate derivation from Tegmark et al
        def gam(J, T3):
            return ((1e-11*T3**1/2)/(1 + 60*T3**-4) + 1e-12*T3) * (0.33 + 0.9*np.exp(-((J-3.5)/0.9)**2))
        E20 = phys.kB * 512
        E31 = (5/3) * E20
        L_nH0_teg = (
            (5/4) * gam(2, T3) * E20 * np.exp(-E20/T)
            + (7/4) * gam(2, T3) * E31 * np.exp(-E31/T)
        ) * 1e7 * phys.ele # erg cm^3 / s
        LH2_teg = Lr / (1 + Lr/L_nH0_teg)
        H2 = - (
            LH2_teg * xH2 * n # erg / s
            * 1e-7 / phys.ele / phys.kB # K / erg
        )
    else:
        print("Need to specify H2 cooling rate.")
    
    # After virialization, turn off compton cooling
    # Gives conservative estimate for critical T_vir/M_halo for collapse
    if not vir_switch:
        return phys.kB * (adiabatic + compton + line + dm_term / phys.kB + H2) * phys.dtdz(rs)
    else:
        return phys.kB * (adiabatic + line + dm_term / phys.kB + H2) * phys.dtdz(rs)
    
    
##############################################
##############################################
# Fitting functions for top-hat halo profile #
##############################################
##############################################

rho_DM = phys.rho_DM
rho_baryon = phys.rho_baryon

# Total energy density in matter [eV / cm^3]
@njit
def rho_TH(rs, rs_vir):
    A = rs_vir/rs
    rho_0 = rho_DM + rho_baryon
    return rho_0 * rs**3 * np.exp(1.9*A / (1-0.75*A**2)) # sign error in Eqn 23 of astro-ph/9603007??

# Number density of hydrogen nuclei [cm^-3]
@njit
def n_TH(rs, rs_vir):
    norm_fac = nH / (rho_DM + rho_baryon)
    return norm_fac * rho_TH(rs, rs_vir)

# Time derivative of number density of hydrogen nuclei [cm^-3 s^-1]
def dn_dt_TH(rs, rs_vir):
    if isinstance(rs, float):
        rs_pert = np.array([1.001*rs, 0.999*rs])
        n_pert = n_TH(rs_pert, rs_vir)
        return (n_pert[1] - n_pert[0]) / (rs_pert[1] - rs_pert[0]) / phys.dtdz(rs)
    else:
        n_interp = UnivariateSpline(rs[::-1], n_TH(rs, rs_vir)[::-1])
        return n_interp.derivative()(rs) / phys.dtdz(rs)
    

##############################################
##############################################
# Halo integration and virialization         #
##############################################
##############################################
    
# Virialized quantities
def rho_vir(rs): 
    return 18*np.pi**2 * (phys.rho_DM + phys.rho_baryon) * rs**3
# def T_vir(rs, M):
#     return 485 * phys.h**(2/3) * (M / 1e4)**(2/3) * (rs / 100) * phys.kB
def T_vir(rs, M, mu=0.6, OmM=phys.omega_m):
    return (
        phys.G * phys.mp / 2 / phys.kB / phys.c**2
        * (1e8 / 2 / cm_per_kpc**2 / phys.G)**(1/3)
        * mu # scale molecular weight
        * (g_per_Msun*M)**(2/3) # scale M_halo
        * (18*np.pi**2)**(1/3) # scale virial density
        * OmM**(1/3) # scale Omega_m
        * rs # scale redshift
    ) * phys.kB

# Conversion between virial temperature and halo mass
# Inverse of T_vir(rs, M)
def M_given_T(rs, T, mu=0.6, OmM=phys.omega_m):
    return (
        T / phys.kB / (
            phys.G * phys.mp / 2 / phys.kB / phys.c**2
            * (1e8 / 2 / cm_per_kpc**2 / phys.G)**(1/3)
            * mu
            * (18*np.pi**2)**(1/3) # scale virial density
            * OmM**(1/3) # scale Omega_m
            * rs # scale redshift
        )
    )**(3/2) / g_per_Msun
    # return 1e4 * (T / (485 * phys.h**(2/3) * (rs / 100) * phys.kB))**(3/2)

# Estimate of pressure determined density from Tegmark et al
# I think this expression is wrong??? Does not cause virialization at T_vir
def rho_pres(rs, M):
    rho_0 = phys.rho_DM + phys.rho_baryon
    return rho_0 * rs**3 * (1 + (6/5)*T_vir(rs, M) / phys.Tm_std(rs))**(3/2)

# Conditions for virialization
def vir_event(rs, var, rs_vir, M):
    xe, xH2, Tm = var
    rho = rho_TH(np.array(rs), rs_vir)
    T_cond = T_vir(rs_vir, M)
    rho_cond = rho_vir(rs_vir)
    
    # Make sure this condition doesn't kick in before density and temperature can grow
    # if (phys.Tm_std(rs) < 0.7*T_cond) and (rho_TH(2000, rs_vir) * (rs/2000)**3 < 0.7*rho_cond):
    if (n_TH(rs, rs_vir) > n_TH(rs*np.exp(0.01), rs_vir)):
        if (rho < rho_cond) and (Tm < T_cond):
            return min((rho_cond-rho)/rho_cond, (T_cond-Tm)/T_cond)
        elif rho > rho_cond:
            return (rho_cond-rho)/rho_cond
        else:
            return (T_cond-Tm)/T_cond
    else:
        return 1
vir_event.terminal = True

# Evolution equations all together
def evol_eqns(rs, var, rs_vir, M, early=False, vir_switch=False, nvir=None, dists=0,
              H2_form_rate='new', H2_cool_rate='new', DM_switch=False, DM_args=None, 
              f_suppress=False, LW=False):
    """Top-hat halo evolution equations
 
    Parameters
    ----------
    rs : float
        redshift
    var : ndarray of floats
        Variables to evolve. 
        If early==True, this is just xH2.
        If early==False, this is [xe, xH2, Tm].
    rs_vir : float
        virialization redshift
    M : float
        Halo mass
    early : bool
        If this flag is true, use 'standard' values for ionization/temperature,
        since the ODE system is stiff at early redshifts.
    vir_switch : bool
        If this flag is true, halo should be evolved assuming it is 
        virialized, i.e. density is held constant.
    nvir : float
        If vir_switch==True, this is the virialized density to use.
    dists : float or Spectrum
        dNdE spectrum of nonthermal distortion to CMB blackbody at rs; 
        0 if there is no distortion
    H2_form_rate : string
        If 'new', use the H_2 formation rate from astro-ph/0606437.
        If 'old', use the H_2 formation rate from astro-ph/9603007.
    H2_cool_rate : string
        If 'new', use the H_2 formation rate from astro-ph/9803315.
        If 'old', use the H_2 cooling rate from astro-ph/9603007.
    DM_switch : bool
        If true, includes energy injection by dark matter
    DM_args : tuple
        Properties of the dark matter model. Should include in order:
        Dark matter mass in [eV],
        Lifetime in [s] or cross-section in [cm^3 / s],
        type of injection,
        injected particle,
        energy deposition f's
    f_suppress : bool
        If true, suppresses f's by a factor of number density to IGM # density
    LW : bool
        If true, includes LW photodissociation of H2.
 
    Returns
    -------
    ndarray of floats
        Evolution equations for var of a top-hat halo
    """
    
    # Determine spectral distortion at this redshift
    if dists != 0 :
        rs_index, = np.where(dists.rs > rs)
        if len(rs_index) == 0:
            dists = 0
        else:
            dists = dists[rs_index[-1]]
    
    # Determine which H2 formation rate to use
    if H2_form_rate=='new':
        dxH2_dz_at_rs = lambda xe, xH2, Tm, n : dxH2_dz(xe, xH2, Tm, n, rs, dists=dists, LW=LW)
    elif H2_form_rate=='old':
        dxH2_dz_at_rs = lambda xe, xH2, Tm, n : dxH2_dz2(xe, xH2, Tm, n, rs)
    else:
        print("Need to specify H2 formation rate.")
    
    # Equations for early redshifts, when ODEs are stiff
    if early:
        xe = phys.x_std(rs)
        Tm = phys.Tm_std(rs)
        xH2 = var
        n = n_TH(np.array(rs), rs_vir)
        return dxH2_dz_at_rs(xe, xH2, Tm, n)
    
    # Later redshifts, before virialization
    if not vir_switch:
        xe, xH2, Tm = var
        n = n_TH(np.array(rs), rs_vir)
        dndt = dn_dt_TH(rs, rs_vir)
        return np.array([
            dxe_dz(xe, Tm, n, rs, DM_switch=DM_switch, DM_args=DM_args, f_suppress=f_suppress),
            dxH2_dz_at_rs(xe, xH2, Tm, n),
            dTm_dz(xe, xH2, Tm, n, dndt, rs, H2_cool_rate=H2_cool_rate, 
                   DM_switch=DM_switch, DM_args=DM_args, f_suppress=f_suppress)
        ])
    
    # After virizialization. Must provide virialized density, nvir.
    else:
        xe, xH2, Tm = var
        if nvir==None:
            raise ValueError(
                'Need to provide virialized density, nvir.'
            )
        return np.array([
            dxe_dz(xe, Tm, nvir, rs, 
                   DM_switch=DM_switch, DM_args=DM_args, f_suppress=f_suppress),
            dxH2_dz_at_rs(xe, xH2, Tm, nvir),
            dTm_dz(xe, xH2, Tm, nvir, 0, rs, vir_switch=vir_switch, H2_cool_rate=H2_cool_rate, 
                   DM_switch=DM_switch, DM_args=DM_args, f_suppress=f_suppress)
        ])

# Integration, with the virialization conditions
def halo_integrate(rs_vir, M_halo, init_H2, start_rs=3000., end_rs=5., early_rs=900., dists=0,
                  H2_form_rate='new', H2_cool_rate='new', DM_switch=False, DM_args=None, 
                  f_suppress=False, LW=False):
    """Get the evolution of a top-hat halo
 
    Parameters
    ----------
    rs_vir : float
        virialization redshift
    M_halo : float
        Halo mass
    init_H2 : ndarray of float
        initial condition for xH2
    start_rs : float
        Starting redshift for evolution
    end_rs : float
        Ending redshift for evolution
    early_rs : float
        Redshift above which the evolution eqns are stiff
    dists : float or Spectra
        dNdE spectra of nonthermal distortion to CMB blackbody; 
        0 if there is no distortion
    H2_form_rate : string
        If 'new', use the H_2 formation rate from astro-ph/0606437.
        If 'old', use the H_2 formation rate from astro-ph/9603007.
    H2_cool_rate : string
        If 'new', use the H_2 formation rate from astro-ph/9803315.
        If 'old', use the H_2 cooling rate from astro-ph/9603007.
    DM_switch : bool
        If true, includes energy injection by dark matter
    DM_args : tuple
        Properties of the dark matter model. Should include in order:
        Dark matter mass in [eV],
        Lifetime in [s] or cross-section in [cm^3 / s],
        type of injection,
        injected particle,
        energy deposition f's
    f_suppress : bool
        If true, suppresses f's by a factor of halo # density to IGM # density
    LW : bool
        If true, includes LW photodissociation of H2
 
    Returns
    -------
    dict
        Top hat halo solution. 't' gives the redshifts,
        'y' is an array containing xe, xH2, Tm, and n.
    float
        Redshift at which the halo is actually virialized.
    """
    rs_list = 10**np.arange(np.log10(start_rs), np.log10(end_rs), -0.01)
    high_rs_list = rs_list[rs_list >= early_rs]
    rs_list = rs_list[rs_list < early_rs]
    
    # Use standard ionization/temperature at early redshifts
    #     because the ode system is stiff
    early_eqns = lambda rs, var: evol_eqns(rs, var, rs_vir, M_halo, early=True, vir_switch=False, 
                                           dists=dists, H2_form_rate=H2_form_rate, H2_cool_rate=H2_cool_rate, 
                                           DM_switch=DM_switch, DM_args=DM_args, f_suppress=f_suppress)
    early_soln = solve_ivp(early_eqns, [high_rs_list[0], high_rs_list[-1]], 
                           init_H2, t_eval=high_rs_list, rtol=1e-10, atol=1e-10)
    y_early = np.array([
        phys.x_std(early_soln['t']),
        early_soln['y'][0],
        phys.Tm_std(early_soln['t'])
    ])
    # print(f"Is the integration at recombination good?  {early_soln['success']}")
    if not early_soln['success']:
        print(early_soln['message'])
    
    # Before virialization
    init_cond = [phys.x_std(rs_list[0]), early_soln['y'][0,-1], phys.Tm_std(rs_list[0])]
    halo_eqns = lambda rs, var: evol_eqns(rs, var, rs_vir, M_halo, vir_switch=False, 
                                          dists=dists, H2_form_rate=H2_form_rate, H2_cool_rate=H2_cool_rate, 
                                          DM_switch=DM_switch, DM_args=DM_args, f_suppress=f_suppress, LW=LW)
    halo_event = lambda rs, var: vir_event(rs, var, rs_vir, M_halo)
    halo_event.terminal = True
    halo_soln = solve_ivp(halo_eqns, [rs_list[0], rs_list[-1]], 
                          init_cond, t_eval=rs_list, rtol=1e-10, atol=1e-10, events=halo_event)
    # print(f"Is the integration before virialization good?  {halo_soln['success']}")
    if not halo_soln['success']:
        print(halo_soln['message'])
    
    # After virialization
    rs_list_vir = 10**np.arange(np.log10(halo_soln['t'][-1])-0.01, np.log10(end_rs), -0.01)
    if T_vir(rs_vir, M_halo) > halo_soln['y'][2,-1]:
        T_next = T_vir(rs_vir, M_halo)
    else:
        T_next = halo_soln['y'][2,-1]
    init_cond_vir = [halo_soln['y'][0,-1], halo_soln['y'][1,-1], T_next]
    nvir = n_TH(halo_soln['t_events'][0][0], rs_vir)
    halo_eqns_vir = lambda rs, var: evol_eqns(rs, var, rs_vir, M_halo, vir_switch=True, nvir=nvir, 
                                              dists=dists, H2_form_rate=H2_form_rate, H2_cool_rate=H2_cool_rate, 
                                              DM_switch=DM_switch, DM_args=DM_args, f_suppress=f_suppress, LW=LW)
    halo_soln_vir = solve_ivp(halo_eqns_vir, [rs_list_vir[0], rs_list_vir[-1]], 
                              init_cond_vir, t_eval=rs_list_vir, rtol=1e-10, atol=1e-10)
    # print(f"Is the integration after virialization good?  {halo_soln_vir['success']}")
    if not halo_soln_vir['success']:
        print(halo_soln_vir['message'])
    
    # Stitch solutions together
    t_full = np.hstack((early_soln['t'], halo_soln['t'], halo_soln_vir['t']))
    y_full = np.hstack((y_early, halo_soln['y'], halo_soln_vir['y']))
    n_full = np.hstack((n_TH(early_soln['t'], rs_vir), 
                        n_TH(halo_soln['t'], rs_vir), 
                        np.ones_like(rs_list_vir)*nvir))
    y_full = np.vstack((y_full, n_full))
    result = {
        't' : t_full,
        'y' : y_full
    }
    return result, halo_soln_vir['t'][0]


##############################################
##############################################
# Halo collapse/parameter scan               #
##############################################
##############################################

# Does this halo succeed in collapsing?
def collapse_criterion(rs, Tm, rs_vir):
    T_interp = interp1d(rs, Tm)
    fac = 0.75
    print(f"using collapse factor {fac}")
    if T_interp(fac*rs_vir) < fac * T_interp(rs_vir):
        return True
    else:
        return False
    
def shooting_scheme(rs_vir_list, dists=0, H2_form_rate='new', H2_cool_rate='new', 
                    DM_switch=False, DM_args=None, f_suppress=False, LW=False):
    M_halo_list = np.zeros_like(rs_vir_list, dtype=float)
    T_vir_list = np.zeros_like(rs_vir_list, dtype=float)
    
    for ii, rs_vir in enumerate(tqdm(rs_vir_list)):
        #print(f"1+z_vir = {rs_vir:.0f}")
        init_H2 = [0]
        
        T_high = 3e4 * phys.kB
        M_halo_high = M_given_T(rs_vir, T_high)
        test_high, rs_vir_high = halo_integrate(
            rs_vir, M_halo_high, init_H2, start_rs=2000., end_rs=0.65*rs_vir,
            dists=dists, H2_form_rate=H2_form_rate, H2_cool_rate=H2_cool_rate,
            DM_switch=DM_switch, DM_args=DM_args, f_suppress=f_suppress, LW=LW
        )
        col_high = collapse_criterion(test_high['t'], test_high['y'][2], rs_vir_high)
        
        T_low = max(phys.TCMB(rs_vir), 2e2 * phys.kB)
        # If integrator is breaking with T_low, then raise T_low
        try:
            M_halo_low  = M_given_T(rs_vir, T_low)
            test_low, rs_vir_low = halo_integrate(
                rs_vir, M_halo_low, init_H2, start_rs=2000., end_rs=0.65*rs_vir,
                dists=dists, H2_form_rate=H2_form_rate, H2_cool_rate=H2_cool_rate,
                DM_switch=DM_switch, DM_args=DM_args, f_suppress=f_suppress, LW=LW
            )
        except:
            T_low *= 2
            M_halo_low  = M_given_T(rs_vir, T_low)
            test_low, rs_vir_low = halo_integrate(
                rs_vir, M_halo_low, init_H2, start_rs=2000., end_rs=0.65*rs_vir,
                dists=dists, H2_form_rate=H2_form_rate, H2_cool_rate=H2_cool_rate,
                DM_switch=DM_switch, DM_args=DM_args, f_suppress=f_suppress, LW=LW
            )
        
        col_low = collapse_criterion(test_low['t'], test_low['y'][2], rs_vir_low)

        if not(col_high and not col_low):
            if col_high and col_low:
                M_halo_list[ii] = M_halo_low
                T_vir_list[ii] = T_low
            else:
                print(f'Bad temperature range at 1+z={rs_vir:.1f}; Neither halos collapse.')
        else:
            while M_halo_high / M_halo_low > 1.05:
                M_halo_mid = np.sqrt(M_halo_high * M_halo_low)
                #print(f"Trying T_vir = {T_vir(rs_vir, M_halo_mid)/phys.kB:.0f} K")
                test_mid, rs_vir_mid = halo_integrate(
                    rs_vir, M_halo_mid, init_H2, start_rs=2000., end_rs=0.65*rs_vir,
                    dists=dists, H2_form_rate=H2_form_rate, H2_cool_rate=H2_cool_rate,
                    DM_switch=DM_switch, DM_args=DM_args, f_suppress=f_suppress, LW=LW
                )
                col_mid = collapse_criterion(test_mid['t'], test_mid['y'][2], rs_vir_mid)
                if col_mid:
                    M_halo_high = M_halo_mid
                else:
                    M_halo_low = M_halo_mid
            M_halo_list[ii] = np.sqrt(M_halo_high * M_halo_low)
            T_vir_list[ii] = T_vir(rs_vir, M_halo_list[ii])
            #print(f"Got M_halo = {M_halo_list[ii]:.2E} solar masses")
        # print("~~~")
    
    return {
        'rs' : rs_vir_list,
        'M_halo' : M_halo_list, 
        'T_vir' : T_vir_list
    }

def f_LW(phot_spec, ALW=2., bLW=0.6):
    """ Factor by which M_halo increases due to Lyman-Werner feedback.
    This expression is taken from 2110.13919.
 
    Parameters
    ----------
    phot_spec : Spectrum
        Photon spectrum under consideration.
    ALW : float
        Coefficient of LW feedback
    bLW : float
        Power index for LW feedback
 
    Returns
    -------
    float
        Dimensionless factor for LW feedback.
    """
    # Find photons within Lyman-Werner band
    convert = phys.nB * phot_spec.eng * hplanck * phys.c / (4*np.pi) * phys.ele * 1e4
    intensity = convert * phot_spec.dNdE # units of J s^{-1} m^{-2} Hz^{-1} sr^{-1}
    intensity_interp = interp1d(phot_spec.eng, intensity)
    
    # Average intensity over the band 
    # Normalize to get J21
    return 1 + ALW * J21**bLW


##############################################
##############################################
# Tools for packaging data for collapse scan #
##############################################
##############################################

def repack_distortions(DH_data, one_iter=False, iter=-1):
    """ Conveniently repackages 'distortions' output of DarkHistory
    into the total spectral distortion at each redshift step.
 
    Parameters
    ----------
    DH_data : dictionary
        Output of main.embarrassingly_parallel_evolve().
    one_iter : bool
        True if the DH_data has only one iteration.
    iter : int
        If one_iter==False, specifies the iteration to use.
 
    Returns
    -------
    float
        Total spectral distortion at each redshift step.
    """

    if one_iter:
        datum = DH_data['data']
    else:
        datum = DH_data['data'][iter]

    rs_list = datum['rs']
    summed_specs = datum['distortions'].copy()
    for ii, rs in enumerate(tqdm(rs_list)):
        temp_dist = datum['distortions'].copy()
        temp_dist.redshift(rs)
        weights = np.zeros_like(datum['rs'])
        weights[:ii] = 1
        summed_specs[ii] = temp_dist.sum_specs(weight=weights)

    return summed_specs

def repack_DH_data(DH_data, one_iter=False, iter=-1):
    """ Conveniently repackages dark matter parameters and f_c's
    for the halo evolution code
 
    Parameters
    ----------
    DH_data : dictionary
        Output of main.embarrassingly_parallel_evolve().
    one_iter : bool
        True if the DH_data has only one iteration.
    iter : int
        If one_iter==False, specifies the iteration to use.
 
    Returns
    -------
    tuple
        Dark matter parameters and list of f_c's as interpolation functions
    """

    if one_iter:
        datum = DH_data['data']
    else:
        datum = DH_data['data'][iter]

    # Make interpolation functions for f's
    f_list = {}
    for key in datum['f']:
        f_list[key] = interp1d(
            datum['rs'], datum['f'][key],
            bounds_error=False, fill_value=0
        )

    # Package DM information together
    repacked_data = (
        DH_data['DM_params']['mDM'], # mDM in [eV]
        DH_data['DM_params']['inj_param'], # lifetime in [s]
        DH_data['DM_params']['DM_process'], # injection type
        DH_data['DM_params']['pri'], # injected particle
        f_list, # energy deposition f's
    )
    return repacked_data