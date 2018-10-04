import sys
sys.path.append("../..")

import numpy as np
import darkhistory.physics as phys
import darkhistory.spec.spectools as spectools
import time

def get_kappa_2s(photspec):
    """ Compute kappa_2s for use in kappa_DM function

    Parameters
    ----------
    photspec : Spectrum object
        spectrum of photons. Assumed to be in dNdE mode. spec.toteng() should return Energy per baryon.

    Returns
    -------
    kappa_2s : float
        The added photoionization rate from the 1s to the 2s state due to DM photons.
    """
    # Convenient Variables
    eng = photspec.eng
    rs = photspec.rs
    Lambda = phys.width_2s1s
    Tcmb = phys.TCMB(rs)
    lya_eng = phys.lya_eng

    # Photon phase space density (E >> kB*T approximation)
    def Boltz(E):
        return np.exp(-E/Tcmb)

    bounds = spectools.get_bin_bound(eng)
    mid = spectools.get_indx(bounds, lya_eng/2)

    # Phase Space Density of DM
    f_nu = photspec.dNdE * phys.c**3 / (
        8 * np.pi * (eng/phys.hbar)**2
    )

    # Complementary (E - h\nu) phase space density of DM
    f_nu_p = np.zeros(mid)

    # Index of point complementary to eng[k]
    comp_indx = spectools.get_indx(bounds, lya_eng - eng[0])

    # Find the bin in which lya_eng - eng[k] resides. Store f_nu of that bin in f_nu_p.
    for k in np.arange(mid):
        while (lya_eng - eng[k]) < bounds[comp_indx]:
            comp_indx -= 1
        f_nu_p[k] = f_nu[comp_indx]

    # Setting up the numerical integration

    # Bin sizes
    diffs = np.append(bounds[1:mid], lya_eng/2) - np.insert(bounds[1:mid], 0, 0)
    diffs /= (2 * np.pi * phys.hbar)

    dLam_dnu = phys.get_dLam2s_dnu()
    rates = dLam_dnu(eng[:mid]/(2 * np.pi * phys.hbar))

    boltz = Boltz(eng[:mid])
    boltz_p = Boltz(lya_eng - eng[:mid])

    # The Numerical Integral
    kappa_2s = np.sum(
        diffs * rates * (f_nu[:mid] + boltz) * (f_nu_p + boltz_p)
    )/phys.width_2s1s - Boltz(lya_eng)

    return kappa_2s

def kappa_DM(photspec, xe):
    """ Compute kappa_DM of the modified tla.

    Parameters
    ----------
    photspec : Spectrum object
        spectrum of photons. Assumed to be in dNdE mode. spec.toteng() should return Energy per baryon.

    Returns
    -------
    kappa_DM : float
        The added photoionization rate due to products of DM.
    """
    eng = photspec.eng
    rs = photspec.rs
    x1s_times_R_Lya = phys.rate_2p1s_times_x1s(xe,rs)
    Lambda = phys.width_2s1s

    # The bin number containing 10.2eV
    lya_index = spectools.get_indx(eng, phys.lya_eng)

    # The bins between 10.2eV and 13.6eV
    exc_bounds = spectools.get_bounds_between(
        eng, phys.lya_eng, phys.rydberg
    )

    # Effect on 2p state due to DM products
    kappa_2p = (
        photspec.dNdE[lya_index] * phys.nB * rs**3 *
        np.pi**2 * (phys.hbar * phys.c)**3 / phys.lya_eng**2
    )

    # Effect on 2s state
    kappa_2s = get_kappa_2s(photspec)

    return (
        kappa_2p*3*x1s_times_R_Lya/4 + kappa_2s*(1-xe)*Lambda/4
    )/(3*x1s_times_R_Lya/4 + (1-xe)*Lambda/4)


#---- f_c functions ----#
#continuum
def getf_continuum(photspec, norm_fac):
    # All photons below 10.2eV get deposited into the continuum
    return photspec.toteng(
        bound_type='eng',
        bound_arr=np.array([0,phys.lya_eng])
    )[0] * norm_fac

#excitation
def getf_excitation(photspec, norm_fac, dt, xe, n, method):
    if((method == 'old') or (method == 'ion')):
        # All photons between 10.2eV and 13.6eV are deposited into excitation
        tot_excite_eng = (
            photspec.toteng(
                bound_type='eng',
                bound_arr=np.array([phys.lya_eng,phys.rydberg])
            )[0]
        )
        f_excite_HI = tot_excite_eng * norm_fac
    else:
        # Only photons in the 10.2eV bin participate in 1s->2p excitation.
        # 1s->2s transition handled more carefully.

        # Convenient variables
        kappa = kappa_DM(photspec, xe)

        f_excite_HI = (
            kappa * (3*phys.rate_2p1s_times_x1s(xe,photspec.rs)*phys.nH + phys.width_2s1s*n[0]) *
            phys.lya_eng * (norm_fac / phys.nB / photspec.rs**3 * dt)
        )
    return f_excite_HI

#HI, HeI, HeII ionization
def getf_ion(photspec, norm_fac, n, method):
    # The bin number containing 10.2eV
    lya_index = spectools.get_indx(photspec.eng, phys.lya_eng)
    # The bin number containing 13.6eV
    ryd_index = spectools.get_indx(photspec.eng, phys.rydberg)

    if method == 'old':
        # All photons above 13.6 eV deposit their 13.6eV into HI ionization
        tot_ion_eng = phys.rydberg * photspec.totN(
            bound_type='eng',
            bound_arr=np.array([phys.lya_eng, 10*photspec.eng[-1]])
        )
        f_HI = tot_ion_eng * norm_fac
        f_HeI = 0
        f_HeII = 0
    else:
        # Photons may also deposit their energy into HeI and HeII single ionization

        # Bin boundaries of photon spectrum capable of photoionization, and number of photons in those bounds.
        ion_bounds = spectools.get_bounds_between(photspec.eng, phys.rydberg)
        ion_Ns = photspec.totN(bound_type='eng', bound_arr=ion_bounds)

        # Probability of being absorbed within time step dt in channel a is P_a = \sigma(E)_a n_a c*dt
        ionHI, ionHeI, ionHeII = [phys.photo_ion_xsec(photspec.eng[ryd_index:],channel) * n[i]
                                  for i,channel in enumerate(['H0','He0','He1'])]

        # The first energy might be less than 13.6, meaning no photo-ionization.
        # The photons in this box are hopefully all between 13.6 and 24.6, so they can only ionize H
        if photspec.eng[ryd_index] < phys.rydberg:
            ionHI[0] = 1

        # Relative likelihood of photoionization of HI is then P_HI/sum(P_a)
        totList = ionHI + ionHeI + ionHeII + 1e-12
        ionHI, ionHeI, ionHeII = [ llist/totList for llist in [ionHI, ionHeI, ionHeII] ]

        f_HI, f_HeI, f_HeII = [
            np.sum(ion_Ns * llist * norm_fac)
            for llist in [phys.rydberg*ionHI, phys.He_ion_eng*ionHeI, 4*phys.rydberg*ionHeII]
        ]
    return (f_HI, f_HeI, f_HeII)


def compute_fs(photspec, x, dE_dVdt_inj, time_step, method='old'):
    """ Compute f(z) fractions for continuum photons, photoexcitation of HI, and photoionization of HI, HeI, HeII

    Given a spectrum of deposited photons, resolve its energy into continuum photons,
    HI excitation, and HI, HeI, HeII ionization in that order.

    Parameters
    ----------
    photspec : Spectrum object
        spectrum of photons. Assumed to be in dNdE mode. spec.toteng() should return energy per baryon per time.
    x : list of floats
        number of (HI, HeI, HeII) divided by nH at redshift photspec.rs
    dE_dVdt_inj : float
        energy injection rate DM, dE/dVdt |_inj
    method : {'old','ion','new'}
        'old': All photons >= 13.6eV ionize hydrogen, within [10.2, 13.6)eV excite hydrogen, < 10.2eV are labelled continuum.
        'ion': Same as 'old', but now photons >= 13.6 can ionize HeI and HeII also.
        'new': Same as 'ion', but now [10.2, 13.6)eV photons treated more carefully.

    Returns
    -------
    tuple of floats
        Ratio of deposited energy to a given channel over energy deposited by DM.
        The order of the channels is {continuum photons, HI excitation, HI ionization, HeI ion, HeII ion}
    """
    chi = phys.nHe/phys.nH
    xHeIII = chi - x[1] - x[2]
    xHII = 1 - x[0]
    xe = xHII + x[2] + 2*xHeIII
    n = x * phys.nH * photspec.rs**3

    # norm_fac converts from total deposited energy to f_c(z) = (dE/dVdt)dep / (dE/dVdt)inj
    norm_fac = phys.nB * photspec.rs**3 / time_step / dE_dVdt_inj

    f_continuum = getf_continuum(photspec, norm_fac)
    f_excite_HI = getf_excitation(photspec, norm_fac, time_step, xe, n, method)
    f_HI, f_HeI, f_HeII = getf_ion(photspec, norm_fac, n, method)

    return np.array([f_continuum, f_excite_HI, f_HI, f_HeI, f_HeII])