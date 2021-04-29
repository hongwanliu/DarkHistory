""" Total energy deposition from low-energy electrons and photons into the IGM.
  
  As detailed in Section III.F of the paper sub-3keV photons and electrons (dubbed low-energy photons and electrons) deposit their energy into the IGM in the form of hydrogen/helium ionization, hydrogen excitation, heat, or continuum photons, each of which corresponds to a channel, c.
"""

import sys
sys.path.append("../..")

import numpy as np

from darkhistory import physics as phys
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec import spectools

#---- f_c functions ----#
#continuum
def getf_continuum(photspec, norm_fac, cross_check=False):
    # All photons below 10.2eV get deposited into the continuum
    if not cross_check:
        return photspec.toteng(
            bound_type='eng',
            bound_arr=np.array([photspec.eng[0],phys.lya_eng])
        )[0] * norm_fac
    else:
        return np.dot(
            photspec.N[photspec.eng < 10.2],
            photspec.eng[photspec.eng < 10.2]*norm_fac
        )

#excitation
def getf_exc(photspec, norm_fac, method, cross_check=False):
    if((method == 'old') or (method=='He') or (method == 'ion') or True):
        # All photons between 11.2eV and 13.6eV are deposited into excitation
        # partial binning
        if not cross_check:
            tot_exc_eng = (
                photspec.toteng(
                    bound_type='eng',
                    bound_arr=np.array([phys.lya_eng,phys.rydberg])
                )[0]
            )
        else:
            tot_lya_eng = np.dot(
                photspec.N[(photspec.eng >= 10.2) & (photspec.eng <= 13.6)],
                photspec.eng[(photspec.eng >= 10.2) & (photspec.eng <= 13.6)]
            )
        f_exc = tot_exc_eng * norm_fac
    else:
        raise TypeError('option not supported yet')
        # Only photons in the 10.2eV bin participate in 1s->2p excitation.
        # 1s->2s transition handled more carefully.

        # Convenient variables
        #kappa = kappa_DM(photspec, xe)

        # Added this line since rate_2p1s_times_x1s function was removed.
        #rate_2p1s_times_x1s = (
        #    8 * np.pi * phys.hubble(photspec.rs)/
        #    (3*(phys.nH * photspec.rs**3 * (phys.c/phys.lya_freq)**3))
        #)

        #f_Lya = (
        #    kappa * (
        #        3*rate_2p1s_times_x1s*phys.nH + phys.width_2s1s_H*n[0]
        #    ) *
        #    phys.lya_eng * (norm_fac / phys.nB / photspec.rs**3 * dt)
        #)
    return f_exc

#HI, HeI, HeII ionization
def getf_ion(photspec, norm_fac, dt, n, method, cross_check=False):
    # The bin number containing 10.2eV
    lya_index = spectools.get_indx(photspec.eng, phys.lya_eng)
    # The bin number containing 13.6eV
    ryd_index = spectools.get_indx(photspec.eng, phys.rydberg)

    if ((method == 'old') | (method == 'no_He')):
        # All photons above 13.6 eV deposit their 13.6eV into HI ionization
        #!!! The factor of 10 is probably unecessary
        if not cross_check:
            tot_ion_eng = phys.rydberg * photspec.totN(
                bound_type='eng',
                bound_arr=np.array([phys.rydberg, 10*photspec.eng[-1]])
            )[0]
        else:
            tot_ion_eng = phys.rydberg*np.sum(
                photspec.N[photspec.eng > 13.6]
            )
        f_HI = tot_ion_eng * norm_fac
        f_HeI = 0
        f_HeII = 0
    elif method == 'He':

        # Neglect HeII photoionization
        # !!! Not utilizing partial binning!
        rates = np.array([
            n[i]*phys.photo_ion_xsec(photspec.eng, chan)
            for i,chan in enumerate(['HI', 'HeI'])
        ])

        norm_prob = np.sum(rates, axis=0)

        prob = np.array([
            np.divide(
                rate, norm_prob,
                out = np.zeros_like(photspec.eng),
                where=(norm_prob > 0)
            ) for rate in rates
        ])

        ion_eng_H = phys.rydberg * np.sum(prob[0] * photspec.N)

        ion_eng_He = phys.He_ion_eng * np.sum(prob[1] * photspec.N)

        f_HI   = ion_eng_H  * norm_fac
        f_HeI  = ion_eng_He * norm_fac
        f_HeII = 0

    else:
        # Photons may also deposit their energy into HeI and HeII single ionization
        # !!! Not utilizing partial binning!
        rates = np.array([
            n[i]*phys.photo_ion_xsec(photspec.eng, chan)
            for i,chan in enumerate(['HI', 'HeI', 'HeII'])
        ])

        norm_prob = np.sum(rates, axis=0)

        prob = np.array([
            np.divide(
                rate, norm_prob,
                out = np.zeros_like(photspec.eng),
                where=(norm_prob > 0)
            ) for rate in rates
        ])

        ion_eng_H    = phys.rydberg    * np.sum(prob[0] * photspec.N)
        ion_eng_HeI  = phys.He_ion_eng * np.sum(prob[1] * photspec.N)
        ion_eng_HeII = 4*phys.rydberg  * np.sum(prob[2] * photspec.N)

        f_HI   = ion_eng_H    * norm_fac
        f_HeI  = ion_eng_HeI  * norm_fac
        f_HeII = ion_eng_HeII * norm_fac

    return (f_HI, f_HeI, f_HeII)


def compute_fs(photspec, x, dE_dVdt_inj, dt, method='old', cross_check=False):
    """ Compute f(z) fractions for continuum photons, photoexcitation of HI, and photoionization of HI, HeI, HeII

    Given a spectrum of deposited photons, resolve its energy into continuum photons,
    continuum photons, HI excitation, and HI, HeI, HeII ionization in that order.

    Parameters
    ----------
    photspec : Spectrum object
        spectrum of photons. spec.toteng() should return energy per baryon.
    x : list of floats
        number of (HI, HeI, HeII) divided by nH at redshift photspec.rs
    dE_dVdt_inj : float
        energy injection rate DM, dE/dVdt |_inj
    dt : float
        time in seconds over which these photons were deposited.
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
    norm_fac = phys.nB * photspec.rs**3 / dt / dE_dVdt_inj

    f_continuum = getf_continuum(photspec, norm_fac, cross_check)
    f_exc = getf_exc(photspec, norm_fac, method, cross_check)
    f_HI, f_HeI, f_HeII = getf_ion(photspec, norm_fac, dt, n, method, cross_check)

    return {'H ion': f_HI, 'HeI ion': f_HeI, 'HeII ion': f_HeII, 'H exc': f_exc, 'cont': f_continuum}

def propagating_lowE_photons_fracs(photspec, x, dt):
    """ Of the low energy photons, compute the fraction that did NOT get absorbed

    Given a spectrum of deposited photons...

    Parameters
    ----------
    photspec : Spectrum object
        spectrum of photons. spec.toteng() should return energy per baryon.
    x : list of floats
        number of (HI, HeI) divided by nH at redshift photspec.rs
    dt : float
        time in seconds over which these photons were deposited.

    Returns
    -------
    ndarray
        Returns fraction of photons that freestream through a time step with energy bins between 13.6eV and 54.4eV.  Photons in other bins are assumed to be absorbed.
    """
    n = phys.nH*photspec.rs**3*x

    ratios = np.array([
        n[i]*phys.photo_ion_xsec(photspec.eng, chan) * phys.c * dt
        for i,chan in enumerate(['HI', 'HeI'])
    ])


    prop_fracs = np.exp(-np.sum(ratios, axis=0))
    prop_fracs[photspec.eng>=54.4] = 0
    prop_fracs[photspec.eng<=13.6] = 0
    return prop_fracs

def get_ionized_elec(phot_spec, eleceng, x, method='He'):
    """ Compute spectrum of photoionized electrons of HI, HeI, HeII

    Given a spectrum of deposited electrons and photons, resolve their energy into
    H ionization, and ionization, H excitation, heating, and continuum photons in that order.

    Parameters
     ----------
    phot_spec : Spectrum object
        spectrum of photons. Assumed to be in dNdE mode. spec.totN() should return number *per baryon*.
    eleceng : ndarray
        abscissa of energies for electron spectrum
    x : list of floats
        number of (HI, HeI, HeII) divided by nH at redshift photon_spectrum.rs
    method : {'no_He', 'He_recomb', 'He', 'HeII'}
        Method for evaluating helium ionization. 

        * *'no_He'* -- all ionization assigned to hydrogen, or HeI if hydrogen reionization has completed;
        * *'He_recomb'* -- all photoionized helium atoms recombine; and 
        * *'He'* -- all photoionized helium atoms do not recombine;
        * *'HeII'* -- all ionization assigned to HeII.

        Default is 'no_He'. 
    separate_higheng : bool, optional
        If True, returns separate high energy deposition. 

    Returns
    -------
    Spectrum of photoionized electrons
    """

    if method == 'no_He':

        if x[0]>.005:
            ion_pot = phys.rydberg
        else:
            ion_pot = 4*phys.rydberg

        ion_bounds = spectools.get_bounds_between(
            phot_spec.eng, ion_pot
        )
        ion_engs = np.exp((np.log(ion_bounds[1:])+np.log(ion_bounds[:-1]))/2)

        ionized_elec = Spectrum(
            ion_engs,
            phot_spec.totN(bound_type="eng", bound_arr=ion_bounds),
            rs=phot_spec.rs,
            spec_type='N'
        )

        new_eng = ion_engs - ion_pot
        ionized_elec.shift_eng(new_eng)

        # rebin so that ionized_elec may be added to elec_spec
        ionized_elec.rebin(eleceng)

    elif (method == 'He') or (method == 'He_recomb'):

        n = phys.nH*phot_spec.rs**3*x

        rates = np.array([
            n[i]*phys.photo_ion_xsec(phot_spec.eng, chan) 
            for i,chan in enumerate(['HI', 'HeI', 'HeII'])
        ])

        norm_prob = np.sum(rates, axis=0)

        # Probability of photoionizing HI vs. HeI.
        prob = np.array([
            np.divide(
                rate, norm_prob, 
                out = np.zeros_like(phot_spec.eng),
                #where=(phot_spec.eng > phys.rydberg)
                where = norm_prob>0
            ) for rate in rates
        ])

        # Spectra weighted by prob.
        phot_spec_HI   = phot_spec*prob[0]
        phot_spec_HeI  = phot_spec*prob[1]
        phot_spec_HeII = phot_spec*prob[2]

        # Bin boundaries, including the lowest (13.6, 24.6) eV bin.
        ion_bounds_HI   = spectools.get_bounds_between(
            phot_spec.eng, phys.rydberg
        )
        ion_bounds_HeI  = spectools.get_bounds_between(
            phot_spec.eng, phys.He_ion_eng
        )
        ion_bounds_HeII = spectools.get_bounds_between(
            phot_spec.eng, 4*phys.rydberg
        )

        # Bin centers. 
        ion_engs_HI   = np.exp(
            (np.log(ion_bounds_HI[1:]) + np.log(ion_bounds_HI[:-1]))/2
        )
        ion_engs_HeI  = np.exp(
            (np.log(ion_bounds_HeI[1:]) + np.log(ion_bounds_HeI[:-1]))/2
        )
        ion_engs_HeII = np.exp(
            (np.log(ion_bounds_HeII[1:]) + np.log(ion_bounds_HeII[:-1]))/2
        )

        # Spectrum object containing secondary electron 
        # from ionization. 
        ionized_elec_HI   = Spectrum(
            ion_engs_HI,
            phot_spec_HI.totN(bound_type='eng', bound_arr=ion_bounds_HI),
            rs=phot_spec.rs, spec_type='N'
        )

        ionized_elec_HeI  = Spectrum(
            ion_engs_HeI,
            phot_spec_HeI.totN(bound_type='eng', bound_arr=ion_bounds_HeI),
            rs=phot_spec.rs, spec_type='N'
        )

        ionized_elec_HeII = Spectrum(
            ion_engs_HeII,
            phot_spec_HeII.totN(bound_type='eng', bound_arr=ion_bounds_HeII),
            rs=phot_spec.rs, spec_type='N'
        )

        # electron energy (photon energy - ionizing potential).
        new_eng_HI   = ion_engs_HI   - phys.rydberg
        new_eng_HeI  = ion_engs_HeI  - phys.He_ion_eng 
        new_eng_HeII = ion_engs_HeII - 4*phys.rydberg 

        # change the Spectrum abscissa to the correct electron energy.
        ionized_elec_HI.shift_eng(new_eng_HI)
        ionized_elec_HeI.shift_eng(new_eng_HeI)
        ionized_elec_HeII.shift_eng(new_eng_HeII)
        # rebin so that ionized_elec may be added to elec_spec.
        ionized_elec_HI.rebin(eleceng)
        ionized_elec_HeI.rebin(eleceng)
        ionized_elec_HeII.rebin(eleceng)

        ionized_elec = ionized_elec_HI + ionized_elec_HeI + ionized_elec_HeII

    elif method == 'HeII':

        eng_threshold = 4*phys.rydberg
        ion_bounds = spectools.get_bounds_between(
            phot_spec.eng, eng_threshold
        )
        ion_engs = np.exp((np.log(ion_bounds[1:])+np.log(ion_bounds[:-1]))/2)

        ionized_elec = Spectrum(
            ion_engs,
            phot_spec.totN(bound_type="eng", bound_arr=ion_bounds),
            rs=phot_spec.rs,
            spec_type='N'
        )

        new_eng = ion_engs - eng_threshold
        ionized_elec.shift_eng(new_eng)

        # rebin so that ionized_elec may be added to elec_spec
        ionized_elec.rebin(eleceng)

    else: 

        raise TypeError('invalid method.')

    return ionized_elec
