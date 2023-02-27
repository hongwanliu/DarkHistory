""" Total energy deposition from low-energy electrons and photons into the IGM.
  
  As detailed in Section III.F of the paper sub-3keV photons and electrons (dubbed low-energy photons and electrons) deposit their energy into the IGM in the form of hydrogen/helium ionization, hydrogen excitation, heat, or continuum photons, each of which corresponds to a channel, c.
"""

import sys
sys.path.append("../..")

import numpy as np

from darkhistory import physics as phys
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec import spectools
from darkhistory.low_energy import lowE_electrons
from darkhistory.low_energy import lowE_photons

def compute_fs(MEDEA_interp, elec_spec, phot_spec, x, dE_dVdt_inj, dt, highengdep, cmbloss=0, method='no_He', separate_higheng=True, cross_check=False):
    """ Compute f(z) fractions for continuum photons, photoexcitation of HI, and photoionization of HI, HeI, HeII

    Given a spectrum of deposited electrons and photons, resolve their energy into
    H ionization, and ionization, H excitation, heating, and continuum photons in that order.

    Parameters
     ----------
    phot_spec : Spectrum object
        spectrum of photons. Assumed to be in dNdE mode. spec.totN() should return number *per baryon*.
    elec_spec : Spectrum object
        spectrum of electrons. Assumed to be in dNdE mode. spec.totN() should return number *per baryon*.
    x : list of floats
        number of (HI, HeI, HeII) divided by nH at redshift photon_spectrum.rs
    dE_dVdt_inj : float
        DM energy injection rate, dE/dVdt injected.  This is for unclustered DM (i.e. without structure formation).
    dt : float
        time in seconds over which these spectra were deposited.
    highengdep : list of floats
        total amount of energy deposited by high energy particles into {H_ionization, H_excitation, heating, continuum} per baryon per time, in that order.
    cmbloss : float
        Total amount of energy in upscattered photons that came from the CMB, per baryon per time, (1/n_B)dE/dVdt. Default is zero.
    method : {'no_He', 'He_recomb', 'He'}
        Method for evaluating helium ionization. 

        * *'no_He'* -- all ionization assigned to hydrogen;
        * *'He_recomb'* -- all photoionized helium atoms recombine; and 
        * *'He'* -- all photoionized helium atoms do not recombine. 
    separate_higheng : bool, optional
        If True, returns separate high energy deposition. 

    Returns
    -------
    ndarray or tuple of ndarray
    f_c(z) for z within spec.rs +/- dt/2
    The order of the channels is {H Ionization, He Ionization, H Excitation, Heating and Continuum} 

    Notes
    -----
    The CMB component hasn't been subtracted from the continuum photons yet
    Think about the exceptions that should be thrown (elec_spec.rs should equal phot_spec.rs)
    """

    # np.array syntax below needed so that a fresh copy of eng and N are passed to the
    # constructor, instead of simply a reference.
    
    if elec_spec.rs < 0:
        raise ValueError('elec_spec.rs not set!')
    if phot_spec.rs < 0:
        raise ValueError('phot_spec.rs not set!')
    if elec_spec.rs != phot_spec.rs:
        raise ValueError('elec_spec.rs != phot_spec.rs')

    if method == 'no_He':

        ion_bounds = spectools.get_bounds_between(
            phot_spec.eng, phys.rydberg
        )
        ion_engs = np.exp((np.log(ion_bounds[1:])+np.log(ion_bounds[:-1]))/2)

        ionized_elec = Spectrum(
            ion_engs,
            phot_spec.totN(bound_type="eng", bound_arr=ion_bounds),
            rs=phot_spec.rs,
            spec_type='N'
        )

        new_eng = ion_engs - phys.rydberg
        ionized_elec.shift_eng(new_eng)

        # rebin so that ionized_elec may be added to elec_spec
        ionized_elec.rebin(elec_spec.eng)

        tmp_elec_spec = Spectrum(
            np.array(elec_spec.eng), np.array(elec_spec.N), 
            rs=elec_spec.rs, spec_type='N'
        )
        tmp_elec_spec.N += ionized_elec.N

        f_phot = lowE_photons.compute_fs(
            phot_spec, x, dE_dVdt_inj, dt, 'old', cross_check
        )
        #print(phot_spec.rs, f_phot[0], phot_spec.toteng(), cmbloss, dE_dVdt_inj)

        f_elec = lowE_electrons.compute_fs(
            MEDEA_interp, tmp_elec_spec, 1-x[0], dE_dVdt_inj, dt
        )

        # print('photons:', f_phot[2], f_phot[3]+f_phot[4], f_phot[1], 0, f_phot[0])
        # print('electrons:', f_elec[2], f_elec[3], f_elec[1], f_elec[4], f_elec[0])

        # f_low is {H ion, He ion, Lya Excitation, Heating, Continuum}
        f_low = np.array([
            f_phot[2]+f_elec[2],
            f_phot[3]+f_phot[4]+f_elec[3],
            f_phot[1]+f_elec[1],
            f_elec[4],
            f_phot[0]+f_elec[0] 
                - cmbloss*phys.nB*phot_spec.rs**3 / dE_dVdt_inj
        ])

        f_high = np.array([
            highengdep[0], 0, highengdep[1],
            highengdep[2], highengdep[3]
        ]) * phys.nB * phot_spec.rs**3 / dE_dVdt_inj

        if separate_higheng:
            return (f_low, f_high)
        else:
            return f_low + f_high

    elif method == 'He':

        # Neglect HeII photoionization. Photoionization rates.
        n = phys.nH*phot_spec.rs**3*x

        rates = np.array([
            n[i]*phys.photo_ion_xsec(phot_spec.eng, chan) 
            for i,chan in enumerate(['HI', 'HeI'])
        ])

        norm_prob = np.sum(rates, axis=0)

        # Probability of photoionizing HI vs. HeI.
        prob = np.array([
            np.divide(
                rate, norm_prob, 
                out = np.zeros_like(phot_spec.eng),
                where=(phot_spec.eng > phys.rydberg)
            ) for rate in rates
        ])

        # Spectra weighted by prob.
        phot_spec_HI  = phot_spec*prob[0]
        phot_spec_HeI = phot_spec*prob[1]

        # Bin boundaries, including the lowest (13.6, 24.6) eV bin.
        ion_bounds_HI = spectools.get_bounds_between(
            phot_spec.eng, phys.rydberg
        )
        ion_bounds_HeI = spectools.get_bounds_between(
            phot_spec.eng, phys.He_ion_eng
        )

        # Bin centers. 
        ion_engs_HI = np.exp(
            (np.log(ion_bounds_HI[1:]) + np.log(ion_bounds_HI[:-1]))/2
        )
        ion_engs_HeI = np.exp(
            (np.log(ion_bounds_HeI[1:]) + np.log(ion_bounds_HeI[:-1]))/2
        )

        # Spectrum object containing secondary electron 
        # from ionization. 
        ionized_elec_HI = Spectrum(
            ion_engs_HI,
            phot_spec_HI.totN(bound_type='eng', bound_arr=ion_bounds_HI),
            rs=phot_spec.rs, spec_type='N'
        )

        ionized_elec_HeI = Spectrum(
            ion_engs_HeI,
            phot_spec_HeI.totN(bound_type='eng', bound_arr=ion_bounds_HeI),
            rs=phot_spec.rs, spec_type='N'
        )

        # electron energy (photon energy - ionizing potential).
        new_eng_HI  = ion_engs_HI  - phys.rydberg
        new_eng_HeI = ion_engs_HeI - phys.He_ion_eng 

        # change the Spectrum abscissa to the correct electron energy.
        ionized_elec_HI.shift_eng(new_eng_HI)
        ionized_elec_HeI.shift_eng(new_eng_HeI)
        # rebin so that ionized_elec may be added to elec_spec.
        ionized_elec_HI.rebin(elec_spec.eng)
        ionized_elec_HeI.rebin(elec_spec.eng)

        tmp_elec_spec = Spectrum(
            np.array(elec_spec.eng), np.array(elec_spec.N),
            rs=elec_spec.rs, spec_type='N' 
        )

        tmp_elec_spec.N += (ionized_elec_HI.N + ionized_elec_HeI.N)

        f_phot = lowE_photons.compute_fs(
            phot_spec, x, dE_dVdt_inj, dt, 'helium', cross_check
        )
        f_elec = lowE_electrons.compute_fs(
            MEDEA_interp, tmp_elec_spec, 1-x[0], dE_dVdt_inj, dt
        )

        # f_low is {H ion, He ion, Lya Excitation, Heating, Continuum}
        f_low = np.array([
            f_phot[2]+f_elec[2],
            f_phot[3]+f_phot[4]+f_elec[3],
            f_phot[1]+f_elec[1],
            f_elec[4],
            f_phot[0]+f_elec[0] 
                - cmbloss*phys.nB*phot_spec.rs**3 / dE_dVdt_inj
        ])

        f_high = np.array([
            highengdep[0], 0, highengdep[1],
            highengdep[2], highengdep[3]
        ]) * phys.nB * phot_spec.rs**3 / dE_dVdt_inj

        if separate_higheng:
            return (f_low, f_high)
        else:
            return f_low + f_high

    elif method == 'He_recomb':

        # Neglect HeII photoionization. Photoionization rates.
        n = phys.nH*phot_spec.rs**3*x

        rates = np.array([
            n[i]*phys.photo_ion_xsec(phot_spec.eng, chan) 
            for i,chan in enumerate(['HI', 'HeI'])
        ])

        norm_prob = np.sum(rates, axis=0)

        # Probability of photoionizing HI vs. HeI.
        prob = np.array([
            np.divide(
                rate, norm_prob, 
                out = np.zeros_like(phot_spec.eng),
                where=(phot_spec.eng > phys.rydberg)
            ) for rate in rates
        ])

        # Spectra weighted by prob.
        phot_spec_HI  = phot_spec*prob[0]
        phot_spec_HeI = phot_spec*prob[1]

        # Bin boundaries, including the lowest (13.6, 24.6) eV bin.
        ion_bounds_HI = spectools.get_bounds_between(
            phot_spec.eng, phys.rydberg
        )
        ion_bounds_HeI = spectools.get_bounds_between(
            phot_spec.eng, phys.He_ion_eng
        )

        # Bin centers. 
        ion_engs_HI = np.exp(
            (np.log(ion_bounds_HI[1:]) + np.log(ion_bounds_HI[:-1]))/2
        )
        ion_engs_HeI = np.exp(
            (np.log(ion_bounds_HeI[1:]) + np.log(ion_bounds_HeI[:-1]))/2
        )

        # Spectrum object containing secondary electron 
        # from ionization. 
        ionized_elec_HI = Spectrum(
            ion_engs_HI,
            phot_spec_HI.totN(bound_type='eng', bound_arr=ion_bounds_HI),
            rs=phot_spec.rs, spec_type='N'
        )

        ionized_elec_HeI = Spectrum(
            ion_engs_HeI,
            phot_spec_HeI.totN(bound_type='eng', bound_arr=ion_bounds_HeI),
            rs=phot_spec.rs, spec_type='N'
        )

        # electron energy (photon energy - ionizing potential).
        new_eng_HI  = ion_engs_HI  - phys.rydberg
        new_eng_HeI = ion_engs_HeI - phys.He_ion_eng 

        # change the Spectrum abscissa to the correct electron energy.
        ionized_elec_HI.shift_eng(new_eng_HI)
        ionized_elec_HeI.shift_eng(new_eng_HeI)
        # rebin so that ionized_elec may be added to elec_spec.
        ionized_elec_HI.rebin(elec_spec.eng)
        ionized_elec_HeI.rebin(elec_spec.eng)

        tmp_elec_spec = Spectrum(
            np.array(elec_spec.eng), np.array(elec_spec.N),
            rs=elec_spec.rs, spec_type='N' 
        )

        tmp_elec_spec.N += (ionized_elec_HI.N + ionized_elec_HeI.N)

        # Every ionized helium recombines to produce an 11 eV electron. 
        recomb_elec = spectools.rebin_N_arr(
            np.array([phot_spec_HeI.totN()]), 
            np.array([phys.He_ion_eng - phys.rydberg]), elec_spec.eng
        )
        tmp_elec_spec.N += recomb_elec.N

        # Every photon that photoionizes goes into hydrogen ionization now.
        # We can just use 'old' to do this computation.
        f_phot = lowE_photons.compute_fs(
            phot_spec, x, dE_dVdt_inj, dt, 'old', cross_check
        )
        f_elec = lowE_electrons.compute_fs(
            MEDEA_interp, tmp_elec_spec, 1-x[0], dE_dVdt_inj, dt
        )

        # f_low is {H ion, He ion, Lya Excitation, Heating, Continuum}
        f_low = np.array([
            f_phot[2]+f_elec[2],
            f_phot[3]+f_phot[4]+f_elec[3],
            f_phot[1]+f_elec[1],
            f_elec[4],
            f_phot[0]+f_elec[0] 
                - cmbloss*phys.nB*phot_spec.rs**3 / dE_dVdt_inj
        ])

        f_high = np.array([
            highengdep[0], 0, highengdep[1],
            highengdep[2], highengdep[3]
        ]) * phys.nB * phot_spec.rs**3 / dE_dVdt_inj

        if separate_higheng:
            return (f_low, f_high)
        else:
            return f_low + f_high

    else: 

        raise TypeError('invalid method.')
