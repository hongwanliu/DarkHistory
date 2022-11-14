"""Function for calculating the electron cooling transfer function.
"""

import numpy as np
import darkhistory.physics as phys
import darkhistory.spec.transferfunction as tf
import darkhistory.spec.spectools as spectools

from config import load_data

from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec.spectra import Spectra

from darkhistory.electrons.ics.ics_spectrum import ics_spec
from darkhistory.electrons.ics.ics_engloss_spectrum import engloss_spec

from scipy.linalg import solve_triangular


def get_elec_cooling_tf(
    eleceng, photeng, rs, xHII, xHeII=0,
    raw_thomson_tf=None, raw_rel_tf=None, raw_engloss_tf=None,
    coll_ion_sec_elec_specs=None, coll_exc_sec_elec_specs=None,
    ics_engloss_data=None,
    method='new', H_states=None,
    check_conservation_eng=False, simple_ICS=False, verbose=False
):

    """Transfer functions for complete electron cooling through inverse Compton
    scattering (ICS) and atomic processes.

    Parameters
    ----------
    eleceng : ndarray, shape (m, )
        The electron kinetic energy abscissa.
    photeng : ndarray
        The photon energy abscissa.
    rs : float
        The redshift (1+z).
    xHII : float
        Ionized hydrogen fraction, nHII/nH.
    xHeII : float, optional
        Singly-ionized helium fraction, nHe+/nH. Default is 0.
    raw_thomson_tf : TransFuncAtRedshift, optional
        Thomson ICS scattered photon spectrum transfer function. If None, uses
        the default transfer function. Default is None.
    raw_rel_tf : TransFuncAtRedshift, optional
        Relativistic ICS scattered photon spectrum transfer function. If None,
        uses the default transfer function. Default is None.
    raw_engloss_tf : TransFuncAtRedshift, optional
        Thomson ICS scattered electron net energy loss transfer function.
        If None, uses the default transfer function. Default is None.
    coll_ion_sec_elec_specs : tuple of 3 ndarrays, shapes (m, m), optional
        Normalized collisional ionization secondary electron spectra, order HI,
        HeI, HeII, indexed by injected electron energy by outgoing electron
        energy. If None, the function calculates this. Default is None.
    coll_exc_sec_elec_specs : tuple of 3 ndarray, shapes (m, m), optional
        Normalized collisional excitation secondary electron spectra, order HI,
        HeI, HeII, indexed by injected electron energy by outgoing electron
        energy. If None, the function calculates this. Default is None.
    ics_engloss_data : EnglossRebinData
        An `EnglossRebinData` object which stores rebinning information (based
        on ``eleceng`` and ``photeng``) for speed. Default is None.
    method : {'old', 'MEDEA', 'new'}
        if method == 'old', see 0906.1197;
        if method == 'MEDEA',
            see Mon. Not. R. Astron. Soc. 422, 420–433 (2012);
        if method == 'new', same as MEDEA,
            but with more excited states from CCC database
        if method == 'eff', same as 'new'. 
    H_states : list of str
        Excited states to track.
    check_conservation_eng : bool
        If True, lower=True, checks for energy conservation. Default is False.
    simple_ICS : bool
        If True, calculates energy deposited into ICS from transfer function, 
        instead of the spectrum of secondary photons. Default is False.
    verbose : bool
        If True, prints energy conservation checks. Default is False.

    Returns
    -------

    tuple
        Transfer functions for electron cooling deposition and spectra.

    See Also
    ---------
    :class:`.TransFuncAtRedshift`
    :class:`.EnglossRebinData`
    :mod:`.ics`

    Notes
    -----

    The entries of the output tuple are (see Sec IIIC of the paper):

    0. The secondary propagating photon transfer function :math:`\\overline{\\mathsf{T}}_\\gamma`;
    1. The low-energy electron transfer function :math:`\\overline{\\mathsf{T}}_e`;
    2. Energy deposited into ionization :math:`\\overline{\\mathbf{R}}_\\text{ion}`;
    3. Energy deposited into excitation :math:`\\overline{\\mathbf{R}}_\\text{exc}`;
    4. Energy deposited into heating :math:`\\overline{\\mathbf{R}}_\\text{heat}`;
    5. Upscattered CMB photon total energy :math:`\\overline{\\mathbf{R}}_\\text{CMB}`, and
    6. Numerical error away from energy conservation.

    Items 2--5 are vectors that when dotted into an electron spectrum with
    abscissa ``eleceng``, return the energy deposited/CMB energy upscattered
    for that spectrum.

    Items 0--1 are :class:`.TransFuncAtRedshift` objects. For each of these
    objects ``tf`` and a given electron spectrum ``elec_spec``,
    ``tf.sum_specs(elec_spec)`` returns the propagating photon/low-energy
    electron spectrum after cooling.

    The default version of the three ICS transfer functions that are required
    by this function is provided in :mod:`.tf_data`.

    """

    if method == 'eff': 
        # 'eff' is used for using distortions to calculate f_exc, but still relies
        # on the same electron cooling method as 'new'. 
        method = 'new'

    id_mat = np.identity(eleceng.size)

    # Use default ICS transfer functions if not specified.
    if (
        (raw_thomson_tf is None) |
        (raw_rel_tf is None) |
        (raw_engloss_tf is None)
    ):
        ics_tf = load_data('ics_tf')

    if (raw_thomson_tf is None):
        raw_thomson_tf = ics_tf['thomson']

    if (raw_rel_tf is None):
        raw_rel_tf = ics_tf['rel']

    if (raw_engloss_tf is None):
        raw_engloss_tf = ics_tf['engloss']

    # atoms that take part in electron cooling process through ionization
    atoms = ['HI', 'HeI', 'HeII']
    exc_types = H_states+['HeI', 'HeII']

    # ionization and excitation energies
    ion_potentials = {
        'HI': phys.rydberg, 'HeI': phys.He_ion_eng, 'HeII': 4*phys.rydberg
    }

    exc_potentials = {state: phys.H_exc_eng(state) for state in H_states}
    exc_potentials['HeI'] = phys.He_exc_eng['23s']
    exc_potentials['HeII'] = 4*phys.lya_eng

    # Set the electron fraction and number densities
    xe = xHII + xHeII
    ns = {
        'HI': (1 - xHII)*phys.nH*rs**3,
        'HeI': (phys.nHe/phys.nH - xHeII)*phys.nH*rs**3,
        'HeII': xHeII*phys.nH*rs**3
    }

    # v/c of electrons
    beta_ele = np.sqrt(1 - 1/(1 + eleceng/phys.me)**2)

    # collisional atomic cross-sections
    coll_xsec = {'exc': phys.coll_exc_xsec, 'ion': phys.coll_ion_xsec}

    # get_species -- a convenience function
    def get_sp(exc):
        if exc[:2] == 'He':
            return exc
        else:
            return 'HI'

    # collisional excitation rates
    exc_rates = {exc: ns[get_sp(exc)] * beta_ele * phys.c *
                 coll_xsec['exc'](eleceng, get_sp(exc), method, exc)
                 for exc in exc_types}

    # Deposited energy in various channels
    deposited_exc_eng_arr = {exc: np.zeros_like(eleceng) for exc in exc_types}
    deposited_heat_eng_arr = np.zeros_like(eleceng)

    if coll_ion_sec_elec_specs is None:

        # Compute the (normalized) collisional ionization spectra.
        coll_ion_sec_elec_specs = {
            species: phys.coll_ion_sec_elec_spec(eleceng, eleceng, species)
            for species in atoms
        }

    if coll_exc_sec_elec_specs is None:

        # Make empty dictionaries
        coll_exc_sec_elec_specs = {}
        coll_exc_sec_elec_tf = {}
        # eng_underflow = 0
        # N_underflow = {}

        # Compute the (normalized) collisional excitation spectra.

        # Electron with energy eleceng produces a spectrum with one particle
        # of energy eleceng - exc_potential.
        for exc in exc_types:
            exc_pot = exc_potentials[exc]
            coll_exc_sec_elec_tf[exc] = tf.TransFuncAtRedshift(
                np.squeeze(id_mat[:, np.where(eleceng > exc_pot)]),
                in_eng=eleceng, rs=-1*np.ones_like(eleceng),
                eng=eleceng[eleceng > exc_pot] - exc_pot,
                dlnz=-1, spec_type='N'
            )

            # electrons downscattered below eleceng[0] unfortunately are lost
            #  to the code here we keep track of their energy and number of
            #  downscatters. First, we make sure that their residual energy
            #  goes into prompt heating -- keeping track of the rate of
            #  excitation in 1s
            deposited_heat_eng_arr += (
                coll_exc_sec_elec_tf[exc].eng_underflow * exc_rates[exc]
            )

            # Each non-zero underflow bin counts as one excitation
            deposited_exc_eng_arr[exc] += np.sum(
                    (coll_exc_sec_elec_tf[exc].eng_underflow > 0)*1.0
            ) * exc_rates[exc] * exc_potentials[exc]

            # Rebin the data so that the spectra stored above have an abscissa
            # of eleceng again (instead of eleceng - phys.lya_eng for HI etc.)
            coll_exc_sec_elec_tf[exc].rebin(eleceng)

            # Put them in a dictionary
            coll_exc_sec_elec_specs[exc] = coll_exc_sec_elec_tf[exc].grid_vals

    sec_specs = {'exc': coll_exc_sec_elec_specs,
                 'ion': coll_ion_sec_elec_specs}

    #####################################
    # Inverse Compton
    #####################################

    T = phys.TCMB(rs)

    # Photon transfer function for single primary electron single scattering.
    # This is dN/(dE dt), dt = 1 s.
    # 2->2 process:
    #   -eleceng is incoming electron energy
    #   -we don't care about incoming photon, we averaged over blackbody
    #   -photeng is outgoing photon energy
    #   -What's the outgoing electron energy?  Dunno, because we could have had
    #    any CMB photon in the initial state !!!
    phot_ICS_tf = ics_spec(
        eleceng, photeng, T, thomson_tf=raw_thomson_tf,
        rel_tf=raw_rel_tf, T_ref=phys.TCMB(400)
    )

    # Energy loss transfer function for single primary electron
    # single scattering. This is dN/(dE dt), dt = 1 s.
    # - Similar to above, but now keep track of the final electron energy !!!
    # - In other words, E_in - E_fin = Delta is the amount of energy gained by
    #   the photons, regardless of their initial energy..
    engloss_ICS_tf = engloss_spec(
        eleceng, delta=photeng, T=T,
        thomson_tf=raw_engloss_tf, rel_tf=raw_rel_tf
    )

    # Downcasting speeds up np.dot
    phot_ICS_tf._grid_vals = phot_ICS_tf.grid_vals.astype('float64')
    engloss_ICS_tf._grid_vals = engloss_ICS_tf.grid_vals.astype('float64')

    # Switch the spectra type here to type 'N'.
    if phot_ICS_tf.spec_type == 'dNdE':
        phot_ICS_tf.switch_spec_type()
    if engloss_ICS_tf.spec_type == 'dNdE':
        engloss_ICS_tf.switch_spec_type()

    # Define some useful lengths.
    N_eleceng = eleceng.size
    N_photeng = photeng.size

    # Create the secondary electron transfer functions.

    # (empty) ICS transfer function.
    elec_ICS_tf = tf.TransFuncAtRedshift(
        np.zeros((N_eleceng, N_eleceng)), in_eng=eleceng,
        rs=rs*np.ones_like(eleceng), eng=eleceng,
        dlnz=-1, spec_type='N'
    )

    if ics_engloss_data is not None:
        elec_ICS_tf._grid_vals = ics_engloss_data.rebin(
            engloss_ICS_tf.grid_vals
        )
    else:
        # A specialized function: takes the energy loss spectrum and knows how
        # to rebin into eleceng. Turns out that it is basically a delta
        # function in the original in_eng bin (only 1s worth of scattering
        # occurred) and the rest goes usually into the lower neighbor's bin !!!
        elec_ICS_tf._grid_vals = spectools.engloss_rebin_fast(
            eleceng, photeng, engloss_ICS_tf.grid_vals, eleceng
        )

    # dE_ICS_dt = 4/3*phys.thomson_xsec*phys.c * beta_ele**2/(1-beta_ele**2) *
    # phys.CMB_eng_density(phys.TCMB(rs))
    dE_ICS_dt = engloss_ICS_tf.toteng()
    ICS_engloss_arr = dE_ICS_dt
    if simple_ICS:
        elec_ICS_tf._grid_vals[0, 0] = -dE_ICS_dt[0]/eleceng[0]
        elec_ICS_tf._grid_vals[1:, 1:] = np.diag(
            dE_ICS_dt[1:]/(eleceng[:-1] - eleceng[1:])
        )
        elec_ICS_tf._grid_vals[1:, :-1] = -np.diag(
            dE_ICS_dt[1:]/(eleceng[:-1] - eleceng[1:])
        )

    # Total upscattered photon energy.
    cont_loss_ICS_vec = np.zeros_like(eleceng)
    # Enforces energy conservation.
    deposited_ICS_vec = np.zeros_like(eleceng)

    #########################
    # Collisional Excitation and Ionization
    #########################

    elec_tf = {'exc': {}, 'ion': {}}

    for process in ['exc', 'ion']:
        for i, species in enumerate(atoms):

            # If we're not looking at Hydrogen excitation, then don't worry
            # about different excited states
            if not ((process == 'exc') & (species == 'HI')):
                # Collisional excitation or ionization rates.
                rate_vec = (
                    ns[species] *
                    coll_xsec[process](eleceng, species, method) *
                    beta_ele * phys.c
                )

                # Normalized electron spectrum after excitation or ionization.
                elec_tf[process][species] = tf.TransFuncAtRedshift(
                    rate_vec[:, np.newaxis]*sec_specs[process][species],
                    in_eng=eleceng, rs=rs*np.ones_like(eleceng),
                    eng=eleceng, dlnz=-1, spec_type='N'
                )

            # If we're considering Hydrogen excitation, keep track of all l=p states to 10p, and also 2s
            else:
                for exc in exc_types[:-2]:
                    rate_vec = ns[species] * coll_xsec[process](
                            eleceng, species=species, method=method, state=exc
                    ) * beta_ele * phys.c
                    # if exc == '2p': #!!!FIX
                    #    rate_vec = ns[species] * coll_xsec[process](
                    #            eleceng, species='HI', method='old'
                    #    ) * beta_ele * phys.c/4

                    elec_tf[process][exc] = tf.TransFuncAtRedshift(
                        rate_vec[:, np.newaxis]*sec_specs[process][exc],
                        in_eng=eleceng, rs=rs*np.ones_like(eleceng),
                        eng=eleceng, dlnz=-1, spec_type='N'
                    )
                    #if (exc[0] == '2'):
                        #print(exc, rate_vec[45]*10.2, coll_xsec[process](eleceng, species=species, method=method, state=exc)[45])
                        #print(elec_tf[process][exc].totN()[45]*10.2)
                    #if (exc[0] == '3'):
                        #print(exc, rate_vec[45]*13.6*8/9, coll_xsec[process](eleceng, species=species, method=method, state=exc)[45])
                        #print(elec_tf[process][exc].totN()[45]*13.6*8/9)

    deposited_exc_vec = {exc: np.zeros_like(eleceng) for exc in exc_types}

    #############################################
    # Heating
    #############################################

    dE_heat_dt = phys.elec_heating_engloss_rate(eleceng, xe, rs, method=method,
                                                Te=phys.TCMB(rs))
    deposited_heat_vec = np.zeros_like(eleceng)

    MEDEA_heat = False
    if MEDEA_heat:
        rate_vec = dE_heat_dt/eleceng*20
        # rate_vec=np.ones_like(eleceng)
        # Electron with energy eleceng produces a spectrum with one particle
        # of energy eleceng*.95
        heat_sec_elec_tf = tf.TransFuncAtRedshift(
            id_mat,
            in_eng=eleceng, rs=-1*np.ones_like(eleceng),
            eng=eleceng*.95,
            dlnz=-1, spec_type='N'
        )

        #low_sec_elec_tf += tf.TransFuncAtRedshift(
        #    id_mat,
        #    in_eng = eleceng, rs = -1*np.ones_like(eleceng),
        #    eng = eleceng*.05,
        #    dlnz = -1, spec_type = 'N'
        #)

        # Rebin the data so that the spectra stored above now have an abscissa
        # of eleceng again (instead of eleceng*.95)
        heat_sec_elec_tf.rebin(eleceng)
        elec_heat_spec_grid = rate_vec[:, np.newaxis]*heat_sec_elec_tf._grid_vals
        #elec_heat_spec_grid = heat_sec_elec_tf._grid_vals
    else:

        # new_eleceng = eleceng - dE_heat_dt

        # if not np.all(new_eleceng[1:] > eleceng[:-1]):
        #     utils.compare_arr([new_eleceng, eleceng])
        #     raise ValueError('heating loss is too large: smaller time step required.')

        # # After the check above, we can define the spectra by
        # # manually assigning slightly less than 1 particle along
        # # diagonal, and a small amount in the bin below.

        # # N_n-1 E_n-1 + N_n E_n = E_n - dE_dt
        # # N_n-1 + N_n = 1
        # # therefore, (1 - N_n) E_n-1 - (1 - N_n) E_n = - dE_dt
        # # i.e. N_n = 1 - dE_dt/(E_n - E_n-1)

        elec_heat_spec_grid = np.identity(eleceng.size)
        elec_heat_spec_grid[0, 0] -= dE_heat_dt[0]/eleceng[0]
        elec_heat_spec_grid[1:, 1:] += np.diag(
            dE_heat_dt[1:]/(eleceng[:-1] - eleceng[1:])
        )
        elec_heat_spec_grid[1:, :-1] -= np.diag(
            dE_heat_dt[1:]/(eleceng[:-1] - eleceng[1:])
        )
    #print(elec_heat_spec_grid)


    #############################################
    # Initialization of secondary spectra 
    #############################################

    # !!! Delete
    ## Low and high energy boundaries
    #eleceng_high = eleceng[eleceng >= loweng]
    #eleceng_high_ind = np.arange(eleceng.size)[eleceng >= loweng]
    #eleceng_low = eleceng[eleceng < loweng]
    #eleceng_low_ind  = np.arange(eleceng.size)[eleceng < loweng]


    #if eleceng_low.size == 0:
    #    raise TypeError('Energy abscissa must contain a low energy bin below 3 keV.')

    # Empty containers for quantities.
    # Final secondary photon spectrum.
    sec_phot_tf = tf.TransFuncAtRedshift(
        np.zeros((N_eleceng, N_photeng)), in_eng = eleceng,
        rs = rs*np.ones_like(eleceng), eng=photeng,
        dlnz=-1, spec_type='N'
    )

    #!!! Delete
    ## Final secondary low energy electron spectrum.
    #sec_lowengelec_tf = tf.TransFuncAtRedshift(
    #    np.zeros((N_eleceng, N_eleceng)), in_eng = eleceng,
    #    rs = rs*np.ones_like(eleceng), eng=eleceng,
    #    dlnz=-1, spec_type='N'
    #)

    # Continuum energy loss rate per electron, dU_CMB/dt.
    CMB_upscatter_eng_rate = phys.thomson_xsec*phys.c*phys.CMB_eng_density(phys.TCMB(rs))
    CMB_upscatter_rate = phys.thomson_xsec*phys.c*phys.CMB_N_density(phys.TCMB(rs))
    
    ##!!! take the prompt photons - thomson*c*normalized blackbody, compare to dE_ICS_dt

    # Normalized CMB spectrum.
    norm_CMB_spec = Spectrum(photeng, phys.CMB_spec(photeng, phys.TCMB(rs)), spec_type='dNdE')
    norm_CMB_spec /= norm_CMB_spec.totN()

    # Get the CMB spectrum upscattered from cont_loss_ICS_vec.
    upscattered_CMB_grid = np.outer(CMB_upscatter_rate*np.ones_like(eleceng), norm_CMB_spec.N)

    # Secondary scattered electron spectrum.
    sec_elec_spec_N_arr = (
        elec_ICS_tf.grid_vals
        + np.sum([elec_tf['exc'][exc].grid_vals     for exc     in exc_types], axis=0)
        + np.sum([elec_tf['ion'][species].grid_vals for species in atoms], axis=0)
        + elec_heat_spec_grid
    )
    #print(elec_ICS_tf.grid_vals)
    #print(dE_ICS_dt[1:]/(eleceng[:-1] - eleceng[1:]))

    # Secondary photon spectrum (from ICS).
    sec_phot_spec_N_arr = phot_ICS_tf.grid_vals  - upscattered_CMB_grid
    sec_phot_spec_N_arr[beta_ele < .1] = loweng_ICS_distortion(
        eleceng[beta_ele < .1], photeng, phys.TCMB(rs))
    #crap = sec_phot_spec_N_arr.copy()

    # Deposited ICS array.
    ICS_err_arr = (
        # Total amount of energy of the electrons that got scattered
        np.sum(elec_ICS_tf.grid_vals, axis=1)*eleceng

        # Total amount of energy in the secondary electron spectrum
        - np.dot(elec_ICS_tf.grid_vals, eleceng)
        # The difference is the amount of energy that the electron lost
        # through scattering, and that should be equal to the energy gained
        # by photons

        # This is -[the energy gained by photons]
        # (Total amount of energy in the upscattered photons -
        # the energy these photons started with)
        - np.dot(sec_phot_spec_N_arr, photeng)
    )
    # This is only non-zero due to numerical errors. It is very small.

    # Energy loss is not taken into account for eleceng > 20*phys.me
    ICS_err_arr[eleceng > 20*phys.me - phys.me] -= (
        CMB_upscatter_eng_rate
    )
    # !!! A legacy of Tracy's code: For electrons with a boost of 20 or more
    # We pretend the initial CMB photon had zero energy.
    # To do this, get rid of CMB_upscatter_eng.

    # Continuum energy loss array.
    continuum_engloss_arr = CMB_upscatter_eng_rate*np.ones_like(eleceng)
    # Energy loss is not taken into account for eleceng > 20*phys.me
    continuum_engloss_arr[eleceng > 20*phys.me - phys.me] = 0
 
    # Deposited excitation array.
    for exc in exc_types:
        deposited_exc_eng_arr[exc] += exc_potentials[exc]*elec_tf['exc'][exc].totN()

    # Deposited H ionization array.
    deposited_H_ion_eng_arr = ion_potentials['HI']*elec_tf['ion']['HI'].totN()/2
    
    # Deposited He ionization array.
    deposited_He_ion_eng_arr = np.sum([
        ion_potentials[species]*elec_tf['ion'][species].totN()/2 
    for species in ['HeI', 'HeII']], axis=0)

    # Deposited heating array.
    deposited_heat_eng_arr += dE_heat_dt
    
    no_self_scatter = True
    if no_self_scatter:
        # Remove self-scattering, re-normalize. 
        np.fill_diagonal(sec_elec_spec_N_arr, 0)
        
        toteng_no_self_scatter_arr = (
            np.dot(sec_elec_spec_N_arr, eleceng)
            #+ np.dot(sec_phot_spec_N_arr, photeng)
            #- continuum_engloss_arr
            #+ ICS_engloss_arr #!!! ICS modification
            + ICS_err_arr
            + np.sum([deposited_exc_eng_arr[exc] for exc in exc_types], axis=0)
            + deposited_H_ion_eng_arr
            + deposited_He_ion_eng_arr
            + deposited_heat_eng_arr
        )

        if simple_ICS:
            toteng_no_self_scatter_arr += ICS_engloss_arr
        else:
            toteng_no_self_scatter_arr += np.dot(sec_phot_spec_N_arr, photeng) 

    tind = 0
    #print(
    #    deposited_exc_eng_arr['2s'][tind],
    #    deposited_exc_eng_arr['2p'][tind],
    #    deposited_exc_eng_arr['3p'][tind],
    #    deposited_H_ion_eng_arr[tind],
    #    deposited_He_ion_eng_arr[tind],
    #    deposited_heat_eng_arr[tind],
    #    np.dot(sec_elec_spec_N_arr, eleceng)[tind],
    #    np.dot(sec_phot_spec_N_arr, photeng)[tind]-continuum_engloss_arr[tind] + deposited_ICS_eng_arr[tind]
    #    #toteng_no_self_scatter_arr[tind]
    #)

    if no_self_scatter:
        fac_arr = eleceng/toteng_no_self_scatter_arr
        
        sec_elec_spec_N_arr *= fac_arr[:, np.newaxis]
        #sec_phot_spec_N_arr *= fac_arr[:, np.newaxis]
        #continuum_engloss_arr  *= fac_arr
        if simple_ICS:
            ICS_engloss_arr        *= fac_arr #!!! ICS modification
        else:
            sec_phot_spec_N_arr *= fac_arr[:, np.newaxis]
            continuum_engloss_arr  *= fac_arr
        ICS_err_arr  *= fac_arr
        for exc in exc_types:
            deposited_exc_eng_arr[exc]  *= fac_arr
        deposited_H_ion_eng_arr  *= fac_arr
        deposited_He_ion_eng_arr  *= fac_arr
        deposited_heat_eng_arr *= fac_arr
    
    #!!! Delete
    # Zero out deposition/ICS processes below loweng. 
    #Change loweng to 10.2eV!!!  Then assign everything below 10.2 eV to heat.  Then get rid of sec_lowengelec_spec
    #mask = eleceng < loweng
    #print(deposited_ICS_eng_arr[mask],
    #        deposited_H_ion_eng_arr[mask],
    #        deposited_He_ion_eng_arr[mask], '\n',
    #        deposited_heat_eng_arr[mask])
    #deposited_ICS_eng_arr[eleceng < loweng]  = 0
    #for exc in exc_types:
    #    deposited_exc_eng_arr[exc][eleceng < loweng]  = 0
    #deposited_H_ion_eng_arr[eleceng < loweng]  = 0
    #deposited_He_ion_eng_arr[eleceng < loweng]  = 0
    ##deposited_heat_eng_arr[eleceng < loweng] = eleceng[eleceng<loweng]
    
    #continuum_engloss_arr[eleceng < loweng]  = 0
    
    #sec_phot_spec_N_arr[eleceng < loweng] = 0
    
    #!!! Delete
    ## Scattered low energy and high energy electrons. 
    ## Needed for final low energy electron spectra.
    ##!!! Try getting rid of this
    #sec_lowengelec_N_arr = np.identity(eleceng.size)
    #sec_lowengelec_N_arr[eleceng >= loweng] = 0
    #sec_lowengelec_N_arr[eleceng_high_ind[0]:, :eleceng_high_ind[0]] += sec_elec_spec_N_arr[eleceng_high_ind[0]:, :eleceng_high_ind[0]]

    #sec_highengelec_N_arr = np.zeros_like(sec_elec_spec_N_arr)
    #sec_highengelec_N_arr[:, eleceng_high_ind[0]:] = (
    #    sec_elec_spec_N_arr[:, eleceng_high_ind[0]:]
    #)
    
    # T = N.T + Prompt
    #tind=45
    #print((id_mat - sec_elec_spec_N_arr)[tind][:tind+1])
    #print(sec_elec_spec_N_arr[tind][:tind+1])
    #print(np.sum(sec_elec_spec_N_arr, axis=1))
    #print(deposited_heat_eng_arr)
    #sec_elec_spec_N_arr[sec_elec_spec_N_arr<1e-8] = 0.
    inv_mat = id_mat - sec_elec_spec_N_arr

    ICS_err_vec = solve_triangular(
        inv_mat,
        ICS_err_arr, lower=True, check_finite=False, unit_diagonal=True
    )

    for exc in exc_types:
        deposited_exc_vec[exc] = solve_triangular(
            inv_mat,
            deposited_exc_eng_arr[exc], lower=True,
            check_finite=False, unit_diagonal=True
        )

    deposited_H_ion_vec = solve_triangular(
        inv_mat,
        deposited_H_ion_eng_arr, lower=True,
        check_finite=False, unit_diagonal=True
    )

    deposited_He_ion_vec = solve_triangular(
        inv_mat,
        deposited_He_ion_eng_arr, lower=True,
        check_finite=False, unit_diagonal=True
    )

    deposited_heat_vec = solve_triangular(
        inv_mat,
        deposited_heat_eng_arr, lower=True,
        check_finite=False, unit_diagonal=True
    )

    cont_loss_ICS_vec = solve_triangular(
        inv_mat,
        continuum_engloss_arr, lower=True,
        check_finite=False, unit_diagonal=True
    )

    sec_phot_specs = solve_triangular(
        inv_mat,
        sec_phot_spec_N_arr, lower=True,
        check_finite=False, unit_diagonal=True
    )

    if simple_ICS:
        deposited_ICS_vec = solve_triangular(
            inv_mat,
            ICS_engloss_arr, lower=True,
            check_finite=False, unit_diagonal=True
        )

    # Subtract continuum from sec_phot_specs. After this point,
    # sec_phot_specs will contain the *distortions* to the CMB,

    # Normalized CMB spectrum.
    norm_CMB_spec = Spectrum(
        photeng, phys.CMB_spec(photeng, phys.TCMB(rs)), spec_type='dNdE'
    )
    norm_CMB_spec /= norm_CMB_spec.toteng()

    # Get the CMB spectrum upscattered from cont_loss_ICS_vec.
    upscattered_CMB_grid = np.outer(cont_loss_ICS_vec, norm_CMB_spec.N)

    # Subtract this spectrum from sec_phot_specs to get the final
    # transfer function.

    sec_phot_tf._grid_vals = sec_phot_specs
    if not simple_ICS:
        deposited_ICS_vec = np.dot(sec_phot_tf._grid_vals, photeng)

    #!!! Delete
    #sec_lowengelec_tf._grid_vals = sec_lowengelec_specs

    # Conservation checks.
    failed_conservation_check = False

    if check_conservation_eng:

        conservation_check = (
            eleceng
            #- np.dot(sec_lowengelec_tf.grid_vals, eleceng)
            # + cont_loss_ICS_vec
            #- np.dot(sec_phot_tf.grid_vals, photeng)
            #- ICS_engloss_vec #!!! ICS modification
            - np.sum([deposited_exc_vec[exc] for exc in exc_types], axis=0)
            - deposited_He_ion_vec
            - deposited_H_ion_vec
            - deposited_heat_vec
            - deposited_ICS_vec
            #- ICS_err_vec
        )

        if np.any(np.abs(conservation_check/eleceng) > 1e-2):
            failed_conservation_check = True

        if verbose or failed_conservation_check:

            for i, eng in enumerate(eleceng):

                print('***************************************************')
                print('rs: ', rs)
                print('injected energy: ', eng)

                #!!! Delete
                #print(
                #    'Fraction of Energy in low energy electrons: ',
                #    np.dot(sec_lowengelec_tf.grid_vals[i], eleceng)/eng
                #)

                # print('Energy in photons: ',
                #     np.dot(sec_phot_tf.grid_vals[i], photeng)
                # )
                # print('Continuum_engloss: ', cont_loss_ICS_vec[i])

                if simple_ICS:
                    print(
                        'Fraction of Energy in photons - Continuum: ', (
                            deposited_ICS_vec[i] #ICS modification
                        )
                    )
                else:
                    print(
                        'Fraction of Energy in photons - Continuum: ', (
                            np.dot(sec_phot_tf.grid_vals[i], photeng)/eng
                             - cont_loss_ICS_vec[i]
                        )
                    )

                print(
                    'Fraction Deposited in ionization: ', 
                    (deposited_H_ion_vec[i] + deposited_He_ion_vec[i])/eng
                )
                for exc in exc_types:
                    print(
                        'Fraction Deposited in excitation '+exc+': ', 
                        deposited_exc_vec[exc][i]/eng
                    )

                print(
                    'Fraction Deposited in heating: ', 
                    deposited_heat_vec[i]/eng
                )

                print(
                    'Energy is conserved up to (%): ',
                    conservation_check[i]/eng*100
                )
                print('Fraction Deposited in ICS (Numerical Error): ', 
                    ICS_err_vec[i]/eng
                )
                
                print(
                    'Energy conservation with deposited (%): ',
                    (conservation_check[i] - ICS_err_vec[i])/eng*100
                )
                print('***************************************************')
                
            if failed_conservation_check:
                raise RuntimeError('Conservation of energy failed.')

    return (
        sec_phot_tf,
        {'H' : deposited_H_ion_vec, 'He' : deposited_He_ion_vec}, deposited_exc_vec, deposited_heat_vec,
        deposited_ICS_vec, ICS_err_vec
    )

def get_elec_cooling_tfTMP(
    eleceng, photeng, rs, xHII, xHeII=0, 
    raw_thomson_tf=None, raw_rel_tf=None, raw_engloss_tf=None,
    coll_ion_sec_elec_specs=None, coll_exc_sec_elec_specs=None,
    ics_engloss_data=None, 
    check_conservation_eng = False, verbose=False
):

    """Transfer functions for complete electron cooling through inverse Compton scattering (ICS) and atomic processes.
    Copied from the original version of DarkHistory. 

    Parameters
    ----------
    eleceng : ndarray, shape (m, )
        The electron kinetic energy abscissa.
    photeng : ndarray
        The photon energy abscissa.
    rs : float
        The redshift (1+z). 
    xHII : float
        Ionized hydrogen fraction, nHII/nH. 
    xHeII : float, optional
        Singly-ionized helium fraction, nHe+/nH. Default is 0. 
    raw_thomson_tf : TransFuncAtRedshift, optional
        Thomson ICS scattered photon spectrum transfer function. If None, uses the default transfer function. Default is None.
    raw_rel_tf : TransFuncAtRedshift, optional
        Relativistic ICS scattered photon spectrum transfer function. If None, uses the default transfer function. Default is None.
    raw_engloss_tf : TransFuncAtRedshift, optional
        Thomson ICS scattered electron net energy loss transfer function. If None, uses the default transfer function. Default is None.
    coll_ion_sec_elec_specs : tuple of 3 ndarrays, shapes (m, m), optional 
        Normalized collisional ionization secondary electron spectra, order HI, HeI, HeII, indexed by injected electron energy by outgoing electron energy. If None, the function calculates this. Default is None.
    coll_exc_sec_elec_specs : tuple of 3 ndarray, shapes (m, m), optional 
        Normalized collisional excitation secondary electron spectra, order HI, HeI, HeII, indexed by injected electron energy by outgoing electron energy. If None, the function calculates this. Default is None.
    ics_engloss_data : EnglossRebinData
        An `EnglossRebinData` object which stores rebinning information (based on ``eleceng`` and ``photeng``) for speed. Default is None.
    check_conservation_eng : bool
        If True, lower=True, checks for energy conservation. Default is False.
    verbose : bool
        If True, prints energy conservation checks. Default is False.
    
    Returns
    -------

    tuple
        Transfer functions for electron cooling deposition and spectra.

    See Also
    ---------
    :class:`.TransFuncAtRedshift`
    :class:`.EnglossRebinData`
    :mod:`.ics`

    Notes
    -----
    
    The entries of the output tuple are (see Sec IIIC of the paper):

    0. The secondary propagating photon transfer function :math:`\\overline{\\mathsf{T}}_\\gamma`; 
    1. The low-energy electron transfer function :math:`\\overline{\\mathsf{T}}_e`; 
    2. Energy deposited into ionization :math:`\\overline{\\mathbf{R}}_\\text{ion}`; 
    3. Energy deposited into excitation :math:`\\overline{\\mathbf{R}}_\\text{exc}`; 
    4. Energy deposited into heating :math:`\\overline{\\mathbf{R}}_\\text{heat}`;
    5. Upscattered CMB photon total energy :math:`\\overline{\\mathbf{R}}_\\text{CMB}`, and
    6. Numerical error away from energy conservation.

    Items 2--5 are vectors that when dotted into an electron spectrum with abscissa ``eleceng``, return the energy deposited/CMB energy upscattered for that spectrum. 

    Items 0--1 are :class:`.TransFuncAtRedshift` objects. For each of these objects ``tf`` and a given electron spectrum ``elec_spec``, ``tf.sum_specs(elec_spec)`` returns the propagating photon/low-energy electron spectrum after cooling.

    The default version of the three ICS transfer functions that are required by this function is provided in :mod:`.tf_data`.

    """

    # Use default ICS transfer functions if not specified.

    #ics_tf = load_data('ics_tf')

    #raw_thomson_tf = ics_tf['thomson']
    #raw_rel_tf     = ics_tf['rel']
    #raw_engloss_tf = ics_tf['engloss']

    if coll_ion_sec_elec_specs is None:

        # Compute the (normalized) collisional ionization spectra.
        coll_ion_sec_elec_specs = (
            phys.coll_ion_sec_elec_spec(eleceng, eleceng, species='HI'),
            phys.coll_ion_sec_elec_spec(eleceng, eleceng, species='HeI'),
            phys.coll_ion_sec_elec_spec(eleceng, eleceng, species='HeII')
        )

    if coll_exc_sec_elec_specs is None:

        # Compute the (normalized) collisional excitation spectra.
        id_mat = np.identity(eleceng.size)

        # Electron with energy eleceng produces a spectrum with one particle
        # of energy eleceng - phys.lya.eng. Similar for helium. 
        coll_exc_sec_elec_tf_HI = tf.TransFuncAtRedshift(
            np.squeeze(id_mat[:, np.where(eleceng > phys.lya_eng)]),
            in_eng = eleceng, rs = -1*np.ones_like(eleceng),
            eng = eleceng[eleceng > phys.lya_eng] - phys.lya_eng,
            dlnz = -1, spec_type = 'N'
        )

        coll_exc_sec_elec_tf_HeI = tf.TransFuncAtRedshift(
            np.squeeze(
                id_mat[:, np.where(eleceng > phys.He_exc_eng['23s'])]
            ),
            in_eng = eleceng, rs = -1*np.ones_like(eleceng),
            eng = (
                eleceng[eleceng > phys.He_exc_eng['23s']] 
                - phys.He_exc_eng['23s']
            ), 
            dlnz = -1, spec_type = 'N'
        )

        coll_exc_sec_elec_tf_HeII = tf.TransFuncAtRedshift(
            np.squeeze(id_mat[:, np.where(eleceng > 4*phys.lya_eng)]),
            in_eng = eleceng, rs = -1*np.ones_like(eleceng),
            eng = eleceng[eleceng > 4*phys.lya_eng] - 4*phys.lya_eng,
            dlnz = -1, spec_type = 'N'
        )

        # Rebin the data so that the spectra stored above now have an abscissa
        # of eleceng again (instead of eleceng - phys.lya_eng for HI etc.)
        coll_exc_sec_elec_tf_HI.rebin(eleceng)
        coll_exc_sec_elec_tf_HeI.rebin(eleceng)
        coll_exc_sec_elec_tf_HeII.rebin(eleceng)

        # Put them in a tuple.
        coll_exc_sec_elec_specs = (
            coll_exc_sec_elec_tf_HI.grid_vals,
            coll_exc_sec_elec_tf_HeI.grid_vals,
            coll_exc_sec_elec_tf_HeII.grid_vals
        )

    # Set the electron fraction. 
    xe = xHII + xHeII
    
    # v/c of electrons
    beta_ele = np.sqrt(1 - 1/(1 + eleceng/phys.me)**2)

    #####################################
    # Inverse Compton
    #####################################

    T = phys.TCMB(rs)

    # Photon transfer function for single primary electron single scattering.
    # This is dN/(dE dt), dt = 1 s.
    phot_ICS_tf = ics_spec(
        eleceng, photeng, T, thomson_tf = raw_thomson_tf, rel_tf = raw_rel_tf
    )

    # Downcasting speeds up np.dot
    phot_ICS_tf._grid_vals = phot_ICS_tf.grid_vals.astype('float64')

    # Energy loss transfer function for single primary electron
    # single scattering. This is dN/(dE dt), dt = 1 s.
    engloss_ICS_tf = engloss_spec(
        eleceng, photeng, T, thomson_tf = raw_engloss_tf, rel_tf = raw_rel_tf
    )

    # Downcasting speeds up np.dot
    engloss_ICS_tf._grid_vals = engloss_ICS_tf.grid_vals.astype('float64')

    # Switch the spectra type here to type 'N'.
    if phot_ICS_tf.spec_type == 'dNdE':
        phot_ICS_tf.switch_spec_type()
    if engloss_ICS_tf.spec_type == 'dNdE':
        engloss_ICS_tf.switch_spec_type()


    # Define some useful lengths.
    N_eleceng = eleceng.size
    N_photeng = photeng.size

    # Create the secondary electron transfer functions.

    # ICS transfer function.
    elec_ICS_tf = tf.TransFuncAtRedshift(
        np.zeros((N_eleceng, N_eleceng)), in_eng = eleceng,
        rs = rs*np.ones_like(eleceng), eng = eleceng,
        dlnz = -1, spec_type = 'N'
    )

    if ics_engloss_data is not None:
        elec_ICS_tf._grid_vals = ics_engloss_data.rebin(
            engloss_ICS_tf.grid_vals
        )
    else:
        elec_ICS_tf._grid_vals = spectools.engloss_rebin_fast(
            eleceng, photeng, engloss_ICS_tf.grid_vals, eleceng
        )
    
    # Total upscattered photon energy.
    cont_loss_ICS_vec = np.zeros_like(eleceng)
    # Deposited energy, enforces energy conservation.
    deposited_ICS_vec = np.zeros_like(eleceng)
    
    #########################
    # Collisional Excitation  
    #########################


    # Collisional excitation rates.
    rate_vec_exc_HI = (
        (1 - xHII)*phys.nH*rs**3 * phys.coll_exc_xsec(eleceng, species='HI') * beta_ele * phys.c
    )
    
    rate_vec_exc_HeI = (
        (phys.nHe/phys.nH - xHeII)*phys.nH*rs**3 * phys.coll_exc_xsec(eleceng, species='HeI') * beta_ele * phys.c
    )
    
    rate_vec_exc_HeII = (
        xHeII*phys.nH*rs**3 * phys.coll_exc_xsec(eleceng, species='HeII') * beta_ele * phys.c
    )

    # Normalized electron spectrum after excitation.
    elec_exc_HI_tf = tf.TransFuncAtRedshift(
        rate_vec_exc_HI[:, np.newaxis]*coll_exc_sec_elec_specs[0],
        in_eng = eleceng, rs = rs*np.ones_like(eleceng),
        eng = eleceng, dlnz = -1, spec_type  = 'N'
    )

    elec_exc_HeI_tf = tf.TransFuncAtRedshift(
        rate_vec_exc_HeI[:, np.newaxis]*coll_exc_sec_elec_specs[1],
        in_eng = eleceng, rs = rs*np.ones_like(eleceng),
        eng = eleceng, dlnz = -1, spec_type  = 'N'
    )

    elec_exc_HeII_tf = tf.TransFuncAtRedshift(
        rate_vec_exc_HeII[:, np.newaxis]*coll_exc_sec_elec_specs[2],
        in_eng = eleceng, rs = rs*np.ones_like(eleceng),
        eng = eleceng, dlnz = -1, spec_type  = 'N'
    )
   
    # Deposited energy for excitation.
    deposited_exc_vec = np.zeros_like(eleceng)


    #########################
    # Collisional Ionization  
    #########################

    # Collisional ionization rates.
    rate_vec_ion_HI = (
        (1 - xHII)*phys.nH*rs**3 
        * phys.coll_ion_xsec(eleceng, species='HI') * beta_ele * phys.c
    )
    
    rate_vec_ion_HeI = (
        (phys.nHe/phys.nH - xHeII)*phys.nH*rs**3 
        * phys.coll_ion_xsec(eleceng, species='HeI') * beta_ele * phys.c
    )
    
    rate_vec_ion_HeII = (
        xHeII*phys.nH*rs**3
        * phys.coll_ion_xsec(eleceng, species='HeII') * beta_ele * phys.c
    )

    # Normalized secondary electron spectra after ionization.
    elec_spec_ion_HI   = (
        rate_vec_ion_HI[:,np.newaxis]   * coll_ion_sec_elec_specs[0]
    )
    elec_spec_ion_HeI  = (
        rate_vec_ion_HeI[:,np.newaxis]  * coll_ion_sec_elec_specs[1]
    )
    elec_spec_ion_HeII = (
        rate_vec_ion_HeII[:,np.newaxis] * coll_ion_sec_elec_specs[2]
    )

    # Construct TransFuncAtRedshift objects.
    elec_ion_HI_tf = tf.TransFuncAtRedshift(
        elec_spec_ion_HI, in_eng = eleceng, rs = rs*np.ones_like(eleceng), 
        eng = eleceng, dlnz = -1, spec_type = 'N'
    )
    elec_ion_HeI_tf = tf.TransFuncAtRedshift(
        elec_spec_ion_HeI, in_eng = eleceng, rs = rs*np.ones_like(eleceng), 
        eng = eleceng, dlnz = -1, spec_type = 'N'
    )
    elec_ion_HeII_tf = tf.TransFuncAtRedshift(
        elec_spec_ion_HeII, in_eng = eleceng, rs = rs*np.ones_like(eleceng), 
        eng = eleceng, dlnz = -1, spec_type = 'N'
    )

    # Deposited energy for ionization.
    deposited_ion_vec = np.zeros_like(eleceng)

     #############################################
    # Heating
    #############################################
    
    dE_heat_dt = phys.elec_heating_engloss_rate(eleceng, xe, rs)
    
    deposited_heat_vec = np.zeros_like(eleceng)

    # new_eleceng = eleceng - dE_heat_dt

    # if not np.all(new_eleceng[1:] > eleceng[:-1]):
    #     utils.compare_arr([new_eleceng, eleceng])
    #     raise ValueError('heating loss is too large: smaller time step required.')

    # # After the check above, we can define the spectra by
    # # manually assigning slightly less than 1 particle along
    # # diagonal, and a small amount in the bin below. 

    # # N_n-1 E_n-1 + N_n E_n = E_n - dE_dt
    # # N_n-1 + N_n = 1
    # # therefore, (1 - N_n) E_n-1 - (1 - N_n) E_n = - dE_dt
    # # i.e. N_n = 1 + dE_dt/(E_n-1 - E_n)

    elec_heat_spec_grid = np.identity(eleceng.size)
    elec_heat_spec_grid[0,0] -= dE_heat_dt[0]/eleceng[0]
    elec_heat_spec_grid[1:, 1:] += np.diag(
        dE_heat_dt[1:]/(eleceng[:-1] - eleceng[1:])
    )
    elec_heat_spec_grid[1:, :-1] -= np.diag(
        dE_heat_dt[1:]/(eleceng[:-1] - eleceng[1:])
    )


    #############################################
    # Initialization of secondary spectra 
    #############################################

    # Low and high energy boundaries
    loweng = 3000
    eleceng_high = eleceng[eleceng > loweng]
    eleceng_high_ind = np.arange(eleceng.size)[eleceng > loweng]
    eleceng_low = eleceng[eleceng <= loweng]
    eleceng_low_ind  = np.arange(eleceng.size)[eleceng <= loweng]


    if eleceng_low.size == 0:
        raise TypeError('Energy abscissa must contain a low energy bin below 3 keV.')

    # Empty containers for quantities.
    # Final secondary photon spectrum.
    sec_phot_tf = tf.TransFuncAtRedshift(
        np.zeros((N_eleceng, N_photeng)), in_eng = eleceng,
        rs = rs*np.ones_like(eleceng), eng = photeng,
        dlnz = -1, spec_type = 'N'
    )
    # Final secondary low energy electron spectrum.
    sec_lowengelec_tf = tf.TransFuncAtRedshift(
        np.zeros((N_eleceng, N_eleceng)), in_eng = eleceng,
        rs = rs*np.ones_like(eleceng), eng = eleceng,
        dlnz = -1, spec_type = 'N'
    )

    # Continuum energy loss rate per electron, dU_CMB/dt.
    CMB_upscatter_eng_rate = phys.thomson_xsec*phys.c*phys.CMB_eng_density(T)
    
    # Secondary scattered electron spectrum.
    sec_elec_spec_N_arr = (
        elec_ICS_tf.grid_vals
        + elec_exc_HI_tf.grid_vals 
        + elec_exc_HeI_tf.grid_vals 
        + elec_exc_HeII_tf.grid_vals
        + elec_ion_HI_tf.grid_vals 
        + elec_ion_HeI_tf.grid_vals 
        + elec_ion_HeII_tf.grid_vals
        + elec_heat_spec_grid
    )
    
    # Secondary photon spectrum (from ICS). 
    sec_phot_spec_N_arr = phot_ICS_tf.grid_vals
    
    # Deposited ICS array.
    deposited_ICS_eng_arr = (
        np.sum(elec_ICS_tf.grid_vals, axis=1)*eleceng
        - np.dot(elec_ICS_tf.grid_vals, eleceng)
        - (np.dot(sec_phot_spec_N_arr, photeng) - CMB_upscatter_eng_rate)
    )

    # Energy loss is not taken into account for eleceng > 20*phys.me
    deposited_ICS_eng_arr[eleceng > 20*phys.me - phys.me] -= ( 
        CMB_upscatter_eng_rate
    )

    # Continuum energy loss array.
    continuum_engloss_arr = CMB_upscatter_eng_rate*np.ones_like(eleceng)
    # Energy loss is not taken into account for eleceng > 20*phys.me
    continuum_engloss_arr[eleceng > 20*phys.me - phys.me] = 0
    
    # Deposited excitation array.
    deposited_exc_eng_arr = (
        phys.lya_eng*np.sum(elec_exc_HI_tf.grid_vals, axis=1)
        + phys.He_exc_eng['23s']*np.sum(elec_exc_HeI_tf.grid_vals, axis=1)
        + 4*phys.lya_eng*np.sum(elec_exc_HeII_tf.grid_vals, axis=1)
    )
    
    # Deposited ionization array.
    deposited_ion_eng_arr = (
        phys.rydberg*np.sum(elec_ion_HI_tf.grid_vals, axis=1)/2
        + phys.He_ion_eng*np.sum(elec_ion_HeI_tf.grid_vals, axis=1)/2
        + 4*phys.rydberg*np.sum(elec_ion_HeII_tf.grid_vals, axis=1)/2
    )

    # Deposited heating array.
    deposited_heat_eng_arr = dE_heat_dt
    
    # Remove self-scattering, re-normalize. 
    np.fill_diagonal(sec_elec_spec_N_arr, 0)
    
    toteng_no_self_scatter_arr = (
        np.dot(sec_elec_spec_N_arr, eleceng)
        + np.dot(sec_phot_spec_N_arr, photeng)
        - continuum_engloss_arr
        + deposited_ICS_eng_arr
        + deposited_exc_eng_arr
        + deposited_ion_eng_arr
        + deposited_heat_eng_arr
    )

    fac_arr = eleceng/toteng_no_self_scatter_arr
    
    sec_elec_spec_N_arr *= fac_arr[:, np.newaxis]
    sec_phot_spec_N_arr *= fac_arr[:, np.newaxis]
    continuum_engloss_arr  *= fac_arr
    deposited_ICS_eng_arr  *= fac_arr
    deposited_exc_eng_arr  *= fac_arr
    deposited_ion_eng_arr  *= fac_arr
    deposited_heat_eng_arr *= fac_arr
    
    # Zero out deposition/ICS processes below loweng. 
    deposited_ICS_eng_arr[eleceng < loweng]  = 0
    deposited_exc_eng_arr[eleceng < loweng]  = 0
    deposited_ion_eng_arr[eleceng < loweng]  = 0
    deposited_heat_eng_arr[eleceng < loweng] = 0
    
    continuum_engloss_arr[eleceng < loweng]  = 0
    
    sec_phot_spec_N_arr[eleceng < loweng] = 0
    
    # Scattered low energy and high energy electrons. 
    # Needed for final low energy electron spectra.
    sec_lowengelec_N_arr = np.identity(eleceng.size)
    sec_lowengelec_N_arr[eleceng >= loweng] = 0
    sec_lowengelec_N_arr[eleceng_high_ind[0]:, :eleceng_high_ind[0]] += sec_elec_spec_N_arr[eleceng_high_ind[0]:, :eleceng_high_ind[0]]

    sec_highengelec_N_arr = np.zeros_like(sec_elec_spec_N_arr)
    sec_highengelec_N_arr[:, eleceng_high_ind[0]:] = (
        sec_elec_spec_N_arr[:, eleceng_high_ind[0]:]
    )
    
    # T = E.T + Prompt
    deposited_ICS_vec  = solve_triangular(
        np.identity(eleceng.size) - sec_elec_spec_N_arr,
        deposited_ICS_eng_arr, lower=True, check_finite=False
    )
    deposited_exc_vec  = solve_triangular(
        np.identity(eleceng.size) - sec_elec_spec_N_arr, 
        deposited_exc_eng_arr, lower=True, check_finite=False
    )
    deposited_ion_vec  = solve_triangular(
        np.identity(eleceng.size) - sec_elec_spec_N_arr, 
        deposited_ion_eng_arr, lower=True, check_finite=False
    )
    deposited_heat_vec = solve_triangular(
        np.identity(eleceng.size) - sec_elec_spec_N_arr, 
        deposited_heat_eng_arr, lower=True, check_finite=False
    )
    
    cont_loss_ICS_vec = solve_triangular(
        np.identity(eleceng.size) - sec_elec_spec_N_arr, 
        continuum_engloss_arr, lower=True, check_finite=False
    )
    
    sec_phot_specs = solve_triangular(
        np.identity(eleceng.size) - sec_elec_spec_N_arr, 
        sec_phot_spec_N_arr, lower=True, check_finite=False
    )
    
    # Prompt: low energy e produced in secondary spectrum upon scattering (sec_lowengelec_N_arr).
    # T : high energy e produced (sec_highengelec_N_arr). 
    sec_lowengelec_specs = solve_triangular(
        np.identity(eleceng.size) - sec_highengelec_N_arr,
        sec_lowengelec_N_arr, lower=True, check_finite=False
    )

    # Subtract continuum from sec_phot_specs. After this point, 
    # sec_phot_specs will contain the *distortions* to the CMB. 

    # Normalized CMB spectrum. 
    norm_CMB_spec = Spectrum(
        photeng, phys.CMB_spec(photeng, phys.TCMB(rs)), spec_type='dNdE'
    )
    norm_CMB_spec /= norm_CMB_spec.toteng()

    # Get the CMB spectrum upscattered from cont_loss_ICS_vec. 
    upscattered_CMB_grid = np.outer(cont_loss_ICS_vec, norm_CMB_spec.N)

    # Subtract this spectrum from sec_phot_specs to get the final
    # transfer function.

    sec_phot_tf._grid_vals = sec_phot_specs - upscattered_CMB_grid
    sec_lowengelec_tf._grid_vals = sec_lowengelec_specs

    # Conservation checks.
    failed_conservation_check = False

    if check_conservation_eng:

        conservation_check = (
            eleceng
            - np.dot(sec_lowengelec_tf.grid_vals, eleceng)
            # + cont_loss_ICS_vec
            - np.dot(sec_phot_tf.grid_vals, photeng)
            - deposited_exc_vec
            - deposited_ion_vec
            - deposited_heat_vec
        )

        if np.any(np.abs(conservation_check/eleceng) > 0.1):
            failed_conservation_check = True

        if verbose or failed_conservation_check:

            for i,eng in enumerate(eleceng):

                print('***************************************************')
                print('rs: ', rs)
                print('injected energy: ', eng)

                print(
                    'Fraction of Energy in low energy electrons: ',
                    np.dot(sec_lowengelec_tf.grid_vals[i], eleceng)/eng
                )

                # print('Energy in photons: ', 
                #     np.dot(sec_phot_tf.grid_vals[i], photeng)
                # )
                
                # print('Continuum_engloss: ', cont_loss_ICS_vec[i])
                
                print(
                    'Fraction of Energy in photons - Continuum: ', (
                        np.dot(sec_phot_tf.grid_vals[i], photeng)/eng
                        # - cont_loss_ICS_vec[i]
                    )
                )

                print(
                    'Fraction Deposited in ionization: ', 
                    deposited_ion_vec[i]/eng
                )

                print(
                    'Fraction Deposited in excitation: ', 
                    deposited_exc_vec[i]/eng
                )

                print(
                    'Fraction Deposited in heating: ', 
                    deposited_heat_vec[i]/eng
                )

                print(
                    'Energy is conserved up to (%): ',
                    conservation_check[i]/eng*100
                )
                print('Fraction Deposited in ICS (Numerical Error): ', 
                    deposited_ICS_vec[i]/eng
                )
                
                print(
                    'Energy conservation with deposited (%): ',
                    (conservation_check[i] - deposited_ICS_vec[i])/eng*100
                )
                print('***************************************************')
                
            if failed_conservation_check:
                raise RuntimeError('Conservation of energy failed.')

    return (
        sec_phot_tf, sec_lowengelec_tf,
        deposited_ion_vec, deposited_exc_vec, deposited_heat_vec,
        cont_loss_ICS_vec, deposited_ICS_vec
    )


def loweng_ICS_distortion(eleceng, photeng, T):
    """ Spectrum of secondary photons from ICS by low energy electrons.

    Parameters
    ----------
    eleceng : float
        Electron energy.
    photeng : float
        Photon energy.
    T : float
        Photon temperature.

    Returns
    -------
    ndarray
        Grid values of photon spectrum.
    """
    y = photeng/T

    prefac = (
        phys.c*(3/8)*phys.thomson_xsec/4
        * (
            8*np.pi*T**2
            / (phys.ele_compton*phys.me)**3
        )
    )

    beta = phys.np.sqrt(1 - 1/(1 + eleceng/phys.me)**2)
    P_beta_2 = 32/9*y**3/(1 - np.exp(-y))**3*(
            np.exp(-2*y)*(y + 4) + np.exp(-y)*(y - 4)
        )

    spec = Spectra(
            prefac*np.outer(beta**2, P_beta_2), in_eng=eleceng, eng=photeng,
            spec_type='dNdE'
        )

    spec.switch_spec_type()
    return spec._grid_vals

