""" The main DarkHistory function.

"""
import numpy as np
import pickle
from numpy.linalg import matrix_power
from scipy.interpolate import interp1d, interp2d

# from config import data_path, photeng, eleceng
# from tf_data import *

from darkhistory.config import load_data


import darkhistory.physics as phys

from darkhistory.spec import pppc
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec.spectra import Spectra
import darkhistory.spec.transferfunction as tf

from darkhistory.spec.spectools import EnglossRebinData
from darkhistory.spec.spectools import discretize, get_bin_bound

from darkhistory.electrons import positronium as pos
from darkhistory.electrons.elec_cooling import get_elec_cooling_tf
from darkhistory.electrons.elec_cooling import get_elec_cooling_tfTMP
from darkhistory.low_energy.lowE_deposition import compute_fs as compute_fs_OLD
import darkhistory.low_energy.atomic as atomic
import darkhistory.low_energy.bound_free as bf

import darkhistory.photons.phot_dep as phot_dep
import darkhistory.low_energy.lowE_electrons as lowE_electrons

from darkhistory.history import tla


def evolve(
    in_spec_elec=None, in_spec_phot=None,
    rate_func_N=None, rate_func_eng=None,
    DM_process=None, mDM=None, sigmav=None,
    lifetime=None, primary=None,
    struct_boost=None,
    start_rs=None, high_rs=np.inf, end_rs=4,
    helium_TLA=False, reion_switch=False,
    reion_rs=None, reion_method='Puchwein',
    heat_switch=True, photoion_rate_func=None, photoheat_rate_func=None,
    xe_reion_func=None, DeltaT=None, alpha_bk=None,
    init_cond=None, coarsen_factor=1, backreaction=True,
    compute_fs_method='no_He', elec_method='new',
    distort=False, fudge=1.125, nmax=10, fexc_switch=True, MLA_funcs=None,
    cross_check=False, reprocess_distortion=True, simple_2s1s=False, iterations=1,
    first_iter=True, init_distort=None, prev_output=None,
    use_tqdm=True, tqdm_jupyter=True, mxstep=1000, rtol=1e-4,verbose=0
):
    """
    Main function computing histories and spectra.

    Parameters
    -----------
    in_spec_elec : :class:`.Spectrum`, optional
        Spectrum per injection event into electrons. *in_spec_elec.rs*
        of the :class:`.Spectrum` must be the initial redshift.
    in_spec_phot : :class:`.Spectrum`, optional
        Spectrum per injection event into photons. *in_spec_phot.rs*
        of the :class:`.Spectrum` must be the initial redshift.
    rate_func_N : function, optional
        Function returning number of injection events per volume per time, with
        redshift :math:`(1+z)` as an input.
    rate_func_eng : function, optional
        Function returning energy injected per volume per time, with redshift
        :math:`(1+z)` as an input.
    DM_process : {'swave', 'decay'}, optional
        Dark matter process to use.
    sigmav : float, optional
        Thermally averaged cross section for dark matter annihilation.
    lifetime : float, optional
        Decay lifetime for dark matter decay.
    primary : string, optional
        Primary channel of annihilation/decay. See :func:`.get_pppc_spec`
        for complete list. Use *'elec_delta'* or *'phot_delta'* for delta
        function injections of a pair of photons/an electron-positron pair.
    struct_boost : function, optional
        Energy injection boost factor due to structure formation.
    start_rs : float, optional
        Starting redshift :math:`(1+z)` to evolve from. Default is
        :math:`(1+z)` = 3000. Specify only for use with *DM_process*.
        Otherwise, initialize *in_spec_elec.rs* and/or
        *in_spec_phot.rs* directly.
    high_rs : float, optional
        Threshold redshift used to deal with stiff ODE.
        For rs > high_rs, solve for x_HII using xHII_std and Tm using Tm_std.
        For rs < high_rs, solve the differential equations numerically.
    end_rs : float, optional
        Final redshift :math:`(1+z)` to evolve to. Default is :math:`1+z = 4`.
    reion_switch : bool
        Reionization model included if *True*, default is *False*.
    helium_TLA : bool
        If *True*, the TLA is solved with helium. Default is *False*.
    reion_rs : float, optional
        Redshift :math:`(1+z)` at which reionization effects turn on.
    reion_method : string, optional
        Reionization model, options are {'Puchwein', 'early', 'middle', 'late'}.
    heat_switch : bool, optional
        If *True*, includes photoheating during reionization.
    photoion_rate_func : tuple of functions, optional
        Functions take redshift :math:`1+z` as input, return the
        photoionization rate in s :sup:`-1` of HI, HeI and HeII respectively.
        If not specified, defaults to :func:`.photoion_rate`.
    photoheat_rate_func : tuple of functions, optional
        Functions take redshift :math:`1+z` as input, return the photoheating
        rate in s\ :sup:`-1` of HI, HeI and HeII respectively.
        If not specified, defaults to :func:`.photoheat_rate`.
    xe_reion_func : function, optional
        Specifies a fixed ionization history after reion_rs.
    DeltaT : float, optional
        For fixed reionization models, constant of proportionality for photoheating. See arXiv:2008.01084.
    alpha_bk : float, optional
        Post-reionization heating power law. See arXiv:2008.01084.
    init_cond : tuple of floats
        Specifies the initial (xH, xHe, Tm). Defaults to :func:`.Tm_std`,
        :func:`.x_std` at the *start_rs*.
    coarsen_factor : int
        Coarsening to apply to the transfer function matrix. Default is 1.
    backreaction : bool
        If *False*, uses the baseline TLA solution to calculate :math:`f_c(z)`.
        Default is True.
    compute_fs_method : {'no_He', 'He_recomb', 'He', 'HeII'}
        Method for evaluating helium ionization.

        * *'no_He'* -- all ionization assigned to hydrogen;
        * *'He_recomb'* -- all photoionized helium atoms recombine; and
        * *'He'* -- all photoionized helium atoms do not recombine;
        * *'HeII'* -- all ionization assigned to HeII.

        Default is 'no_He'.
    elec_method : {'new', 'old', 'eff'}
        Method for evaluating electron energy deposition.

        * *'old'* -- Low-energy electrons separated out, resolved with MEDEA. Old ionization and excitation cross sections.
        * *'new'* -- No separation into low-energy electrons, new ionization and excitation cross sections, deexcitations calculated by probability of downscattering to 2s and 2p.
        * *'eff'* -- f_exc computed using distortions, new ionization and excitation cross sections, effective f_exc calculated.

        Default is 'new'.
    distort : bool, optional
        If *True* calculate the distortion. This sets elec_processes to True.
        If *False* speed up the code by skipping slow electron cooling code.
    cross_check : bool, optional
        If *True*, compare against 1604.02457 by using original MEDEA files,
        turning off partial binning, etc. Default is *False*.
    fudge : float, optional
        Value of Recfast fudge factor.
    nmax : int, optional
        If distort==True, sets the maximum H principal quantum number that the MLA tracks.
    fexc_switch : bool, optional
        If *True*, include the source term b_DM to the MLA steady-state
        equation, Mx = b
    MLA_funcs : list, optional
        A list of three interpolating functions for the MLA rates:
        (i) The recombination rate as a function of redshift, alpha_MLA
        (ii) The ionization rate, beta_MLA
        (iii) The MLA ionization rate correction due to energy inj., beta_DM
        For example, if `out` is the output of a main.evolve() run with
        distort set to True,
            [interp1d(out['MLA'][0], out['MLA'][i]), for i in range(1,4)]
        could be passed into MLA_funcs
    reprocess_distortion : bool, optional
        if *True*, set Delta_f != 0, accounting for distortion photons from
        earlier redshifts to be absorbed or stimulate emission, i.e. be
        reprocessed.
    simple_2s1s : bool, optional
        if *True*, fixes the decay rate to :math:`8.22` s:math:`^{-1}`. Default is *False*.
    iterations : int, optional
        Number of iterations to run for the MLA iterative method.
    first_iter : bool, optional
        If *True*, treat this as the first iteration. Default is *True*.
    init_distort : Spectrum object, optional
        The spectral distortion at start_rs. If None, initialized with zeros.
    prev_output : list of dict, optional
        Output from a previous iteration of this function.
    use_tqdm : bool, optional
        If *True*, uses `tqdm` to track progress. Default is *True*.
    tqdm_jupyter : bool, optional
        Uses `tqdm` in Jupyter notebooks if *True*. Otherwise, uses tqdm for terminals. Default is *True*.
    mxstep : int, optional
        The maximum number of steps allowed for each integration point.
        See *scipy.integrate.odeint()* for more information. Default is *1000*.
    rtol : float, optional
        The relative error of the solution. See *scipy.integrate.odeint()* for
        more information. Default is *1e-4*.

    Returns
    -------
    dict
        Result of the calculation, including ...


    Examples
    --------

    1. *Dark matter annihilation* -- dark matter mass of 50 GeV, annihilation
    cross section :math:`2 \\times 10^{-26}` cm\ :sup:`3` s\ :sup:`-1`,
    annihilating to :math:`b \\bar{b}`, solved without backreaction,
    a coarsening factor of 32 and the default structure formation boost: ::

        import darkhistory.physics as phys

        out = evolve(
            DM_process='swave', mDM=50e9, sigmav=2e-26,
            primary='b', start_rs=3000.,
            backreaction=False,
            struct_boost=phys.struct_boost_func()
        )

    2. *Dark matter decay* -- dark matter mass of 100 GeV, decay lifetime
    :math:`3 \\times 10^{25}` s, decaying to a pair of :math:`e^+e^-`,
    solved with backreaction, a coarsening factor of 16: ::

        out = evolve(
            DM_process='decay', mDM=1e8, lifetime=3e25,
            primary='elec_delta', start_rs=3000.,
            backreaction=True
        )

    See Also
    ---------
    :func:`.get_pppc_spec`

    :func:`.struct_boost_func`

    :func:`.photoion_rate`, :func:`.photoheat_rate`

    :func:`.Tm_std`, :func:`.xHII_std` and :func:`.xHeII_std`


    """

    #########################################################################
    #########################################################################
    # Input                                                                 #
    #########################################################################
    #########################################################################

    # Save the initial options for subsequent iterations.
    options = locals().copy()
    # options = dict(
    #     in_spec_elec=in_spec_elec, in_spec_phot=in_spec_phot,
    #     rate_func_N=rate_func_N, rate_func_eng=rate_func_eng,
    #     DM_process=DM_process, mDM=mDM, sigmav=sigmav,
    #     lifetime=lifetime, primary=primary,
    #     struct_boost=struct_boost,
    #     start_rs=start_rs, high_rs=high_rs, end_rs=end_rs,
    #     helium_TLA=helium_TLA, reion_switch=reion_switch,
    #     reion_rs=reion_rs, reion_method=reion_method,
    #     heat_switch=heat_switch, DeltaT=DeltaT, alpha_bk=alpha_bk,
    #     photoion_rate_func=photoion_rate_func, photoheat_rate_func=photoheat_rate_func, xe_reion_func=xe_reion_func,
    #     init_cond=init_cond, coarsen_factor=coarsen_factor, backreaction=backreaction,
    #     compute_fs_method=compute_fs_method, elec_method=elec_method, mxstep=mxstep, rtol=rtol,
    #     distort=distort, fudge=fudge, nmax=nmax, fexc_switch=fexc_switch, MLA_funcs=MLA_funcs,
    #     use_tqdm=use_tqdm, tqdm_jupyter=tqdm_jupyter, cross_check=cross_check,
    #     reprocess_distortion=reprocess_distortion, simple_2s1s=simple_2s1s,
    #     iterations=iterations, first_iter=first_iter, prev_output=prev_output
    # )


    #####################################
    # Initialization for DM_process     #
    #####################################

    # Load data.
    binning = load_data('binning',verbose=verbose)
    photeng = binning['phot']
    eleceng = binning['elec']

    dep_tf_data = load_data('dep_tf',verbose=verbose)

    highengphot_tf_interp = dep_tf_data['highengphot']
    lowengphot_tf_interp = dep_tf_data['lowengphot']
    lowengelec_tf_interp = dep_tf_data['lowengelec']
    highengdep_interp = dep_tf_data['highengdep']

    ics_tf_data = load_data('ics_tf',verbose=verbose)

    ics_thomson_ref_tf = ics_tf_data['thomson']
    ics_rel_ref_tf = ics_tf_data['rel']
    engloss_ref_tf = ics_tf_data['engloss']

    # If compute_fs_method is 'HeII', must be using instantaneous reionization.
    if compute_fs_method == 'HeII':

        print('Using instantaneous reionization at 1+z = ', reion_rs)

        def xe_func(rs):
            rs = np.squeeze(np.array([rs]))
            xHII = phys.x_std(rs)
            xHII[rs < 7] = 1
            return xHII

        xe_reion_func = xe_func


    # Handle the case where a DM process is specified.
    if DM_process == 'swave' or DM_process == 'pwave':
        if sigmav is None or start_rs is None:
            raise ValueError(
                'sigmav and start_rs must be specified.'
            )
        # Get input spectra from PPPC.
        in_spec_elec = pppc.get_pppc_spec(mDM, eleceng, primary, 'elec')
        in_spec_phot = pppc.get_pppc_spec(mDM, photeng, primary, 'phot')

        # Initialize the input spectrum redshift.
        in_spec_elec.rs = start_rs
        in_spec_phot.rs = start_rs

        # Convert to type 'N'.
        in_spec_elec.switch_spec_type('N')
        in_spec_phot.switch_spec_type('N')

        # If struct_boost is none, just set to 1.
        if struct_boost is None:
            def struct_boost(rs):
                return 1.

        # Define the rate functions.
        def rate_func_N(rs):
            return (
                phys.inj_rate(DM_process, rs, mDM=mDM, sigmav=sigmav)
                * struct_boost(rs) / (2*mDM)
            )

        def rate_func_eng(rs):
            return (
                phys.inj_rate(DM_process, rs, mDM=mDM, sigmav=sigmav)
                * struct_boost(rs)
            )

    if DM_process == 'decay':
        if lifetime is None or start_rs is None:
            raise ValueError(
                'lifetime and start_rs must be specified.'
            )

        # The decay rate is insensitive to structure formation
        def struct_boost(rs):
            return 1

        # Get spectra from PPPC.
        in_spec_elec = pppc.get_pppc_spec(
            mDM, eleceng, primary, 'elec', decay=True
        )
        in_spec_phot = pppc.get_pppc_spec(
            mDM, photeng, primary, 'phot', decay=True
        )

        # Initialize the input spectrum redshift.
        in_spec_elec.rs = start_rs
        in_spec_phot.rs = start_rs

        # Convert to type 'N'.
        in_spec_elec.switch_spec_type('N')
        in_spec_phot.switch_spec_type('N')

        # Define the rate functions.
        def rate_func_N(rs):
            return (
                phys.inj_rate('decay', rs, mDM=mDM, lifetime=lifetime) / mDM
            )

        def rate_func_eng(rs):
            return phys.inj_rate('decay', rs, mDM=mDM, lifetime=lifetime)

    #####################################
    # Input Checks                      #
    #####################################

    if (
        not np.array_equal(in_spec_elec.eng, eleceng)
        or not np.array_equal(in_spec_phot.eng, photeng)
    ):
        raise ValueError('in_spec_elec and in_spec_phot must use config.photeng\
                         and config.eleceng respectively as abscissa.')

    if (
        np.any(highengphot_tf_interp.dlnz != lowengphot_tf_interp.dlnz)
        or np.any(highengphot_tf_interp.dlnz != lowengelec_tf_interp.dlnz)
        or np.any(lowengphot_tf_interp.dlnz != lowengelec_tf_interp.dlnz)
    ):
        raise ValueError('TransferFuncInterp objects must all have the same \
                         dlnz.')

    if in_spec_elec.rs != in_spec_phot.rs:
        raise ValueError('Input spectra must have the same rs.')


    if (reion_method is not None) and (xe_reion_func is not None):
        raise ValueError('Either specify reionization model or xe reionization curve, not both')

    if cross_check:
        print('cross_check has been set to True -- No longer using all MEDEA \
              files and no longer using partial-binning.')

    if DeltaT is not None and xe_reion_func is None:
        raise ValueError('DeltaT is only for fixed reionization histories using xe_reion_func.')

    if alpha_bk is not None and xe_reion_func is None:
        raise ValueError('alpha_bk is only for fixed reionization histories using xe_reion_func.')

    if xe_reion_func is not None and (DeltaT is None or alpha_bk is None):
        raise ValueError('Photoheating model needed for fixed reionization histories using xe_reion_func.')

    if elec_method == 'eff' and not distort:
        raise ValueError('Can only use effective f_exc if distortions are calculated.')

    if elec_method == 'eff' and not fexc_switch:
        raise ValueError('Can only use effective f_exc if excitations are passed to the distortion code.')
    if iterations > 1 and first_iter and not distort:

        raise ValueError('No reason to iterate more than once if not using distortions.')

    #####################################
    # Initialization                    #
    #####################################

    # Initialize start_rs for arbitrary injection.
    start_rs = in_spec_elec.rs

    # Initialize the initial x and Tm.
    if init_cond is None:

        # Default to baseline
        xH_init = phys.x_std(start_rs)
        xHe_init = phys.x_std(start_rs, 'HeII')

        #xHeIII_init = phys.xHeII_std(start_rs)
        Tm_init = phys.Tm_std(start_rs)

    else:

        # User-specified.
        xH_init = init_cond[0]
        xHe_init = init_cond[1]
        Tm_init = init_cond[2]

    # Initialize redshift/timestep related quantities.

    # Default step in the transfer function. Note highengphot_tf_interp.dlnz
    # contains 3 different regimes, and we start with the first.
    dlnz = highengphot_tf_interp.dlnz[-1]

    # The current redshift.
    rs = start_rs

    # The timestep between evaluations of transfer functions, including
    # coarsening.
    dt = dlnz * coarsen_factor / phys.hubble(rs)

    # tqdm set-up.
    if use_tqdm:
        if tqdm_jupyter:
            from tqdm import tqdm_notebook as tqdm

        else:
            from tqdm import tqdm

        pbar = tqdm(
            total=np.ceil((np.log(rs) - np.log(end_rs))/dlnz/coarsen_factor)
        )

    # Normalization to convert from per injection event to
    # per baryon per dlnz step.
    def norm_fac(rs):
        return rate_func_N(rs) * (
            dlnz * coarsen_factor / phys.hubble(rs) / (phys.nB * rs**3)
        )

    # If there are no injected electrons, we get a speed-up by ignoring them.
    # If we are calculating distortions, then we need to use one of the new methods,
    # which treats low-energy electrons in the same way as high-energy electrons.
    if (in_spec_elec.totN() > 0) or distort or elec_method != 'old':
        elec_processes = True

        # The excitation states we keep track of in hydrogen
        # We keep track of specific states for hydrogen, but not for
        # HeI and HeII !!!
        # method = 'new'  # or 'MEDEA' or 'AcharyaKhatri' or 'new'

        if elec_method == 'AcharyaKhatri':
            H_states = ['2s', '2p', '3p']
            nmax = 3
        elif elec_method == 'MEDEA':
            H_states = ['2s', '2p', '3p', '4p', '5p',
                        '6p', '7p', '8p', '9p', '10p']
            nmax = 10
        elif elec_method == 'new' or elec_method == 'eff':
            H_states = ['2s', '2p',
                        '3s', '3p', '3d',
                        '4s', '4p', '4d', '4f',
                        '5p', '6p', '7p', '8p', '9p', '10p']


    else:
        elec_processes = False

    if elec_processes:

        #####################################
        # High-Energy Electrons             #
        #####################################

        # Get the data necessary to compute the electron cooling results.
        # coll_ion_sec_elec_specs is \bar{N} for collisional ionization,
        # and coll_exc_sec_elec_specs \bar{N} for collisional excitation.
        # Heating and others are evaluated in get_elec_cooling_tf
        # itself.

        # Contains information that makes converting an energy loss spectrum
        # to a scattered electron spectrum fast.
        if elec_method != 'old':
            (
                coll_ion_sec_elec_specs, coll_exc_sec_elec_specs,
                ics_engloss_data
            ) = get_elec_cooling_data(eleceng, photeng, H_states)
        else:
            (
                coll_ion_sec_elec_specs, coll_exc_sec_elec_specs,
                ics_engloss_data
            ) = get_elec_cooling_dataTMP(eleceng, photeng)


    #########################################################################
    #########################################################################
    # Pre-Loop Preliminaries                                                #
    #########################################################################
    #########################################################################

    # Initialize the arrays that will contain x and Tm results.
    x_arr = np.array([[xH_init, xHe_init]])
    Tm_arr = np.array([Tm_init])

    # Initialize Spectra objects to contain all of the output spectra.

    out_highengphot_specs = Spectra([], spec_type='N')
    out_lowengphot_specs = Spectra([], spec_type='N')
    out_lowengelec_specs = Spectra([], spec_type='N')
    out_distort_specs = Spectra([], spec_type='N')

    # Define these methods for speed.
    append_highengphot_spec = out_highengphot_specs.append
    append_lowengphot_spec = out_lowengphot_specs.append
    append_lowengelec_spec = out_lowengelec_specs.append
    append_distort_spec = out_distort_specs.append

    # Initialize arrays to store f values.
    f_c = np.empty((0, 7))

    if distort:

        # Initialize Spectrum object that stores the distortion

        # The spectrum requires a lower bound lower than lowengphot, and finer binning
        hplanck = phys.hbar * 2*np.pi
        dist_eng = np.exp(np.linspace(np.log(hplanck*1e8),
                                      np.log(phys.rydberg), 2000))
        #dist_eng = np.sort(np.append(dist_eng,
        #                             atomic.get_transition_energies(nmax)))

        # If initial distortion is not given, initialize with zeros
        if init_distort is None:
            distortion = Spectrum(
                dist_eng, np.zeros_like(dist_eng), rs=1, spec_type='N'
            )
        # Otherwise, use given initial distortion
        else:
            distortion = init_distort

        # for masking out n-1 line photons and E>rydberg photons
        dist_mask = np.ones_like(dist_eng)

        bnds = get_bin_bound(dist_eng)
        E_1n = phys.rydberg * (1 - 1/np.arange(2, nmax)**2)
        for E in E_1n:
            ind = (sum(bnds <= E)-1)
            dist_mask[ind] = 0

        dist_mask *= dist_eng < phys.rydberg  # keep E<13.6eV photons

        elec_processes = True

        # rs, alpha, beta, beta_DM
        rs_in = rs*np.exp(dlnz)
        Tm_in = phys.Tm_std(rs_in)
        peebC = phys.peebles_C(phys.x_std(rs_in), rs_in)
        MLA_data = [
            [rs_in],
            [peebC * phys.alpha_recomb(Tm_in, 'HI')],
            [peebC * 4 * np.exp(-phys.lya_eng/Tm_in) * phys.beta_ion(
                Tm_in, 'HI', fudge)],
            [0.]
        ]
        # Stores population of atomic states
        x_full_data = []

        # Radial Matrix elements
        R = atomic.populate_radial(nmax)
        Thetas = bf.populate_thetas(nmax)

    else:
        distortion = None
        # Object to help us interpolate over MEDEA results.

    if elec_method == 'old':
        MEDEA_interp = lowE_electrons.make_interpolator(
            interp_type='2D', cross_check=cross_check
        )



    #########################################################################
    #########################################################################
    # LOOP! LOOP! LOOP! LOOP!                                               #
    #########################################################################
    #########################################################################

    while rs > end_rs:
        # Update tqdm.
        if use_tqdm:
            pbar.update(1)

        #############################
        # First Step Special Cases  #
        #############################
        if rs == start_rs:
            # Initialize the electron and photon arrays.
            # These will carry the spectra produced by applying the
            # transfer function at rs to high-energy photons.
            highengphot_spec_at_rs = in_spec_phot*0
            lowengphot_spec_at_rs = in_spec_phot*0
            lowengelec_spec_at_rs = in_spec_elec*0
            highengdep_at_rs = np.zeros(4)

        #####################################################################
        #####################################################################
        # Electron Cooling                                                  #
        #####################################################################
        #####################################################################

        if xe_reion_func is not None:
            # If reionization is complete, set the residual fraction
            # of neutral atoms to their measured value (arxiv:1503.08228)
            if x_arr[-1, 0] == 1:
                x_arr[-1, 0] = 1-10**(-4.4)
            if x_arr[-1, 1] == phys.chi:
                x_arr[-1, 1] = phys.chi*(1 - 10**(-4.4))
        else:
            # If the universe becomes fully ionized, offset slightly
            # to prevent numerical issues.
            if x_arr[-1, 0] == 1:
                x_arr[-1, 0] = 1-10**(-12.)
            if x_arr[-1, 1] == phys.chi:
                x_arr[-1, 1] = phys.chi*(1 - 10**(-12.))

        # Ionized Fractions
        x_at_rs = np.array([1. - x_arr[-1, 0],
                            phys.chi - x_arr[-1, 1],
                            x_arr[-1, 1]])

        # Electrons in this step, which are comprised of those:
        #  promptly injected from DM (in_spec_elec),
        #  produced by high energy photon processes (lowengelec_spec_at_rs)
        #  photoionized from atoms (ionized_elec)

        # All normalized per baryon
        ionized_elec = phot_dep.get_ionized_elec(lowengphot_spec_at_rs,
                                                 eleceng, x_at_rs, method=compute_fs_method)
        tot_spec_elec = (
            in_spec_elec*norm_fac(rs)+lowengelec_spec_at_rs+ionized_elec
        )
        tot_spec_elec.rs = rs

        # Get the transfer functions corresponding to electron cooling.
        # These are \bar{T}_\gamma, \bar{T}_e and \bar{R}_c.
        if elec_processes:

            if (
                backreaction
                or (compute_fs_method == 'HeII' and rs <= reion_rs)
            ):
                xHII_elec_cooling = x_arr[-1, 0]
                xHeII_elec_cooling = x_arr[-1, 1]
            else:
                xHII_elec_cooling = phys.x_std(rs)
                xHeII_elec_cooling = phys.x_std(rs, 'HeII')

            # Create the electron transfer functions
            if elec_method != 'old':
                (
                    ics_sec_phot_tf,
                    deposited_ion_arr, deposited_exc_arr, deposited_heat_arr,
                    ICS_engloss_vec, ICS_err_vec,
                ) = get_elec_cooling_tf(
                        eleceng, photeng, rs,
                        xHII_elec_cooling, xHeII=xHeII_elec_cooling,
                        raw_thomson_tf=ics_thomson_ref_tf,
                        raw_rel_tf=ics_rel_ref_tf,
                        raw_engloss_tf=engloss_ref_tf,
                        coll_ion_sec_elec_specs=coll_ion_sec_elec_specs,
                        coll_exc_sec_elec_specs=coll_exc_sec_elec_specs,
                        ics_engloss_data=ics_engloss_data,
                        method=elec_method, H_states=H_states
                        # loweng=eleceng[0]
                    )

                # Apply the transfer functions to the input electron
                # spectrum generated in this step

                # deposited energy into ionization, *per baryon in this step*.
                deposited_H_ion = np.dot(
                    deposited_ion_arr['H'], tot_spec_elec.N
                )
                deposited_He_ion = np.dot(
                    deposited_ion_arr['He'], tot_spec_elec.N
                )
                # Lyman-alpha excitation, from any excited state cascading through 2p
                # Probabilities that nl state cascades to 2p state
                # Ps = {'2p': 1.0000, '2s': 0.0, '3p': 0.0,
                #       '4p': 0.2609, '5p': 0.3078, '6p': 0.3259,
                #       '7p': 0.3353, '8p': 0.3410, '9p': 0.3448, '10p': 0.3476}
                Ps = {
                    '2s': 0.0000, '2p': 1.0000, '3s': 1.0000, '3p': 0.0000,
                    '3d': 1.0000, '4s': 0.5841, '4p': 0.2609, '4d': 0.7456,
                    '4f': 1.0000, '5p': 0.3078, '6p': 0.3259, '7p': 0.3353,
                    '8p': 0.3410, '9p': 0.3448, '10p': 0.3476
                }
                deposited_Lya_arr = np.sum([
                    deposited_exc_arr[species] * Ps[species] * phys.lya_eng/phys.H_exc_eng(species) for species in Ps
                ], axis=0)

                deposited_Lya = np.dot(
                    deposited_Lya_arr, tot_spec_elec.N
                )
                # heating
                deposited_heat = np.dot(
                    deposited_heat_arr, tot_spec_elec.N
                )
                # continuum photons, from deexcitation other than 2p->1s
                # Don't include ICS contribution; that gets counted through secondary photons/lowengphot
                deposited_cont_arr = np.sum([
                    deposited_exc_arr[species] * ((1. - Ps[species]) + Ps[species] * (1. - phys.lya_eng/phys.H_exc_eng(species)))for species in Ps
                ], axis=0)
                deposited_cont = np.dot(
                    deposited_cont_arr, tot_spec_elec.N
                )
                # numerical error
                deposited_err = np.dot(
                    ICS_err_vec, tot_spec_elec.N
                )

            else:
                (
                    ics_sec_phot_tf, elec_processes_lowengelec_tf,
                    deposited_ion_arr, deposited_exc_arr, deposited_heat_arr,
                    continuum_loss, deposited_ICS_arr
                ) = get_elec_cooling_tfTMP(
                        eleceng, photeng, rs,
                        xHII_elec_cooling, xHeII=xHeII_elec_cooling,
                        raw_thomson_tf=ics_thomson_ref_tf,
                        raw_rel_tf=ics_rel_ref_tf,
                        raw_engloss_tf=engloss_ref_tf,
                        coll_ion_sec_elec_specs=coll_ion_sec_elec_specs,
                        coll_exc_sec_elec_specs=coll_exc_sec_elec_specs,
                        ics_engloss_data=ics_engloss_data
                    )

                # Low energy electrons from electron cooling,
                # per injection event.
                elec_processes_lowengelec_spec = (
                    elec_processes_lowengelec_tf.sum_specs(in_spec_elec)
                )

                # Add this to lowengelec_at_rs.
                lowengelec_spec_at_rs += (
                    elec_processes_lowengelec_spec*norm_fac(rs)
                )

                # High-energy deposition into ionization,
                # *per baryon in this step*.
                deposited_H_ion = np.dot(
                    deposited_ion_arr,  in_spec_elec.N*norm_fac(rs)
                )
                # High-energy deposition into excitation,
                # *per baryon in this step*.
                deposited_Lya = np.dot(
                    deposited_exc_arr,  in_spec_elec.N*norm_fac(rs)
                )
                # High-energy deposition into heating,
                # *per baryon in this step*.
                deposited_heat = np.dot(
                    deposited_heat_arr, in_spec_elec.N*norm_fac(rs)
                )
                # High-energy deposition into continuum,
                # *per baryon in this step*.
                deposited_cont = np.dot(
                    continuum_loss,  in_spec_elec.N*norm_fac(rs)
                )
                # High-energy deposition numerical error,
                # *per baryon in this step*.
                deposited_err = np.dot(
                    deposited_ICS_arr,  in_spec_elec.N*norm_fac(rs)
                )

            # !!! Need to deal with distort == False case
            if distort:

                # Phase space density for the distortion
                if reprocess_distortion:
                    prefac = phys.nB * (phys.hbar*phys.c*rs)**3 * np.pi**2
                    # Make sure to mask out E_1n resonant photons, otherwise
                    # they will be absorbed twice !!! (implement better)
                    Delta_f = interp1d(
                        dist_eng, prefac*distortion.dNdE/dist_eng**2*dist_mask,
                        bounds_error=False, fill_value=(0, 0), kind='nearest'
                    )

                else:
                    def Delta_f(ee):
                        return 0

                x_1s = 1-x_arr[-1, 0]

                # resonant photons are absorbed when passed through the
                # following function - keep a copy of the unperturbed spectrum
                # in_distortion = distortion.copy()
                if rs == start_rs and init_distort is not None:
                    streaming_lowengphot = init_distort
                    streaming_lowengphot.redshift(rs) # Redshift spectrum from 1+z=0 to rs of loop
                else:
                    streaming_lowengphot = lowengphot_spec_at_rs.copy()

                # Usually taking a smooth spectrum from coarse -> fine binning, so use discretize()
                # Then ensure that redshift/spec_type is correct
                dNdE_interp = interp1d(streaming_lowengphot.eng, streaming_lowengphot.dNdE, bounds_error=False, fill_value=(0,0))
                streaming_lowengphot = discretize(dist_eng, dNdE_interp)
                streaming_lowengphot.rs = rs
                streaming_lowengphot.switch_spec_type('N')

                # Absorb excitation photons and electron collision energy
                if fexc_switch:
                    # Note: photons from streaming_lowengphot get absorbed here
                    delta_b = atomic.f_exc_to_b_numerator(
                        deposited_exc_arr,
                        tot_spec_elec, streaming_lowengphot,
                        H_states, dt, rate_func_eng,
                        nmax, x_1s
                    )

                else:
                    delta_b = {}

                # If true, fixes xHI to be the standard value.
                MLA_cross_check = False

                MLA_step, atomic_dist_spec, x_full = atomic.process_MLA(
                    rs, dt, x_1s, Tm_arr[-1], nmax, dist_eng, R, Thetas,
                    Delta_f, MLA_cross_check,
                    include_BF=True, simple_2s1s=simple_2s1s,
                    #fexc_switch, deposited_exc_arr,
                    #tot_spec_elec, distortion,
                    #H_states, rate_func_eng,
                    delta_b=delta_b, stimulated_emission=True
                )

                # Find the effective contribution dxe/dz from 1) distortions
                # affecting the recombination/photoionization and 2) DM excitations.

                xHI_at_rs, xHeI_at_rs, xHeII_at_rs = (1. - x_arr[-1, 0], phys.chi - x_arr[-1, 1], x_arr[-1, 1])
                xHII_at_rs = 1. - xHI_at_rs
                T_m = Tm_arr[-1]

                peebC = phys.peebles_C(xHII_at_rs, rs, fudge)
                beta_ion = phys.beta_ion(phys.TCMB(rs), 'HI', fudge)
                alpha = phys.alpha_recomb(T_m, 'HI', fudge)

                dxe_dt_std = -peebC * (
                    alpha * xHII_at_rs **2 * phys.nH * rs**3
                    - 4 * beta_ion * xHI_at_rs * np.exp(-phys.lya_eng/phys.TCMB(rs))
                )

                alpha_MLA_at_rs = MLA_step[0]
                beta_MLA_at_rs  = MLA_step[1]
                beta_DM_at_rs   = MLA_step[2]

                dxe_dt_MLA = (
                    - alpha_MLA_at_rs * xHII_at_rs**2 * phys.nH * rs **3
                    + beta_MLA_at_rs * xHI_at_rs
                    + beta_DM_at_rs
                )

                dxe_dt_exc = dxe_dt_MLA - dxe_dt_std


                MLA_data[0].append(rs)

                for i in np.arange(3):
                    MLA_data[i+1].append(MLA_step[i])

                x_full_data.append(x_full)

                # # Subtract off absorbed photons
                # atomic_dist_spec.N -= in_distortion.N - distortion.N

            #######################################
            # Photons from Injected Electrons     #
            #######################################

            # ICS secondary photon spectrum after electron cooling, per baryon
            ics_phot_spec = ics_sec_phot_tf.sum_specs(tot_spec_elec)

            # Get the spectrum from positron annihilation, per injection event.
            # Only half of in_spec_elec is positrons!
            positronium_phot_spec = pos.weighted_photon_spec(photeng) * (
                in_spec_elec.totN()/2
            )

            positronium_phot_spec.switch_spec_type('N')

        # Add injected photons + photons from injected electrons + photons
        # from atomic de-excitations
        # to the photon spectrum that got propagated forward.
        if elec_processes:
            highengphot_spec_at_rs += (
                in_spec_phot + positronium_phot_spec
            ) * norm_fac(rs) + ics_phot_spec
        else:
            highengphot_spec_at_rs += in_spec_phot * norm_fac(rs)

        # Compute the fraction of ionizing photons
        # that free stream within this step
        if (reion_switch is True) & (rs < start_rs):
            lowEprop_mask = phot_dep.propagating_lowE_photons_fracs(
                lowengphot_spec_at_rs, x_at_rs, dt)

        else:
            lowEprop_mask = np.zeros_like(lowengphot_spec_at_rs.eng)

        # Add this fraction to the propagating photons
        highengphot_spec_at_rs += lowEprop_mask * lowengphot_spec_at_rs

        # Get rid of the lowenergy photons that weren't absorbed through
        # photoionization -- they're in highengphot now
        lowengphot_spec_at_rs = (1-lowEprop_mask) * lowengphot_spec_at_rs

        # Set the redshift correctly.
        highengphot_spec_at_rs.rs = rs
        lowengphot_spec_at_rs.rs = rs

        #####################################################################
        #####################################################################
        # Save the Spectra!                                                 #
        #####################################################################
        #####################################################################

        # At this point, highengphot_at_rs, lowengphot_at_rs and
        # lowengelec_at_rs have been computed for this redshift.
        append_highengphot_spec(highengphot_spec_at_rs)
        append_lowengphot_spec(lowengphot_spec_at_rs)
        append_lowengelec_spec(lowengelec_spec_at_rs)

        #####################################################################
        #####################################################################
        # Compute f_c(z)                                                    #
        #####################################################################
        #####################################################################

        # Values of (xHI, xHeI, xHeII) to use for computing f.
        if backreaction or (compute_fs_method == 'HeII' and rs <= reion_rs):
            # Use the previous values with backreaction, or if we are using
            # the HeII method after the reionization redshift.
            x_vec_for_f = np.array(
                [1. - x_arr[-1, 0], phys.chi - x_arr[-1, 1], x_arr[-1, 1]]
            )
        else:
            # Use baseline values if no backreaction.
            x_vec_for_f = np.array([
                    1. - phys.x_std(rs),
                    phys.chi - phys.x_std(rs, 'HeII'),
                    phys.x_std(rs, 'HeII')
            ])


        # if not elec_processes:
        #     f_elec = {chan: 0 for chan in [
        #         'H ion', 'He ion', 'Lya', 'heat', 'cont', 'err']}

        # else:
        # High-energy deposition from input electrons.

        if not elec_processes:

            deposited_H_ion = 0.
            deposited_Lya   = 0.
            deposited_heat  = 0.
            deposited_cont  = 0.
            deposited_err   = 0.

        highengdep_at_rs += np.array([
            deposited_H_ion/dt,
            # deposited_He_ion/dt,
            deposited_Lya/dt,
            deposited_heat/dt,
            deposited_cont/dt
        ])

        if elec_method == 'new':

            # High-energy deposition from input electrons
            norm = phys.nB*rs**3 / rate_func_eng(rs)
            f_elec = {
                'H ion': highengdep_at_rs[0] * norm,
                'He ion': deposited_He_ion/dt * norm,
                'Lya': highengdep_at_rs[1] * norm,
                'heat': highengdep_at_rs[2] * norm,
                'cont': highengdep_at_rs[3] * norm,
                'err': deposited_err/dt * norm
            }

        elif elec_method == 'eff':

            # High-energy deposition from input electrons,
            # but Lya is calculated
            # from the impact of distortions on xe_dot.
            norm = phys.nB*rs**3 / rate_func_eng(rs)
            f_elec = {
                'H ion': highengdep_at_rs[0] * norm,
                'He ion': deposited_He_ion/dt * norm,
                'Lya': highengdep_at_rs[1] * norm,
                'exc': dxe_dt_exc * phys.lya_eng * phys.nH * rs**3 / rate_func_eng(rs),
                'heat': highengdep_at_rs[2] * norm,
                'cont': highengdep_at_rs[3] * norm,
                'err': deposited_err/dt * norm
            }


        else:
            input_spec = lowengelec_spec_at_rs+ionized_elec
            input_spec.rs = rs
            f_elec = lowE_electrons.compute_fs(
                MEDEA_interp, input_spec, 1.-x_vec_for_f[0],
                rate_func_eng(rs), dt
            )

            # print(rs, 'f_cont_low_elec: ', f_elec[0])

            norm = phys.nB*rs**3 / rate_func_eng(rs)
            f_elec = {
                'H ion': f_elec[2] + highengdep_at_rs[0] * norm,
                'He ion': f_elec[3],
                'Lya': f_elec[1] + highengdep_at_rs[1] * norm,
                'heat': f_elec[4] + highengdep_at_rs[2] * norm,
                'cont': f_elec[0] + highengdep_at_rs[3] * norm,
                'err': deposited_err/dt * norm
            }

            # print(rs, 'f_cont_high_elec: ', highengdep_at_rs[3]*norm, 'f_cont_elec: ', f_elec['cont'])

        # if compute_fs_method == 'HeII' and rs > reion_rs:

        #     # For 'HeII', stick with 'no_He' until after
        #     # reionization kicks in.

        #     compute_f_phot_method = 'old'



        f_phot = phot_dep.compute_fs(
            lowengphot_spec_at_rs,
            x_vec_for_f, rate_func_eng(rs), dt,
            method='old', cross_check=cross_check
        )
        # print(rs, 'f_cont_elec', f_elec['cont'],  'f_cont_phot: ', f_phot['cont'])

        # Compute f for TLA: sum of electron and photon contributions
        f_H_ion = f_phot['H ion'] + f_elec['H ion']
        f_He_ion = f_phot['HeI ion'] + f_phot['HeII ion'] + f_elec['He ion']
        f_Lya = f_phot['H exc'] + f_elec['Lya']
        if elec_method == 'eff':
            f_exc = f_elec['exc']
        else:
            f_exc = 0.

        f_heat = f_elec['heat']
        # Including f_elec['cont'] here would be double-counting.
        # It's just deposited_cont, which is accounted for in
        # the distortion already.
        # Just revert to the old way of calculating f: we either use the old way or use the full MLA, which doesn't care about f_cont anyway.
        f_cont = f_phot['cont'] + f_elec['cont']
        # This keeps track of numerical error from ICS,
        # which is absent when there are no electrons
        f_err = f_elec['err']

        # !!! This is MEDEA's old method, should be updated
        # if elec_processes and elec_method == 'MEDEA':
        #     deposited_exc = {state: np.dot(
        #         deposited_exc_arr[state], tot_spec_elec.N
        #     ) for state in H_states}

        #     # Probabilities that nl state cascades to 2p state
        #     Ps = {'2p': 1.0000, '2s': 0.0, '3p': 0.0,
        #           '4p': 0.2609, '5p': 0.3078, '6p': 0.3259,
        #           '7p': 0.3353, '8p': 0.3410, '9p': 0.3448, '10p': 0.3476}

        #     f_cont += sum([
        #         deposited_exc[state] * (
        #             1-Ps[state]  # 2s->1s
        #             + Ps[state] * (1-phys.lya_eng/phys.H_exc_eng(state))
        #         )
        #         for state in Ps.keys()]) / dt * norm


        # if compute_fs_method == 'no_He':
        #     # The old method neglects helium.
        #     f_He_ion = 0.
        # else:
        #     f_He_ion = (
        #         f_phot['HeI ion'] + f_phot['HeII ion'] + f_elec['He ion']
        #     )

        # elif elec_method == 'old':

        #     #### This should be the old method, but somehow we're bringing in new method stuff, like the comment about f _cont.

        #     f_raw = compute_fs_OLD(
        #         MEDEA_interp, lowengelec_spec_at_rs, lowengphot_spec_at_rs,
        #         x_vec_for_f, rate_func_eng(rs), dt,
        #         highengdep_at_rs, method=compute_fs_method, cross_check=cross_check
        #     )

        #     # Compute f for TLA: sum of low and high.
        #     f_H_ion = f_raw[0][0] + f_raw[1][0]
        #     f_Lya = f_raw[0][2] + f_raw[1][2]
        #     f_heat = f_raw[0][3] + f_raw[1][3]

        #     # No need to add f_raw[1][4]. It's already accounted for in
        #     # lowengphot_spec_at_rs.
        #     f_cont = f_raw[0][4]
        #     # This keeps track of numerical error from ICS,
        #     # which is absent when there are no electrons.
        #     f_err = 0

        #     if compute_fs_method == 'old':
        #         f_He_ion = 0.
        #     else:
        #         f_He_ion = f_raw[0][1] + f_raw[1][1]

        # Save the f_c(z) values.
        f_c = np.concatenate((
            f_c,
            [[f_H_ion, f_He_ion, f_Lya, f_heat, f_cont, f_err, f_exc]]
        ))

        # Now that we have f's, calculate the distortion contribution
        if distort:
            # Add atomic contribution to the distortion from this step
            streaming_lowengphot.N += atomic_dist_spec.N

            # Add heating contribution to the distortion from this step
            xe = x_arr[-1, 0] + x_arr[-1, 1] # not including HeIII
            J = 8 * phys.thomson_xsec * (4 * phys.stefboltz / phys.c) * phys.TCMB(rs)**4 * xe * phys.c / 3 / (1 + phys.chi + xe) / phys.me / phys.hubble(rs)
            J_cond = 100
            if J < J_cond:
                TmTr = Tm_arr[-1] - phys.TCMB(rs)
            else:
                dTdz_dm = - phys.dtdz(rs)*(
                    f_heat * rate_func_eng(rs)
                ) / (3/2 * phys.nH * rs**3 * (1 + phys.chi + xe))
                TmTr = - (phys.TCMB(rs) / J) + (- dTdz_dm / phys.dtdz(rs) / phys.hubble(rs)) / J
            dydz = TmTr * phys.thomson_xsec * xe * phys.nH * rs**3 * phys.c / phys.me / rs / phys.hubble(rs)
            y = dydz * (rs * (1 - np.exp(-dlnz * coarsen_factor)))
            y_heat_spec = phys.ymu_distortion(dist_eng, y, rs, 'y')
            streaming_lowengphot.N += y_heat_spec.N

            append_distort_spec(streaming_lowengphot)

            # Total distortion at this step
            tmp_distortion = out_distort_specs.copy()
            tmp_distortion.redshift(rs)
            distortion = tmp_distortion.sum_specs()


        #####################################################################
        #####################################################################
        # ********* AFTER THIS, COMPUTE QUANTITIES FOR NEXT STEP *********  #
        #####################################################################
        #####################################################################

        # Define the next redshift step.
        next_rs = np.exp(np.log(rs) - dlnz * coarsen_factor)

        #####################################################################
        #####################################################################
        # TLA Integration                                                   #
        #####################################################################
        #####################################################################

        # Initial conditions for the TLA, (Tm, xHII, xHeII, xHeIII).
        # This is simply the last set of these variables.
        init_cond_TLA = np.array(
            [Tm_arr[-1], x_arr[-1, 0], x_arr[-1, 1], 0]
        )

        if(np.any(np.isnan(init_cond_TLA))):
            print(rs, init_cond_TLA)
            raise ValueError('Encountered nan in Tm or x')

        if first_iter:
            # Solve using the TLA rate coefficients.
            # Don't use any DM excitation if reprocess_distortion == True,
            # since excitation is calculated in MLA_funcs.

            if reprocess_distortion:

                new_vals = tla.get_history(
                    np.array([rs, next_rs]), init_cond=init_cond_TLA,
                    f_H_ion=f_H_ion, f_H_exc=0., f_heating=f_heat,
                    injection_rate=rate_func_eng, high_rs=high_rs,
                    reion_switch=reion_switch, reion_rs=reion_rs,
                    reion_method=reion_method, heat_switch=heat_switch,
                    DeltaT=DeltaT, alpha_bk=alpha_bk,
                    photoion_rate_func=photoion_rate_func,
                    photoheat_rate_func=photoheat_rate_func,
                    xe_reion_func=xe_reion_func, helium_TLA=helium_TLA,
                    f_He_ion=f_He_ion, mxstep=mxstep, rtol=rtol,
                    recfast_TLA=True, fudge=fudge,
                    MLA_funcs=None
                )

            else:

                new_vals = tla.get_history(
                    np.array([rs, next_rs]), init_cond=init_cond_TLA,
                    f_H_ion=f_H_ion, f_H_exc=f_Lya, f_heating=f_heat,
                    injection_rate=rate_func_eng, high_rs=high_rs,
                    reion_switch=reion_switch, reion_rs=reion_rs,
                    reion_method=reion_method, heat_switch=heat_switch,
                    DeltaT=DeltaT, alpha_bk=alpha_bk,
                    photoion_rate_func=photoion_rate_func,
                    photoheat_rate_func=photoheat_rate_func,
                    xe_reion_func=xe_reion_func, helium_TLA=helium_TLA,
                    f_He_ion=f_He_ion, mxstep=mxstep, rtol=rtol,
                    recfast_TLA=True, fudge=fudge,
                    MLA_funcs=None
                )

        else:
            # Solve using supplied MLA_funcs.

            new_vals = tla.get_history(
                np.array([rs, next_rs]), init_cond=init_cond_TLA,
                f_H_ion=f_H_ion, f_H_exc=None, f_heating=f_heat,
                injection_rate=rate_func_eng, high_rs=high_rs,
                reion_switch=reion_switch, reion_rs=reion_rs,
                reion_method=reion_method, heat_switch=heat_switch,
                DeltaT=DeltaT, alpha_bk=alpha_bk,
                photoion_rate_func=photoion_rate_func,
                photoheat_rate_func=photoheat_rate_func,
                xe_reion_func=xe_reion_func, helium_TLA=helium_TLA,
                f_He_ion=f_He_ion, mxstep=mxstep, rtol=rtol,
                recfast_TLA=False, fudge=fudge,
                MLA_funcs=MLA_funcs
            )

        #####################################################################
        #####################################################################
        # Photon Cooling Transfer Functions                                 #
        #####################################################################
        #####################################################################

        # Get the transfer functions for this step.
        if (
            not backreaction
            and not (compute_fs_method == 'HeII' and rs <= reion_rs)
        ):
            # Interpolate using the baseline solution.
            xHII_to_interp = phys.x_std(rs)
            xHeII_to_interp = phys.x_std(rs, 'HeII')
        else:
            # Interpolate using the current xHII, xHeII values.
            xHII_to_interp = x_arr[-1, 0]
            xHeII_to_interp = x_arr[-1, 1]

        if next_rs > end_rs:

            # Only compute the transfer functions if next_rs > end_rs.
            # Otherwise, we won't be using them.

            highengphot_tf, lowengphot_tf, lowengelec_tf, highengdep_arr = (
                get_tf(
                    rs, xHII_to_interp, xHeII_to_interp,
                    dlnz, coarsen_factor=coarsen_factor, verbose=verbose
                )
            )

            # Get the spectra for the next step by applying the
            # transfer functions.
            highengdep_at_rs = np.dot(
                np.swapaxes(highengdep_arr, 0, 1),
                out_highengphot_specs[-1].N
            )

            highengphot_spec_at_rs = highengphot_tf.sum_specs(
                out_highengphot_specs[-1]
            )

            lowengphot_spec_at_rs = lowengphot_tf.sum_specs(
                out_highengphot_specs[-1]
            )

            lowengelec_spec_at_rs = lowengelec_tf.sum_specs(
                out_highengphot_specs[-1]
            )

            highengphot_spec_at_rs.rs = next_rs
            lowengphot_spec_at_rs.rs = next_rs
            lowengelec_spec_at_rs.rs = next_rs

            # Only save if next_rs > end_rs, since these are the x, Tm
            # values for the next redshift.

            # Save the x, Tm data for the next step in x_arr and Tm_arr.
            Tm_arr = np.append(Tm_arr, new_vals[-1, 0])

            if helium_TLA:
                # Append the calculated xHe to x_arr.
                x_arr = np.append(
                        x_arr,  [[new_vals[-1, 1], new_vals[-1, 2]]], axis=0
                    )
            else:
                # Append the baseline solution value.
                x_arr = np.append(
                    x_arr, [[new_vals[-1, 1], phys.x_std(next_rs, 'HeII')]], axis=0
                )

        # Re-define existing variables.
        rs = next_rs
        dt = dlnz * coarsen_factor/phys.hubble(rs)


    #########################################################################
    #########################################################################
    # END OF LOOP! END OF LOOP!                                             #
    #########################################################################
    #########################################################################

    if use_tqdm:
        pbar.close()

    # Some processing to get the data into presentable shape.
    f = {
        'H ion':  f_c[:, 0],
        'He ion': f_c[:, 1],
        'Lya':    f_c[:, 2],
        'heat':   f_c[:, 3],
        'cont':   f_c[:, 4],
        'err':    f_c[:, 5],
        'eff_exc':    f_c[:, 6]
    }

    # Redshift the distortion to today
    if distort:
        distortion.redshift(1)

    data = {
        'rs': out_highengphot_specs.rs,
        'x': x_arr, 'Tm': Tm_arr,
        'highengphot': out_highengphot_specs,
        'lowengphot': out_lowengphot_specs,
        'lowengelec': out_lowengelec_specs,
        'distortions': out_distort_specs,
        'distortion': distortion,
        'f': f
    }

    if distort:
        data['MLA'] = np.array(MLA_data)
        # Only save states up to 4f.
        data['x_full'] = np.array(x_full_data)[:,:10]

    # End of the iteration.
    iterations -= 1

    # If iteration > 0, then call this function recursively to perform next iteration.
    if iterations > 0:

        MLA_funcs_next_iter = [
            interp1d(MLA_data[0], MLA_data[i], fill_value = 'extrapolate') for i in range(1, 4)
        ]

        if prev_output is not None:

            prev_output.append(data)

        else:

            prev_output = [data]

        # change the options for the next run.
        options['MLA_funcs'] = MLA_funcs_next_iter
        options['iterations'] = iterations
        options['first_iter'] = False
        options['prev_output'] = prev_output


        return evolve(**options)

    else:

        if prev_output is None:

            return data

        else:

            prev_output.append(data)
            return prev_output


def get_elec_cooling_data(eleceng, photeng, H_states):
    """
    Returns electron cooling data for use in :func:`main.evolve`.

    Parameters
    ----------
    eleceng : ndarray
        The electron energy abscissa.
    photeng : ndarray
        The photon energy abscissa.

    Returns
    -------
    tuple of ndarray
        A tuple with containing 3 tuples. The first tuple contains the
        normalized collisional ionization scattered electron spectrum for
        HI, HeI and HeII. The second contains the normalized collisional
        excitation scattered electron spectrum for HI, HeI and HeII. The
        last tuple is an :class:`.EnglossRebinData` object for use in
        rebinning ICS energy loss data to obtain the ICS scattered
        electron spectrum.
    """

    # atoms that take part in electron cooling process through ionization
    atoms = ['HI', 'HeI', 'HeII']
    # We keep track of specific states for hydrogen,
    # but not for HeI and HeII !!!
    exc_types = H_states+['HeI', 'HeII']

    exc_potentials = {state: phys.H_exc_eng(state) for state in H_states}
    exc_potentials['HeI'] = phys.He_exc_eng['23s']
    exc_potentials['HeII'] = 4*phys.lya_eng

    # Compute the (normalized) collisional ionization spectra.
    coll_ion_sec_elec_specs = {species:
                               phys.coll_ion_sec_elec_spec(
                                   eleceng, eleceng, species=species
                               ) for species in atoms}

    # Make empty dictionaries
    coll_exc_sec_elec_specs = {}
    coll_exc_sec_elec_tf = {}

    # Compute the (normalized) collisional excitation spectra.
    id_mat = np.identity(eleceng.size)

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

        # Rebin the data so that the spectra stored above now have an abscissa
        # of eleceng again (instead of eleceng - phys.lya_eng for HI etc.)
        coll_exc_sec_elec_tf[exc].rebin(eleceng)

        # Put them in a dictionary
        coll_exc_sec_elec_specs[exc] = coll_exc_sec_elec_tf[exc].grid_vals

    # Store the ICS rebinning data for speed. Contains information
    # that makes converting an energy loss spectrum to a scattered
    # electron spectrum fast.
    ics_engloss_data = EnglossRebinData(eleceng, photeng, eleceng)

    return (
        coll_ion_sec_elec_specs, coll_exc_sec_elec_specs, ics_engloss_data
    )


def get_elec_cooling_dataTMP(eleceng, photeng):
    """
    Returns electron cooling data for use in :func:`main.evolve`.

    Parameters
    ----------
    eleceng : ndarray
        The electron energy abscissa.
    photeng : ndarray
        The photon energy abscissa.

    Returns
    -------
    tuple of ndarray
        A tuple with containing 3 tuples. The first tuple contains the
        normalized collisional ionization scattered electron spectrum for
        HI, HeI and HeII. The second contains the normalized collisional
        excitation scattered electron spectrum for HI, HeI and HeII. The
        last tuple is an
        :class:`.EnglossRebinData` object for use in rebinning ICS energy loss data to obtain the ICS scattered
        electron spectrum.
    """

    # Compute the (normalized) collisional ionization spectra.
    coll_ion_sec_elec_specs = (
        phys.coll_ion_sec_elec_spec(eleceng, eleceng, species='HI'),
        phys.coll_ion_sec_elec_spec(eleceng, eleceng, species='HeI'),
        phys.coll_ion_sec_elec_spec(eleceng, eleceng, species='HeII')
    )
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

    # Store the ICS rebinning data for speed. Contains information
    # that makes converting an energy loss spectrum to a scattered
    # electron spectrum fast.
    ics_engloss_data = EnglossRebinData(eleceng, photeng, eleceng)

    return (
        coll_ion_sec_elec_specs, coll_exc_sec_elec_specs, ics_engloss_data
    )


def get_tf(rs, xHII, xHeII, dlnz, coarsen_factor=1,verbose=0):
    """
    Returns the interpolated transfer functions.

    Parameters
    ----------
    rs : float
        The current redshift (1+z) to obtain the functions.
    xHII : float
        The ionization fraction nHII/nH.
    xHeII : float
        The ionization fraction nHeII/nH.
    dlnz : float
        The dlnz of the output transfer functions.
    coarsen_factor : int
        The coarsening factor for the output transfer functions.

    Returns
    -------
    tuple
        Contains the high-energy photon, low-energy photon, low-energy
        electron, upscattered CMB photon energy and high-energy deposition
        transfer functions.
    """

    # Load data.

    dep_tf_data = load_data('dep_tf',verbose=verbose)

    highengphot_tf_interp = dep_tf_data['highengphot']
    lowengphot_tf_interp = dep_tf_data['lowengphot']
    lowengelec_tf_interp = dep_tf_data['lowengelec']
    highengdep_interp = dep_tf_data['highengdep']

    if coarsen_factor > 1:
        # rs_to_interpolate = rs
        rs_to_interpolate = np.exp(np.log(rs) - dlnz * coarsen_factor/2)
    else:
        rs_to_interpolate = rs

    highengphot_tf = highengphot_tf_interp.get_tf(
        xHII, xHeII, rs_to_interpolate
    )
    lowengphot_tf = lowengphot_tf_interp.get_tf(
        xHII, xHeII, rs_to_interpolate
    )
    lowengelec_tf = lowengelec_tf_interp.get_tf(
        xHII, xHeII, rs_to_interpolate
    )
    highengdep_arr = highengdep_interp.get_val(
        xHII, xHeII, rs_to_interpolate
    )

    if coarsen_factor > 1:
        prop_tf = np.zeros_like(highengphot_tf._grid_vals)
        for i in np.arange(coarsen_factor):
            prop_tf += matrix_power(highengphot_tf._grid_vals, i)
        lowengphot_tf._grid_vals = np.matmul(
            prop_tf, lowengphot_tf._grid_vals
        )
        lowengelec_tf._grid_vals = np.matmul(
            prop_tf, lowengelec_tf._grid_vals
        )
        highengphot_tf._grid_vals = matrix_power(
            highengphot_tf._grid_vals, coarsen_factor
        )
        # cmbloss_arr = np.matmul(prop_tf, cmbloss_arr)/coarsen_factor
        highengdep_arr = (
            np.matmul(prop_tf, highengdep_arr)/coarsen_factor
        )

    return(
        highengphot_tf, lowengphot_tf,
        lowengelec_tf, highengdep_arr
    )

    # return (
    #     highengphot_tf, lowengphot_tf, lowengelec_tf,
    #     cmbloss_arr, highengdep_arr
    # )


def embarrassingly_parallel_evolve(DM_params, ind, evolve_options_dict, save_dir, file_name_str):
    """
    Embarrassingly parallel scan over DM parameters and saves the output.

    Parameters
    ----------
    DM_params : list of dict
        Dark matter parameters, listed as {'pri':'elec' or 'phot', 'DM_process':'swave' or 'decay', 'mDM':mDM, 'inj_param':sigmav or tau}
    ind : int
        Index of `DM_params` to run for this job.
    evolve_options_dict : dict
        Options to be passed to :func:`main.evolve`. Options that are not specified are set to `None`.
    save_dir : string
        Directory to save the output in.
    file_name_str : string
        Additional descriptive string for file.

    Returns
    -------
    None

    """

    params = DM_params[ind]

    data = evolve(
            DM_process=params['DM_process'], mDM=params['mDM'],
            lifetime=params['inj_param'], sigmav=params['inj_param'],
            primary=params['pri']+'_delta', **evolve_options_dict
    )


    fn = (
        save_dir
        +params['pri']+'_'+params['DM_process']
        +'_'+'log10mDM_'+'{0:2.4f}'.format(np.log10(params['mDM']))
        +'_'+'log10param_'+'{0:2.4f}'.format(np.log10(params['inj_param']))
        +'_'+file_name_str+'_ind_'+str(ind)+'.p'
    )

    pickle.dump({'DM_params':params, 'ind':ind, 'data':data}, open(fn, 'wb'))

    print('Successfully produced file: ', fn)

    return None



def evolve_for_CLASS(
    save_dir, file_name_str, save_DH=False,
    in_spec_elec=None, in_spec_phot=None,
    rate_func_N=None, rate_func_eng=None,
    DM_process=None, primary=None, mDM=None,
    sigmav=None, lifetime=None,
    struct_boost=None,
    start_rs=None, high_rs=np.inf, end_rs=4,
    helium_TLA=False, reion_switch=False,
    reion_rs=None, reion_method='Puchwein',
    heat_switch=True, photoion_rate_func=None, photoheat_rate_func=None,
    xe_reion_func=None, DeltaT=None, alpha_bk=None,
    init_cond=None, coarsen_factor=1, backreaction=True,
    compute_fs_method='no_He', elec_method='new',
    distort=False, fudge=1.125, nmax=10, fexc_switch=True, MLA_funcs=None,
    cross_check=False, reprocess_distortion=True, simple_2s1s=False, iterations=1,
    first_iter=True, init_distort_file=None, prev_output=None,
    use_tqdm=True, tqdm_jupyter=True, mxstep=1000, rtol=1e-4, verbose=0
):
    """
    Run evolve() and save output in format easily readable for use with CLASS.

    Parameters
    ----------
    save_dir : string
        Directory to save the output in.
    file_name_str : string
        Additional descriptive string for file.
    save_DH : bool
        If true, save DarkHistory output.
    See documentation for evolve() for other options.

    Returns
    -------
    None

    """
    # Make a copy of all the function arguments
    params = locals().copy()
    pars_save = params.copy() # because you can't pickle functions like struct_boost_func, which get passed to evolve

    # Variable for naming files later
    if params['DM_process'] == 'decay':
        inj_param = params['lifetime']
    else:
        inj_param = params['sigmav']

    # If structure boost is specified, give evolve() the right function
    if params['struct_boost'] is not None:
        params['struct_boost'] = phys.struct_boost_func(params['struct_boost']) # WQ: not the most flexible way to do this, but okay

    # If init_distort_file is specified, load the inital spectral distortion
    # Assumes spectrum is given in CLASS units at z=0
    if params['init_distort_file'] is not None:
        filename = params.pop('init_distort_file')
        init_dist_arr = np.loadtxt(filename)

        # Convert frequency to energy
        hplanck = phys.hbar * 2*np.pi
        dist_eng = init_dist_arr[:,1] * 1e9 * hplanck

        # Convert spectrum to dNdE
        convert = phys.nB * dist_eng * hplanck * phys.c / (4*np.pi) * phys.ele * 1e4
        dist_dNdE = init_dist_arr[:,2] * 1e-26 / convert

        # CLASS binning is relatively coarse
        # For smooth initial distortion, interpolate so that we don't have rebinning artifacts
        fine_eng = np.exp(np.linspace(np.log(hplanck*1e8), np.log(phys.rydberg), 2000))
        init_dist_interp = interp1d(dist_eng, dist_dNdE, bounds_error=False, fill_value=(0,0))

        params['init_distort'] = Spectrum(dist_eng, dist_dNdE, rs=1, spec_type='dNdE')
        # discretize(fine_eng, init_dist_interp) 
        # Spectrum(
        #     fine_eng, # change from nu in GHz to eV
        #     init_dist_interp(fine_eng), # change from 10^-26 W m^-2 Hz^-1 sr^-1 to dNdE
        #     rs=1, spec_type='dNdE'
        # )

    # Pop the arguments that are not taken by evolve()
    save_dir = params.pop('save_dir')
    file_name_str = params.pop('file_name_str')
    save_DH = params.pop('save_DH')

    # Run evolve() and save DH data if option is set
    DH_data = evolve(**params)
    if save_DH:
        fn = (
            save_dir+'/'
            +params['primary']+'_'+params['DM_process']
            +'_'+'log10mDM_'+'{0:2.4f}'.format(np.log10(params['mDM']))
            +'_'+'log10param_'+'{0:2.4f}'.format(np.log10(inj_param))
            +'_'+file_name_str+'_DHdata.p'
        )
        pickle.dump({'DM_params':pars_save, 'data':DH_data}, open(fn, 'wb'))
        print('Successfully produced DH file: ', fn)

    # Keep only last iteration
    if iterations > 1:
        DH_data = DH_data[-1]

    # Repackage output in nice to read format
    # define redshift and data arrays
    dz = 0.5
    z_list = np.arange(0,10000+2*dz,dz)

    early_inds = np.argwhere(1+z_list > start_rs)
    late_inds = np.argwhere(1+z_list <= end_rs)
    DH_inds = np.argwhere((1+z_list <= start_rs)*(1+z_list > end_rs))

    repackaged = np.zeros((len(z_list),4))
    repackaged[:,0] = z_list


    if distort == True:
            # Convert energies to GHz
            eng = DH_data['distortion'].eng # eV
            hplanck = phys.hbar * 2*np.pi
            nu = eng/hplanck/1e9 # GHz

            # Convert dNdE to spectral radiance
            convert = phys.nB * eng * hplanck * phys.c / (4*np.pi) * phys.ele * 1e4 # 1/eV to W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$
            J = 1e26 * convert * DH_data['distortion'].dNdE # 10^-26 W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$

            distortions = np.zeros((len(DH_data['distortion'].eng),2))
            distortions[:,0] = nu
            distortions[:,1] = J
            fn = (
                save_dir+file_name_str+'_distortions_CLASSformat.txt'
            )
            np.savetxt(
                fn, distortions, header=f"{distortions.shape[0]:.0f}\n", comments=""
            )

    # Fill in x_e and T_m
    repackaged[early_inds,1] = phys.x_std(1+repackaged[early_inds,0]) + phys.x_std(1+repackaged[early_inds,0], species='HeII')
    repackaged[early_inds,2] = phys.Tm_std(1+repackaged[early_inds,0])

    repackaged[DH_inds,1] = np.interp(repackaged[DH_inds,0], DH_data['rs'][::-1]-1, (DH_data['x'][:,0] + DH_data['x'][:,1])[::-1])
    repackaged[DH_inds,2] = np.interp(repackaged[DH_inds,0], DH_data['rs'][::-1]-1, DH_data['Tm'][::-1])

    # for i in range(int(repackaged.shape[0])):
    #     distortions[0,i]= print(DH_data['distortion'].eng[i],DH_data['distortion'].dNdE[i])
    #     distortions[1,i]= print(DH_data['distortion'].eng[i],DH_data['distortion'].dNdE[i])
    # print(DH_data['distortions'].rs,DH_data['distortion'].eng,DH_data['distortion'].dNdE)
    # repackaged[DH_inds,2] = np.interp(repackaged[DH_inds,0], DH_data['distortion'].eng, DH_data['Tm'][::-1])

    repackaged[late_inds,1] = 10**interp1d(
        np.log10(1+repackaged[DH_inds[:2].flatten(),0]), np.log10(repackaged[DH_inds[:2].flatten(),1]),
        fill_value="extrapolate", bounds_error=False
        )(np.log10(1+repackaged[late_inds,0]))
    repackaged[late_inds,2] = 10**interp1d(
        np.log10(1+repackaged[DH_inds[:2].flatten(),0]), np.log10(repackaged[DH_inds[:2].flatten(),2]),
        fill_value="extrapolate", bounds_error=False
        )(np.log10(1+repackaged[late_inds,0]))

    repackaged[:,2] /= phys.kB # convert temperature to K

    # Redshift derivative of matter temp
    repackaged[:,3] = np.gradient(repackaged[:,2], 0.5)

    # Save data as text file
#    fn = (
#        save_dir+'/'
#        +params['primary']+'_'+params['DM_process']
#        +'_'+'log10mDM_'+'{0:2.4f}'.format(np.log10(params['mDM']))
#        +'_'+'log10param_'+'{0:2.4f}'.format(np.log10(inj_param))
#        +'_'+file_name_str+'_CLASSformat.txt'
#    )
#    np.savetxt(
#        fn, repackaged, header=f"{repackaged.shape[0]:.0f}\n", comments=""
#    )

    print(f"{repackaged.shape[0]:.0f}\n")
    for i in range(int(repackaged.shape[0])):
        print("%f %f %f %f "%(repackaged[i,0],repackaged[i,1],repackaged[i,2],repackaged[i,3]))
    return
