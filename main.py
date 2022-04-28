""" The main DarkHistory function.

"""
import numpy as np
from numpy.linalg import matrix_power
from scipy.interpolate import interp1d

# from config import data_path, photeng, eleceng
# from tf_data import *

from config import load_data


import darkhistory.physics as phys

from darkhistory.spec import pppc
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec.spectra import Spectra
import darkhistory.spec.transferfunction as tf

from darkhistory.spec.spectools import EnglossRebinData
from darkhistory.spec.spectools import discretize

from darkhistory.electrons import positronium as pos
from darkhistory.electrons.elec_cooling import get_elec_cooling_tf
from darkhistory.electrons.elec_coolingTMP import get_elec_cooling_tf as \
        get_elec_cooling_tfTMP
from darkhistory.low_energy.lowE_deposition import compute_fs as compute_fs_OLD
import darkhistory.low_energy.atomic as atomic

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
    heat_switch=False, DeltaT=0, alpha_bk=0.5,
    photoion_rate_func=None, photoheat_rate_func=None, xe_reion_func=None,
    init_cond=None, coarsen_factor=1, backreaction=True,
    compute_fs_method='no_He', mxstep=1000, rtol=1e-4,
    distort=False, fudge=1.125, nmax=10, fexc_switch=False, MLA_funcs=None,
    use_tqdm=True, cross_check=False, recfast_TLA=None,
    reprocess_distortion=True
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
    end_rs : float, optional
        Final redshift :math:`(1+z)` to evolve to. Default is 1+z = 4.
    reion_switch : bool
        Reionization model included if *True*, default is *False*.
    helium_TLA : bool
        If *True*, the TLA is solved with helium. Default is *False*.
    reion_rs : float, optional
        Redshift :math:`(1+z)` at which reionization effects turn on.
    photoion_rate_func : tuple of functions, optional
        Functions take redshift :math:`1+z` as input, return the
        photoionization rate in s\ :sup:`-1` of HI, HeI and HeII respectively.
        If not specified, defaults to :func:`.photoion_rate`.
    photoheat_rate_func : tuple of functions, optional
        Functions take redshift :math:`1+z` as input, return the photoheating
        rate in s\ :sup:`-1` of HI, HeI and HeII respectively.
        If not specified, defaults to :func:`.photoheat_rate`.
    xe_reion_func : function, optional
        Specifies a fixed ionization history after reion_rs.
    init_cond : tuple of floats
        Specifies the initial (xH, xHe, Tm). Defaults to :func:`.Tm_std`,
        :func:`.xHII_std` and :func:`.xHeII_std` at the *start_rs*.
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
    mxstep : int, optional
        The maximum number of steps allowed for each integration point.
        See *scipy.integrate.odeint()* for more information. Default is *1000*.
    rtol : float, optional
        The relative error of the solution. See *scipy.integrate.odeint()* for
        more information. Default is *1e-4*.
    use_tqdm : bool, optional
        Uses tqdm if *True*. Default is *True*.
    cross_check : bool, optional
        If *True*, compare against 1604.02457 by using original MEDEA files,
        turning off partial binning, etc. Default is *False*.

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

    #####################################
    # Initialization for DM_process     #
    #####################################

    # Load data.
    binning = load_data('binning')
    photeng = binning['phot']
    eleceng = binning['elec']

    dep_tf_data = load_data('dep_tf')

    highengphot_tf_interp = dep_tf_data['highengphot']
    lowengphot_tf_interp = dep_tf_data['lowengphot']
    lowengelec_tf_interp = dep_tf_data['lowengelec']
    highengdep_interp = dep_tf_data['highengdep']

    ics_tf_data = load_data('ics_tf')

    ics_thomson_ref_tf = ics_tf_data['thomson']
    ics_rel_ref_tf = ics_tf_data['rel']
    engloss_ref_tf = ics_tf_data['engloss']

    # If compute_fs_method is 'HeII', must be using instantaneous reionization.
    if compute_fs_method == 'HeII':

        print('Using instantaneous reionization at 1+z = ', reion_rs)

        def xe_func(rs):
            rs = np.squeeze(np.array([rs]))
            xHII = phys.xHII_std(rs)
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
        highengphot_tf_interp.dlnz != lowengphot_tf_interp.dlnz
        or highengphot_tf_interp.dlnz != lowengelec_tf_interp.dlnz
        or lowengphot_tf_interp.dlnz != lowengelec_tf_interp.dlnz
    ):
        raise ValueError('TransferFuncInterp objects must all have the same \
                         dlnz.')

    if in_spec_elec.rs != in_spec_phot.rs:
        raise ValueError('Input spectra must have the same rs.')

    if cross_check:
        print('cross_check has been set to True -- No longer using all MEDEA \
              files and no longer using partial-binning.')

    #####################################
    # Initialization                    #
    #####################################

    # Initialize start_rs for arbitrary injection.
    start_rs = in_spec_elec.rs

    # Initialize the initial x and Tm.
    if init_cond is None:

        # Default to baseline
        xH_init = phys.xHII_std(start_rs)
        xHe_init = phys.xHeII_std(start_rs)

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
        from tqdm import tqdm_notebook as tqdm
        pbar = tqdm(
            total=np.ceil((np.log(rs) - np.log(end_rs))/dlnz/coarsen_factor)
        )

    # Normalization to convert from per injection event to
    # per baryon per dlnz step.
    def norm_fac(rs):
        return rate_func_N(rs) * (
            dlnz * coarsen_factor / phys.hubble(rs) / (phys.nB * rs**3)
        )

    # If there are no electrons, we get a speed-up by ignoring them.
    if (in_spec_elec.totN() > 0) or distort:
        elec_processes = True

        # The excitation states we keep track of in hydrogen
        # We keep track of specific states for hydrogen, but not for
        # HeI and HeII !!!
        method = 'new'  # or 'MEDEA' or 'AcharyaKhatri' or 'new'

        if method == 'AcharyaKhatri':
            H_states = ['2s', '2p', '3p']
            nmax = 3
        elif method == 'MEDEA':
            H_states = ['2s', '2p', '3p', '4p', '5p',
                        '6p', '7p', '8p', '9p', '10p']
            nmax = 10
        else:
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
        if not cross_check:
            (
                coll_ion_sec_elec_specs, coll_exc_sec_elec_specs,
                ics_engloss_data
            ) = get_elec_cooling_data(eleceng, photeng, H_states)
        else:
            import mainTMP
            (
                coll_ion_sec_elec_specs, coll_exc_sec_elec_specs,
                ics_engloss_data
            ) = mainTMP.get_elec_cooling_data(eleceng, photeng)

        # Spectrum of photons emitted from 2s -> 1s de-excitation
        spec_2s1s = generate_spec_2s1s(photeng)

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
    f_c = np.empty((0, 6))

    if distort:

        # Initialize Spectrum object that stores the distortion

        # The spectrum requires a lower bound lower than lowengphot, and finer
        # binning
        hplanck = phys.hbar * 2*np.pi
        dist_eng = np.exp(np.linspace(np.log(hplanck*1e8),
                                      np.log(phys.rydberg), 500))
        #dist_eng = np.sort(np.append(dist_eng,
        #                             atomic.get_transition_energies(nmax)))
        distortion = Spectrum(dist_eng, np.zeros_like(dist_eng),
                              rs=1, spec_type='N')

        # 2s1s spectrum
        dist_2s1s = discretize(dist_eng, phys.dNdE_2s1s)

        if recfast_TLA is None:
            recfast_TLA = False
        elec_processes = True

        if MLA_funcs is None:
            make_MLA = True
        else:
            make_MLA = False

        alpha_MLA_data = np.array([
            [rs, phys.alpha_recomb(Tm_init, 'HI')],
            [rs, phys.alpha_recomb(Tm_init, 'HI')]
        ])

        tau = atomic.tau_np_1s(2, rs)
        xe_init = xH_init + xHe_init
        x2s = atomic.x2s_steady_state(rs, phys.TCMB(rs), Tm_init,
                                      xe_init, phys.xHI_std(rs), tau)
        x2 = 4*x2s
        beta_ion = phys.beta_ion(Tm_init, 'HI')
        beta_MLA_data = np.array([
            [rs, beta_ion*x2],
            [rs, beta_ion*x2]
        ])

        MLA_data = [[phys.alpha_recomb(Tm_init, 'HI')],
                    [beta_ion*x2]]

        if fexc_switch:
            R_1snp = atomic.Hey_R_initial(np.arange(2, nmax+1), 1)

            A_1snp = 1/3 * phys.rydberg / phys.hbar * (
                phys.alpha * (1-1/np.arange(2, nmax+1)**2)
            )**3 * R_1snp**2

        else:
            A_1snp = None

    else:
        distortion = None
        # Object to help us interpolate over MEDEA results.
        MEDEA_interp = lowE_electrons.make_interpolator(
            interp_type='2D', cross_check=False
        )
        alpha_MLA, beta_MLA = None, None
        if recfast_TLA is None:
            recfast_TLA = True

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
                                                 eleceng, x_at_rs, method='He')
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
                xHII_elec_cooling = phys.xHII_std(rs)
                xHeII_elec_cooling = phys.xHeII_std(rs)

            # Create the electron transfer functions
            if not cross_check:
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
                        method=method, H_states=H_states, spec_2s1s=spec_2s1s
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
                # Lyman-alpha excitation
                deposited_Lya = np.dot(
                    deposited_exc_arr['2p'], tot_spec_elec.N
                )
                # heating
                deposited_heat = np.dot(
                    deposited_heat_arr, tot_spec_elec.N
                )
                # ICS
                deposited_cont = np.dot(
                    ICS_engloss_vec, tot_spec_elec.N
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

            # def beta_MLA(logrs):
            #    rs = np.exp(logrs)

            #    tau = atomic.tau_np_1s(2, rs)
            #    xe = phys.xHII_std(rs)
            #    Tm = phys.Tm_std(rs)
            #    Tr = phys.TCMB(rs)
            #    x2s = atomic.x2s_steady_state(rs, Tr, Tm, xe, 1-xe, tau)
            #    x2 = 4*x2s
            #    beta_ion = phys.beta_ion(Tm, 'HI')

                return np.log(beta_ion*x2)

                # def alpha_MLA(rs):
                #     return phys.alpha_recomb(phys.Tm_std(rs), 'HI')

            # !!! Need to deal with distort == False case
            if distort:

                # Phase space density for the distortion
                if reprocess_distortion:
                    prefac = phys.nB * (phys.hbar*phys.c*rs)**3 * np.pi**2
                    Delta_f = interp1d(
                        dist_eng, prefac * distortion.dNdE/dist_eng**2,
                        bounds_error=False, fill_value=(0, 0)
                    )
                else:
                    def Delta_f(ee):
                        return 0

                alpha_MLA_data[0], beta_MLA_data[0] = (
                    alpha_MLA_data[1], beta_MLA_data[1]
                )

                x_1s = 1-x_arr[-1, 0]
                # resonant photons will be absorbed when passed through
                # the following function
                in_distortion = distortion.copy()
                (
                    alpha_MLA_data[1][1], beta_MLA_data[1][1], atomic_dist_spec
                ) = atomic.get_distortion_and_ionization(
                    rs, dt, x_1s, Tm_arr[-1], nmax, dist_eng,
                    Delta_f, cross_check,
                    True, True, dist_2s1s,
                    fexc_switch, deposited_exc_arr,
                    tot_spec_elec, distortion,
                    H_states, rate_func_eng,
                    A_1snp, stimulated_emission=True
                )
                MLA_data[0].append(alpha_MLA_data[1][1])
                MLA_data[1].append(beta_MLA_data[1][1])

                # Subtract off absorbed photons
                atomic_dist_spec.N -= in_distortion.N - distortion.N

                alpha_MLA_data[1][0], beta_MLA_data[1][0] = rs, rs

                if make_MLA:

                    alpha_MLA = interp1d(
                        alpha_MLA_data[:, 0],
                        alpha_MLA_data[:, 1],
                        kind='linear',
                        fill_value='extrapolate')

                    beta_MLA = interp1d(
                        np.log(beta_MLA_data[:, 0]),
                        np.log(beta_MLA_data[:, 1]),
                        fill_value='extrapolate'
                    )

                else:

                    alpha_MLA = MLA_funcs[0]
                    beta_MLA = MLA_funcs[1]

            #######################################
            # Photons from Injected Electrons     #
            #######################################

            # ICS secondary photon spectrum after electron cooling, per baryon
            ics_phot_spec = ics_sec_phot_tf.sum_specs(tot_spec_elec)
            # print(ics_phot_spec.N)

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
            # If reionization is complete, set the
            # residual fraction of neutral atoms to their measured value
            if x_arr[-1, 0] == 1:
                x_arr[-1, 0] = 1-10**(-4.4)
            if x_arr[-1, 1] == phys.chi:
                x_arr[-1, 1] = phys.chi*(1 - 10**(-4.4))

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

        if distort:
            # Define the spectrum to add to the distortion at this step,
            # without altering the redshift of the original spectrum
            temp_spec = lowengphot_spec_at_rs.copy()
            temp_spec.rebin(dist_eng)
            temp_spec.N += atomic_dist_spec.N

            append_distort_spec(temp_spec)

            # Redshift all contributions to current rs, mask photons>13.6eV,
            # add together to get current distortion
            dist_mask = dist_eng < phys.rydberg

            tmp_distortion = out_distort_specs.copy()
            tmp_distortion.redshift(rs)
            distortion = tmp_distortion.sum_specs() * dist_mask

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
                    1. - phys.xHII_std(rs),
                    phys.chi - phys.xHeII_std(rs),
                    phys.xHeII_std(rs)
            ])

        if not elec_processes:
            f_elec = {chan: 0 for chan in [
                'H ion', 'He ion', 'Lya', 'heat', 'cont', 'err']}

        else:
            # High-energy deposition from input electrons.
            highengdep_at_rs += np.array([
                deposited_H_ion/dt,
                # deposited_He_ion/dt,
                deposited_Lya/dt,
                deposited_heat/dt,
                deposited_cont/dt
            ])

            if not cross_check:

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

            else:
                input_spec = lowengelec_spec_at_rs+ionized_elec
                input_spec.rs = rs
                f_elec = lowE_electrons.compute_fs(
                    MEDEA_interp, input_spec, 1-x_vec_for_f[0],
                    rate_func_eng(rs), dt
                )

                norm = phys.nB*rs**3 / rate_func_eng(rs)
                f_elec = {
                        'H ion': f_elec[2] + highengdep_at_rs[0] * norm,
                        'He ion': f_elec[3],
                        'Lya': f_elec[1] + highengdep_at_rs[1] * norm,
                        'heat': f_elec[4] + highengdep_at_rs[2] * norm,
                        'cont': f_elec[0] + highengdep_at_rs[3] * norm,
                        'err': deposited_err/dt * norm
                }

        if compute_fs_method == 'HeII' and rs > reion_rs:

            # For 'HeII', stick with 'no_He' until after
            # reionization kicks in.

            fs_method = 'old'

        else:

            fs_method = compute_fs_method

        f_phot = phot_dep.compute_fs(
            lowengphot_spec_at_rs,
            x_vec_for_f, rate_func_eng(rs), dt,
            method=fs_method, cross_check=False
        )

        # Compute f for TLA: sum of electron and photon contributions
        f_H_ion = f_phot['H ion'] + f_elec['H ion']
        f_Lya = f_phot['H exc'] + f_elec['Lya']
        f_heat = f_elec['heat']
        # Including f_elec['cont'] here would be double-counting.
        # It's just deposited_cont, which is accounted for in
        # the distortion already.
        f_cont = f_phot['cont']
        # This keeps track of numerical error from ICS,
        # which is absent when there are no electrons
        f_err = f_elec['err']

        if elec_processes and not cross_check and False:
            deposited_exc = {state: np.dot(
                deposited_exc_arr[state], tot_spec_elec.N
            ) for state in H_states}

            # Probabilities that nl state cascades to 2p state
            Ps = {'2p': 1.0000, '2s': 0.0, '3p': 0.0,
                  '4p': 0.2609, '5p': 0.3078, '6p': 0.3259,
                  '7p': 0.3353, '8p': 0.3410, '9p': 0.3448, '10p': 0.3476}

            f_cont += sum([
                deposited_exc[state] * (
                    1-Ps[state]  # 2s->1s
                    + Ps[state] * (1-phys.lya_eng/phys.H_exc_eng(state))
                )
                for state in H_states]) / dt * norm

        if compute_fs_method == 'old':
            # The old method neglects helium.
            f_He_ion = 0.
        else:
            f_He_ion = (
                f_phot['HeI ion'] + f_phot['HeII ion'] + f_elec['He ion']
            )

        if cross_check:

            f_raw = compute_fs_OLD(
                MEDEA_interp, lowengelec_spec_at_rs, lowengphot_spec_at_rs,
                x_vec_for_f, rate_func_eng(rs), dt,
                highengdep_at_rs, method=compute_fs_method, cross_check=False
            )

            # Compute f for TLA: sum of low and high.
            f_H_ion = f_raw[0][0] + f_raw[1][0]
            f_Lya = f_raw[0][2] + f_raw[1][2]
            f_heat = f_raw[0][3] + f_raw[1][3]

            # No need to add f_raw[1][4]. It's already accounted for in
            # lowengphot_spec_at_rs.
            f_cont = f_raw[0][4]
            # This keeps track of numerical error from ICS,
            # which is absent when there are no electrons.
            f_err = 0

            if compute_fs_method == 'old':
                f_He_ion = 0.
            else:
                f_He_ion = f_raw[0][1] + f_raw[1][1]

        # Save the f_c(z) values.
        f_c = np.concatenate((
            f_c,
            [[f_H_ion, f_He_ion, f_Lya, f_heat, f_cont, f_err]]
        ))

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

        # !!!
        # from scipy.interpolate import interp1d
        # prefac = np.pi**2 * phys.nB*(phys.hbar*phys.c*rs)**3
        # E2 = phys.rydberg-phys.lya_eng
        # eng = lowengphot_spec_at_rs.eng
        # dNdE_DM = lowengphot_spec_at_rs.dNdE
        # dnde = interp1d(eng,dNdE_DM)(E2)
        # T2 = E2/np.log(1+E2**2/prefac/dnde)
        # T2=None
        # print(rs, phys.TCMB(rs), T2)
        # if rs>2.5e3 or T2<phys.TCMB(rs):
        #    T2=None

        # Solve the TLA for x, Tm for the *next* step.
        # print(alpha_MLA, phys.alpha_recomb(phys.TCMB(rs),'HI'))

        tau_S = atomic.tau_np_1s(2, rs, 1-x_arr[-1, 0])
        Tr = phys.TCMB(rs)

        x2s = atomic.x2s_steady_state(rs, Tr, Tm_arr[-1],
                                      x_arr[-1, 0], 1-x_arr[-1, 0], tau_S)
        # if alpha_MLA is not None:
        #     print(rs, alpha_MLA(rs)/phys.alpha_recomb(Tm_arr[-1],'HI'),
        #             beta_MLA(rs)/(phys.beta_ion(Tr,'HI')*x2s*4)
        #             )
        # print(phys.peebles_C(x_arr[-1,0],rs))
        # print(rs, init_cond_TLA)
        # print(rs, rate_func_eng(rs))

        if(np.any(np.isnan(init_cond_TLA))):
            print(rs, init_cond_TLA)
            raise ValueError('Encountered nan in Tm or x')

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
            recfast_TLA=recfast_TLA, fudge=fudge,
            alpha_MLA=alpha_MLA, beta_MLA=beta_MLA
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
            xHII_to_interp = phys.xHII_std(rs)
            xHeII_to_interp = phys.xHeII_std(rs)
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
                    dlnz, coarsen_factor=coarsen_factor
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
                    x_arr, [[new_vals[-1, 1], phys.xHeII_std(next_rs)]], axis=0
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
        'err':    f_c[:, 5]
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

    if elec_processes:
        data['MLA'] = np.array(MLA_data)

    return data


# This speeds up the code if main.evolve is used more than once
spec_2s1s = None


def generate_spec_2s1s(photeng):
    global spec_2s1s
    if spec_2s1s is not None:
        return spec_2s1s
    else:
        # A discretized form of the spectrum of 2-photons emitted in the
        # 2s->1s de-excitation process.
        spec_2s1s = discretize(photeng, phys.dNdE_2s1s)/2.
        return spec_2s1s


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


def get_tf(rs, xHII, xHeII, dlnz, coarsen_factor=1):
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

    dep_tf_data = load_data('dep_tf')

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
