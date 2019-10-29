""" The main DarkHistory function.

"""
import numpy as np
from numpy.linalg import matrix_power

# from config import data_path, photeng, eleceng
# from tf_data import *

from config import load_data


import darkhistory.physics as phys

from   darkhistory.spec import pppc
from   darkhistory.spec.spectrum import Spectrum
from   darkhistory.spec.spectra import Spectra
import darkhistory.spec.transferfunction as tf
from   darkhistory.spec.spectools import rebin_N_arr
from   darkhistory.spec.spectools import EnglossRebinData

from darkhistory.electrons import positronium as pos
from darkhistory.electrons.elec_cooling import get_elec_cooling_tf

from darkhistory.low_energy.lowE_deposition import compute_fs
from darkhistory.low_energy.lowE_electrons import make_interpolator

from darkhistory.history import tla

def evolve(
    in_spec_elec=None, in_spec_phot=None,
    rate_func_N=None, rate_func_eng=None,
    DM_process=None, mDM=None, sigmav=None, lifetime=None, primary=None,
    struct_boost=None,
    start_rs=None, end_rs=4, helium_TLA=False,
    reion_switch=False, reion_rs=None, reion_method='Puchwein', heat_switch=True,
    photoion_rate_func=None, photoheat_rate_func=None, xe_reion_func=None,
    init_cond=None, coarsen_factor=1, backreaction=True, 
    compute_fs_method='no_He', mxstep=1000, rtol=1e-4,
    use_tqdm=True, cross_check=False
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
        Function returning number of injection events per volume per time, with redshift :math:`(1+z)` as an input.  
    rate_func_eng : function, optional
        Function returning energy injected per volume per time, with redshift :math:`(1+z)` as an input. 
    DM_process : {'swave', 'decay'}, optional
        Dark matter process to use. 
    sigmav : float, optional
        Thermally averaged cross section for dark matter annihilation. 
    lifetime : float, optional
        Decay lifetime for dark matter decay.
    primary : string, optional
        Primary channel of annihilation/decay. See :func:`.get_pppc_spec` for complete list. Use *'elec_delta'* or *'phot_delta'* for delta function injections of a pair of photons/an electron-positron pair. 
    struct_boost : function, optional
        Energy injection boost factor due to structure formation.
    start_rs : float, optional
        Starting redshift :math:`(1+z)` to evolve from. Default is :math:`(1+z)` = 3000. Specify only for use with *DM_process*. Otherwise, initialize *in_spec_elec.rs* and/or *in_spec_phot.rs* directly. 
    end_rs : float, optional
        Final redshift :math:`(1+z)` to evolve to. Default is 1+z = 4. 
    reion_switch : bool
        Reionization model included if *True*, default is *False*. 
    helium_TLA : bool
        If *True*, the TLA is solved with helium. Default is *False*.
    reion_rs : float, optional
        Redshift :math:`(1+z)` at which reionization effects turn on. 
    photoion_rate_func : tuple of functions, optional
        Functions take redshift :math:`1+z` as input, return the photoionization rate in s\ :sup:`-1` of HI, HeI and HeII respectively. If not specified, defaults to :func:`.photoion_rate`.
    photoheat_rate_func : tuple of functions, optional
        Functions take redshift :math:`1+z` as input, return the photoheating rate in s\ :sup:`-1` of HI, HeI and HeII respectively. If not specified, defaults to :func:`.photoheat_rate`.
    xe_reion_func : function, optional
        Specifies a fixed ionization history after reion_rs.
    init_cond : tuple of floats
        Specifies the initial (xH, xHe, Tm). Defaults to :func:`.Tm_std`, :func:`.xHII_std` and :func:`.xHeII_std` at the *start_rs*. 
    coarsen_factor : int
        Coarsening to apply to the transfer function matrix. Default is 1. 
    backreaction : bool
        If *False*, uses the baseline TLA solution to calculate :math:`f_c(z)`. Default is True.
    compute_fs_method : {'no_He', 'He_recomb', 'He', 'HeII'}
        Method for evaluating helium ionization. 

        * *'no_He'* -- all ionization assigned to hydrogen;
        * *'He_recomb'* -- all photoionized helium atoms recombine; and 
        * *'He'* -- all photoionized helium atoms do not recombine;
        * *'HeII'* -- all ionization assigned to HeII.

        Default is 'no_He'.
    mxstep : int, optional
        The maximum number of steps allowed for each integration point. See *scipy.integrate.odeint()* for more information. Default is *1000*. 
    rtol : float, optional
        The relative error of the solution. See *scipy.integrate.odeint()* for more information. Default is *1e-4*.
    use_tqdm : bool, optional
        Uses tqdm if *True*. Default is *True*. 
    cross_check : bool, optional
        If *True*, compare against 1604.02457 by using original MEDEA files, turning off partial binning, etc. Default is *False*.

    Examples
    --------

    1. *Dark matter annihilation* -- dark matter mass of 50 GeV, annihilation cross section :math:`2 \\times 10^{-26}` cm\ :sup:`3` s\ :sup:`-1`, annihilating to :math:`b \\bar{b}`, solved without backreaction, a coarsening factor of 32 and the default structure formation boost: ::

        import darkhistory.physics as phys

        out = evolve(
            DM_process='swave', mDM=50e9, sigmav=2e-26, 
            primary='b', start_rs=3000., 
            backreaction=False,
            struct_boost=phys.struct_boost_func()
        )

    2. *Dark matter decay* -- dark matter mass of 100 GeV, decay lifetime :math:`3 \\times 10^{25}` s, decaying to a pair of :math:`e^+e^-`, solved with backreaction, a coarsening factor of 16: ::

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
    lowengphot_tf_interp  = dep_tf_data['lowengphot']
    lowengelec_tf_interp  = dep_tf_data['lowengelec']
    highengdep_interp     = dep_tf_data['highengdep']

    ics_tf_data = load_data('ics_tf')

    ics_thomson_ref_tf  = ics_tf_data['thomson']
    ics_rel_ref_tf      = ics_tf_data['rel']
    engloss_ref_tf      = ics_tf_data['engloss']

    # If compute_fs_method is 'HeII', must be using instantaneous reionization. 
    if compute_fs_method == 'HeII':

        #if backreaction: 

        #    raise ValueError('\'HeII\' method cannot be used with backreaction.')

        print('Using instantaneous reionization at 1+z = ', reion_rs)

        def xe_func(rs):
            if rs < reion_rs:
                return 1. + phys.chi
            else:
                return 0.

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
        raise ValueError('in_spec_elec and in_spec_phot must use config.photeng and config.eleceng respectively as abscissa.')

    if (
        highengphot_tf_interp.dlnz    != lowengphot_tf_interp.dlnz
        or highengphot_tf_interp.dlnz != lowengelec_tf_interp.dlnz
        or lowengphot_tf_interp.dlnz  != lowengelec_tf_interp.dlnz
    ):
        raise ValueError('TransferFuncInterp objects must all have the same dlnz.')

    if in_spec_elec.rs != in_spec_phot.rs:
        raise ValueError('Input spectra must have the same rs.')

    if cross_check:
        print('cross_check has been set to True -- No longer using all MEDEA files and no longer using partial-binning.')

    #####################################
    # Initialization                    #
    #####################################

    # Initialize start_rs for arbitrary injection. 
    start_rs = in_spec_elec.rs

    # Initialize the initial x and Tm. 
    if init_cond is None:
        # Default to baseline
        xH_init  = phys.xHII_std(start_rs)
        xHe_init = phys.xHeII_std(start_rs)
        Tm_init  = phys.Tm_std(start_rs)
    else:
        # User-specified.
        xH_init  = init_cond[0]
        xHe_init = init_cond[1]
        Tm_init  = init_cond[2]

    # Initialize redshift/timestep related quantities. 

    # Default step in the transfer function. Note highengphot_tf_interp.dlnz 
    # contains 3 different regimes, and we start with the first. 
    dlnz = highengphot_tf_interp.dlnz[-1]

    # The current redshift. 
    rs   = start_rs

    # The timestep between evaluations of transfer functions, including 
    # coarsening. 
    dt   = dlnz * coarsen_factor / phys.hubble(rs)

    # tqdm set-up.
    if use_tqdm:
        from tqdm import tqdm_notebook as tqdm
        pbar = tqdm(
            total=np.ceil((np.log(rs) - np.log(end_rs))/dlnz/coarsen_factor)
        ) 

    def norm_fac(rs):
        # Normalization to convert from per injection event to 
        # per baryon per dlnz step. 
        return rate_func_N(rs) * (
            dlnz * coarsen_factor / phys.hubble(rs) / (phys.nB * rs**3)
        )

    def rate_func_eng_unclustered(rs):
        # The rate excluding structure formation for s-wave annihilation. 
        # This is the correct normalization for f_c(z). 
        if struct_boost is not None:
            return rate_func_eng(rs)/struct_boost(rs)
        else:
            return rate_func_eng(rs)


    # If there are no electrons, we get a speed up by ignoring them. 
    elec_processes = False
    if in_spec_elec.totN() > 0:
        elec_processes = True

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
        (
            coll_ion_sec_elec_specs, coll_exc_sec_elec_specs,
            ics_engloss_data
        ) = get_elec_cooling_data(eleceng, photeng)

    #########################################################################
    #########################################################################
    # Pre-Loop Preliminaries                                                #
    #########################################################################
    #########################################################################

    # Initialize the arrays that will contain x and Tm results. 
    x_arr  = np.array([[xH_init, xHe_init]])
    Tm_arr = np.array([Tm_init])

    # Initialize Spectra objects to contain all of the output spectra.

    out_highengphot_specs = Spectra([], spec_type='N')
    out_lowengphot_specs  = Spectra([], spec_type='N')
    out_lowengelec_specs  = Spectra([], spec_type='N')

    # Define these methods for speed.
    append_highengphot_spec = out_highengphot_specs.append
    append_lowengphot_spec  = out_lowengphot_specs.append
    append_lowengelec_spec  = out_lowengelec_specs.append

    # Initialize arrays to store f values. 
    f_low  = np.empty((0,5))
    f_high = np.empty((0,5))

    # Initialize array to store high-energy energy deposition rate. 
    highengdep_grid = np.empty((0,4))


    # Object to help us interpolate over MEDEA results. 
    MEDEA_interp = make_interpolator(interp_type='2D', cross_check=cross_check)

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
            lowengphot_spec_at_rs  = in_spec_phot*0
            lowengelec_spec_at_rs  = in_spec_elec*0
            highengdep_at_rs       = np.zeros(4)


        #####################################################################
        #####################################################################
        # Electron Cooling                                                  #
        #####################################################################
        #####################################################################

        # Get the transfer functions corresponding to electron cooling. 
        # These are \bar{T}_\gamma, \bar{T}_e and \bar{R}_c. 
        if elec_processes:

            if (
                backreaction 
                or (compute_fs_method == 'HeII' and not backreaction)# and rs <= reion_rs)
            ):
                xHII_elec_cooling  = x_arr[-1, 0]
                xHeII_elec_cooling = x_arr[-1, 1]
            else:
                xHII_elec_cooling  = phys.xHII_std(rs)
                xHeII_elec_cooling = phys.xHeII_std(rs)

            (
                ics_sec_phot_tf, elec_processes_lowengelec_tf,
                deposited_ion_arr, deposited_exc_arr, deposited_heat_arr,
                continuum_loss, deposited_ICS_arr
            ) = get_elec_cooling_tf(
                    eleceng, photeng, rs,
                    xHII_elec_cooling, xHeII=xHeII_elec_cooling,
                    raw_thomson_tf=ics_thomson_ref_tf, 
                    raw_rel_tf=ics_rel_ref_tf, 
                    raw_engloss_tf=engloss_ref_tf,
                    coll_ion_sec_elec_specs=coll_ion_sec_elec_specs, 
                    coll_exc_sec_elec_specs=coll_exc_sec_elec_specs,
                    ics_engloss_data=ics_engloss_data
                )

            # Apply the transfer function to the input electron spectrum. 

            # Low energy electrons from electron cooling, per injection event.
            elec_processes_lowengelec_spec = (
                elec_processes_lowengelec_tf.sum_specs(in_spec_elec)
            )

            # Add this to lowengelec_at_rs. 
            lowengelec_spec_at_rs += (
                elec_processes_lowengelec_spec*norm_fac(rs)
            )

            # High-energy deposition into ionization, 
            # *per baryon in this step*. 
            deposited_ion  = np.dot(
                deposited_ion_arr,  in_spec_elec.N*norm_fac(rs)
            )
            # High-energy deposition into excitation, 
            # *per baryon in this step*. 
            deposited_exc  = np.dot(
                deposited_exc_arr,  in_spec_elec.N*norm_fac(rs)
            )
            # High-energy deposition into heating, 
            # *per baryon in this step*. 
            deposited_heat = np.dot(
                deposited_heat_arr, in_spec_elec.N*norm_fac(rs)
            )
            # High-energy deposition numerical error, 
            # *per baryon in this step*. 
            deposited_ICS  = np.dot(
                deposited_ICS_arr,  in_spec_elec.N*norm_fac(rs)
            )

            #######################################
            # Photons from Injected Electrons     #
            #######################################

            # ICS secondary photon spectrum after electron cooling, 
            # per injection event.
            ics_phot_spec = ics_sec_phot_tf.sum_specs(in_spec_elec)

            # Get the spectrum from positron annihilation, per injection event.
            # Only half of in_spec_elec is positrons!
            positronium_phot_spec = pos.weighted_photon_spec(photeng) * (
                in_spec_elec.totN()/2
            )
            positronium_phot_spec.switch_spec_type('N')

        # Add injected photons + photons from injected electrons
        # to the photon spectrum that got propagated forward. 
        if elec_processes:
            highengphot_spec_at_rs += (
                in_spec_phot + ics_phot_spec + positronium_phot_spec
            ) * norm_fac(rs)
        else:
            highengphot_spec_at_rs += in_spec_phot * norm_fac(rs)

        # Set the redshift correctly. 
        highengphot_spec_at_rs.rs = rs

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
        if elec_processes:
            # High-energy deposition from input electrons. 
            highengdep_at_rs += np.array([
                deposited_ion/dt,
                deposited_exc/dt,
                deposited_heat/dt,
                deposited_ICS/dt
            ])

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

        if compute_fs_method == 'HeII' and rs > reion_rs:

            # For 'HeII', stick with 'no_He' until after 
            # reionization kicks in.

            f_raw = compute_fs(
                MEDEA_interp, lowengelec_spec_at_rs, lowengphot_spec_at_rs,
                x_vec_for_f, rate_func_eng_unclustered(rs), dt,
                highengdep_at_rs, method='no_He', cross_check=cross_check
            )

        else:

            f_raw = compute_fs(
                MEDEA_interp, lowengelec_spec_at_rs, lowengphot_spec_at_rs,
                x_vec_for_f, rate_func_eng_unclustered(rs), dt,
                highengdep_at_rs, method=compute_fs_method, cross_check=cross_check
            )

        # Save the f_c(z) values.
        f_low  = np.concatenate((f_low,  [f_raw[0]]))
        f_high = np.concatenate((f_high, [f_raw[1]]))

        # print(f_low, f_high)

        # Save CMB upscattered rate and high-energy deposition rate.
        highengdep_grid = np.concatenate(
            (highengdep_grid, [highengdep_at_rs])
        )

        # Compute f for TLA: sum of low and high. 
        f_H_ion = f_raw[0][0] + f_raw[1][0]
        f_exc   = f_raw[0][2] + f_raw[1][2]
        f_heat  = f_raw[0][3] + f_raw[1][3]

        if compute_fs_method == 'old':
            # The old method neglects helium.
            f_He_ion = 0. 
        else:
            f_He_ion = f_raw[0][1] + f_raw[1][1]
        

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
            [Tm_arr[-1], x_arr[-1,0], x_arr[-1,1], 0]
        )

        # Solve the TLA for x, Tm for the *next* step. 
        new_vals = tla.get_history(
            np.array([rs, next_rs]), init_cond=init_cond_TLA,
            f_H_ion=f_H_ion, f_H_exc=f_exc, f_heating=f_heat,
            injection_rate=rate_func_eng_unclustered,
            reion_switch=reion_switch, reion_rs=reion_rs, 
            reion_method=reion_method, heat_switch=heat_switch,
            photoion_rate_func=photoion_rate_func,
            photoheat_rate_func=photoheat_rate_func,
            xe_reion_func=xe_reion_func, helium_TLA=helium_TLA,
            f_He_ion=f_He_ion, mxstep=mxstep, rtol=rtol
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
            xHII_to_interp  = phys.xHII_std(rs)
            xHeII_to_interp = phys.xHeII_std(rs)
        else:
            # Interpolate using the current xHII, xHeII values.
            xHII_to_interp  = x_arr[-1,0]
            xHeII_to_interp = x_arr[-1,1]

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
            
            lowengphot_spec_at_rs  = lowengphot_tf.sum_specs(
                out_highengphot_specs[-1]
            )

            lowengelec_spec_at_rs  = lowengelec_tf.sum_specs(
                out_highengphot_specs[-1]
            )

            highengphot_spec_at_rs.rs = next_rs
            lowengphot_spec_at_rs.rs  = next_rs
            lowengelec_spec_at_rs.rs  = next_rs

            # Only save if next_rs > end_rs, since these are the x, Tm
            # values for the next redshift.

            # Save the x, Tm data for the next step in x_arr and Tm_arr.
            Tm_arr = np.append(Tm_arr, new_vals[-1, 0])

            if helium_TLA:
                # Append the calculated xHe to x_arr. 
                x_arr  = np.append(
                        x_arr,  [[new_vals[-1,1], new_vals[-1,2]]], axis=0
                    )
            else:
                # Append the baseline solution value. 
                x_arr  = np.append(
                    x_arr,  [[new_vals[-1,1], phys.xHeII_std(next_rs)]], axis=0
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

    f_to_return = (f_low, f_high)
    
    # Some processing to get the data into presentable shape. 
    f_low_dict = {
        'H ion':  f_low[:,0],
        'He ion': f_low[:,1],
        'exc':    f_low[:,2],
        'heat':   f_low[:,3],
        'cont':   f_low[:,4]
    }
    f_high_dict = {
        'H ion':  f_high[:,0],
        'He ion': f_high[:,1],
        'exc':    f_high[:,2],
        'heat':   f_high[:,3],
        'cont':   f_high[:,4]
    }

    f = {
        'low': f_low_dict, 'high': f_high_dict
    }

    data = {
        'rs': out_highengphot_specs.rs,
        'x': x_arr, 'Tm': Tm_arr, 
        'highengphot': out_highengphot_specs,
        'lowengphot': out_lowengphot_specs, 
        'lowengelec': out_lowengelec_specs,
        'f': f
    }

    return data

def get_elec_cooling_data(eleceng, photeng):
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
    lowengphot_tf_interp  = dep_tf_data['lowengphot']
    lowengelec_tf_interp  = dep_tf_data['lowengelec']
    highengdep_interp     = dep_tf_data['highengdep']

    if coarsen_factor > 1:
        # rs_to_interpolate = rs
        rs_to_interpolate = np.exp(np.log(rs) - dlnz * coarsen_factor/2)
    else:
        rs_to_interpolate = rs

    highengphot_tf = highengphot_tf_interp.get_tf(
        xHII, xHeII, rs_to_interpolate
    )
    lowengphot_tf  = lowengphot_tf_interp.get_tf(
        xHII, xHeII, rs_to_interpolate
    )
    lowengelec_tf  = lowengelec_tf_interp.get_tf(
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

