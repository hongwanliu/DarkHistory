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
from   darkhistory.spec.spectools import discretize

from darkhistory.electrons import positronium as pos
from darkhistory.electrons.elec_cooling import get_elec_cooling_tf

from darkhistory.photons.phot_dep import compute_fs
from darkhistory.photons.phot_dep import get_ionized_elec
from darkhistory.photons.phot_dep import propagating_lowE_photons_fracs

from darkhistory.history import tla

def evolve(
    in_spec_elec=None, in_spec_phot=None,
    rate_func_N=None, rate_func_eng=None,
    DM_process=None, mDM=None, sigmav=None, lifetime=None, primary=None,
    struct_boost=None,
    start_rs=None, end_rs=4, helium_TLA=False,
    reion_switch=False, reion_rs=None, reion_method='Puchwein', heat_switch=False, DeltaT=0, alpha_bk=0.5,
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
            rs = np.squeeze(np.array([rs]))
            xHII = phys.xHII_std(rs)
            xHII[rs<7] = 1
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
        #xHeIII_init = phys.xHeII_std(start_rs)
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
    elec_processes = True
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

        #Spectrum of photons emitted from 2s -> 1s de-excitation
        spec_2s1s = generate_spec_2s1s(photeng)

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
    f_c  = np.empty((0,5))

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

        x_at_rs = np.array([1. - x_arr[-1, 0], phys.chi - x_arr[-1, 1], x_arr[-1, 1]])
        # Get the transfer functions corresponding to electron cooling. 
        # These are \bar{T}_\gamma, \bar{T}_e and \bar{R}_c. 
        if elec_processes:

            if (
                backreaction 
                or (compute_fs_method == 'HeII' and rs <= reion_rs)
            ):
                xHII_elec_cooling  = x_arr[-1, 0]
                xHeII_elec_cooling = x_arr[-1, 1]
            else:
                xHII_elec_cooling  = phys.xHII_std(rs)
                xHeII_elec_cooling = phys.xHeII_std(rs)

            # Create the electron transfer functions
            (
                ics_sec_phot_tf, crap,#elec_processes_lowengelec_tf,
                deposited_ion_arr, deposited_exc_arr, deposited_heat_arr,
                ICS_engloss_vec, ICS_err_vec,
                deexc_phot_spectra, deposited_Lya_arr
            ) = get_elec_cooling_tf(
                    eleceng, photeng, rs,
                    xHII_elec_cooling, xHeII=xHeII_elec_cooling,
                    raw_thomson_tf=ics_thomson_ref_tf, 
                    raw_rel_tf=ics_rel_ref_tf, 
                    raw_engloss_tf=engloss_ref_tf,
                    coll_ion_sec_elec_specs=coll_ion_sec_elec_specs, 
                    coll_exc_sec_elec_specs=coll_exc_sec_elec_specs,
                    ics_engloss_data=ics_engloss_data,
                    spec_2s1s = spec_2s1s
                    #loweng=eleceng[0]
                )


            ### Apply the transfer function to the input electron spectrum generated in this step ###

            # electrons in this step are comprised of:
            #  promptly injected from DM (in_spec_elec), 
            #  produced by high energy photon processes (lowengelec_spec_at_rs) 
            #  photoionized from atoms (ionized_elec)
            ionized_elec = get_ionized_elec(lowengphot_spec_at_rs, eleceng, x_at_rs, method='He')
            tot_spec_elec = in_spec_elec*norm_fac(rs)+lowengelec_spec_at_rs+ionized_elec

            # deposited energy into ionization, *per baryon in this step*. 
            deposited_H_ion  = np.dot(
                deposited_ion_arr['H'], tot_spec_elec.N
            )
            deposited_He_ion  = np.dot(
                deposited_ion_arr['He'], tot_spec_elec.N
            )
            # Lyman-alpha excitation 
            deposited_Lya  = np.dot(
                deposited_Lya_arr, tot_spec_elec.N
            )
            # heating
            deposited_heat = np.dot(
                deposited_heat_arr, tot_spec_elec.N
            )
            # numerical error
            deposited_err  = np.dot(
                ICS_err_vec, tot_spec_elec.N
            )

            #######################################
            # Photons from Injected Electrons     #
            #######################################

            # ICS secondary photon spectrum after electron cooling, 
            # per injection event.
            ics_phot_spec = ics_sec_phot_tf.sum_specs(tot_spec_elec)

            # secondary photon spectrum from deexcitation of atoms 
            # that were collisionally excited by electrons
            deexc_phot_spec = deexc_phot_spectra.sum_specs(tot_spec_elec)

            # Get the spectrum from positron annihilation, per injection event.
            # Only half of in_spec_elec is positrons!
            positronium_phot_spec = pos.weighted_photon_spec(photeng) * (
                in_spec_elec.totN()/2
            )
            positronium_phot_spec.switch_spec_type('N')


        # Add injected photons + photons from injected electrons + photons from atomic de-excitations
        # to the photon spectrum that got propagated forward. 
        if elec_processes:
            highengphot_spec_at_rs += (
                in_spec_phot + ics_phot_spec + positronium_phot_spec
            ) * norm_fac(rs)
            lowengphot_spec_at_rs = lowengphot_spec_at_rs + deexc_phot_spec
        else:
            highengphot_spec_at_rs += in_spec_phot * norm_fac(rs)

        # Compute the fraction of ionizing photons that free stream within this step
        if (reion_switch == True) & (rs < start_rs):
            # If reionization is complete, set the residual fraction of neutral atoms to their measured value
            if x_arr[-1,0] == 1:
                x_arr[-1,0] = 1-10**(-4.4)
            if x_arr[-1,1] == phys.chi:
                x_arr[-1,1] = phys.chi*(1 - 10**(-4.4))

            lowEprop_mask = propagating_lowE_photons_fracs(lowengphot_spec_at_rs, x_at_rs, dt)
        else:
            lowEprop_mask = np.zeros_like(lowengphot_spec_at_rs.eng)

        # Add this fraction to the propagating photons
        highengphot_spec_at_rs += lowEprop_mask * lowengphot_spec_at_rs


        # Get rid of the lowenergy photons that weren't absorbed -- they're in highengphot now
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
        if elec_processes:
            # High-energy deposition from input electrons. 
            highengdep_at_rs += np.array([
                deposited_H_ion/dt,
                #deposited_He_ion/dt,
                deposited_Lya/dt,
                deposited_heat/dt,
                deposited_err/dt
            ])

            #print(highengdep_at_rs)
            #print(np.array([
            #    deposited_H_ion/dt,
            #    #deposited_He_ion/dt,
            #    deposited_Lya/dt,
            #    deposited_heat/dt,
            #    deposited_err/dt
            #]))

            norm = phys.nB*rs**3 / rate_func_eng(rs)
            # High-energy deposition from input electrons. 
            f_elec = {
                    'H ion'  : highengdep_at_rs[0] * norm,
                    'He ion' : deposited_He_ion/dt * norm,
                    'Lya'    : highengdep_at_rs[1] * norm,
                    'heat'   : highengdep_at_rs[2] * norm,
                    'err'    : highengdep_at_rs[3] * norm
                    }

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

            f_phot = compute_fs(
                lowengphot_spec_at_rs,
                x_vec_for_f, rate_func_eng(rs), dt,
                method='old', cross_check=cross_check
            )

        else:

            f_phot = compute_fs(
                lowengphot_spec_at_rs,
                x_vec_for_f, rate_func_eng(rs), dt,
                method=compute_fs_method, cross_check=cross_check
            )

        # Compute f for TLA: sum of low and high. 
        f_H_ion = f_phot['H ion'] + f_elec['H ion']
        f_Lya   = f_phot['Lya'] + f_elec['Lya']
        f_heat  = f_elec['heat']

        if compute_fs_method == 'old':
            # The old method neglects helium.
            f_He_ion = 0. 
        else:
            f_He_ion = f_phot['HeI ion'] + f_phot['HeII ion'] + f_elec['He ion']
        
        # Save the f_c(z) values.
        f_c = np.concatenate((f_c, [[f_H_ion, f_He_ion, f_Lya, f_heat, f_elec['err']]]))

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
            f_H_ion=f_H_ion, f_H_exc=f_Lya, f_heating=f_heat,
            injection_rate=rate_func_eng,
            reion_switch=reion_switch, reion_rs=reion_rs, 
            reion_method=reion_method, heat_switch=heat_switch, DeltaT=DeltaT, alpha_bk=alpha_bk,
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

    # Some processing to get the data into presentable shape. 
    f = {
        'H ion':  f_c[:,0],
        'He ion': f_c[:,1],
        'Lya':    f_c[:,2],
        'heat':   f_c[:,3],
        'err':    f_c[:,4]
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

# This speeds up the code if main.evolve is used more than once
spec_2s1s = None
def generate_spec_2s1s(photeng):
    global spec_2s1s
    if spec_2s1s != None:
        return spec_2s1s
    else:
        # !!! (Is this off by a factor of h?) 
        # A discretized form of the spectrum of 2-photons emitted in the
        # 2s->1s de-excitation process.
        spec_2s1s = discretize(photeng,phys.dLam2s_dnu)
        return spec_2s1s

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

    # atoms that take part in electron cooling process through ionization
    atoms = ['HI', 'HeI', 'HeII']
    # We keep track of specific states for hydrogen, but not for HeI and HeII !!!
    exc_types  = ['2s', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p', '10p', 'HeI', 'HeII']

    #ionization and excitation energies
    ion_potentials = {'HI': phys.rydberg, 'HeI': phys.He_ion_eng, 'HeII': 4*phys.rydberg}

    exc_potentials         = phys.HI_exc_eng.copy()
    exc_potentials['HeI']  = phys.He_exc_eng['23s']
    exc_potentials['HeII'] = 4*phys.lya_eng

    # Compute the (normalized) collisional ionization spectra.
    coll_ion_sec_elec_specs = {species : phys.coll_ion_sec_elec_spec(eleceng, eleceng, species=species) for species in atoms}

   # Make empty dictionaries
    coll_exc_sec_elec_specs = {}
    coll_exc_sec_elec_tf =  {}

    # Compute the (normalized) collisional excitation spectra.
    id_mat = np.identity(eleceng.size)

    # Electron with energy eleceng produces a spectrum with one particle
    # of energy eleceng - exc_potential.
    for exc in exc_types:
        exc_pot = exc_potentials[exc]
        coll_exc_sec_elec_tf[exc] = tf.TransFuncAtRedshift(
            np.squeeze(id_mat[:, np.where(eleceng > exc_pot)]),
            in_eng = eleceng, rs = -1*np.ones_like(eleceng),
            eng = eleceng[eleceng > exc_pot] - exc_pot,
            dlnz = -1, spec_type = 'N'
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

