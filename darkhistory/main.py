""" The main DarkHistory function."""

import time
import logging
import gc
from tqdm import tqdm

import numpy as np
from numpy.linalg import matrix_power

import astropy.units as u
import astropy.constants as c

from   darkhistory.config import load_data
import darkhistory.physics as phys
from   darkhistory.spec import pppc
from   darkhistory.spec.spectrum import Spectrum
from   darkhistory.spec.spectra import Spectra
import darkhistory.spec.transferfunction as tf
from   darkhistory.spec.spectools import EnglossRebinData
from   darkhistory.electrons import positronium as pos
from   darkhistory.electrons.elec_cooling import get_elec_cooling_tf
from   darkhistory.low_energy.lowE_deposition import compute_fs
from   darkhistory.low_energy.lowE_electrons import make_interpolator
from   darkhistory.history import tla
# SOFTPHOT EDIT
from   darkhistory.soft_photons.soft_photons import SoftPhotonSpectralDistortion, SoftPhotonHistory


def evolve(
    in_spec_elec=None, in_spec_phot=None, rate_func_N=None, rate_func_eng=None, # custom injection API
    DM_process=None, mDM=None, sigmav=None, lifetime=None, primary=None, struct_boost=None, # DM API
    start_rs=3000, end_rs=4, helium_TLA=False,
    reion_switch=False, reion_rs=None,
    photoion_rate_func=None, photoheat_rate_func=None, xe_reion_func=None,
    init_cond=None, coarsen_factor=1, backreaction=True,
    compute_fs_method='no_He', mxstep=1000, rtol=1e-4,
    tf_mode='table', clean_up_tf=True,
    cross_check=False, verbose=0,
):
    """Main evolution function

    Args:
        start_rs (float) : starting redshift 1+z.
        end_rs (float) : ending redshift 1+z.
        reion_switch (bool) : whether to enable reionization.
        helium_TLA (bool) : whether to include helium in the TLA.
        reion_rs (float) : redshift 1+z at which reionization effects turn on.
        photoion_rate_func (tuple of callables) : functions taking 1+z as input, returning the photoionization rate in 1/s of HI, HeI and HeII.
        photoheat_rate_func (tuple of callables) : functions taking 1+z as input, returning the photoheating rate in eV/s of HI, HeI and HeII.
        xe_reion_func (callable) : specifies a fixed ionization history after reion_rs.
        init_cond (tuple of floats, optional) : initial conditions (xH, xHe, Tm [eV]).
        coarsen_factor (int, optional) : coarsening factor for Euler time evolution, defaulting to 1.
        backreaction (bool) : if False, uses the baseline TLA solution to calculate f_c(z).
        compute_fs_method {'no_He', 'He_recomb', 'He'}
        mxstep (int, optional) : The maximum number of steps allowed for each integration point. See *scipy.integrate.odeint()* for more information.
        rtol (float, optional) : The relative error tolerance for the integration.
        cross_check (bool)
        tf_mode {'table', 'nn'} : Specifies transfer function mode being used.
        verbose {0, 1} : set verbosity.
    """
    
    #===== Load data =====

    binning = load_data('binning')
    photeng = binning['phot']
    eleceng = binning['elec']

    if tf_mode == 'table':
        
        dep_tf_data = load_data('dep_tf')
        highengphot_tf_interp = dep_tf_data['highengphot']
        lowengphot_tf_interp  = dep_tf_data['lowengphot']
        lowengelec_tf_interp  = dep_tf_data['lowengelec']
        highengdep_interp     = dep_tf_data['highengdep']
        
        ics_tf_data = load_data('ics_tf')
        ics_thomson_ref_tf = ics_tf_data['thomson']
        ics_rel_ref_tf     = ics_tf_data['rel']
        engloss_ref_tf     = ics_tf_data['engloss']

        dlnz = highengphot_tf_interp.dlnz[-1]
        
    elif tf_mode == 'nn':
        
        import tensorflow
        from nntf.load import load_model
        tensorflow.get_logger().setLevel('ERROR') # disable tf.function retracing warnings
        
        if coarsen_factor != 12:
            logging.warning('Warning: coarsen_factor is set to 12 (required for using nntf).')
            coarsen_factor = 12
        
        dep_tf_data = load_data('hed_tf')
        highengdep_interp = dep_tf_data['highengdep']
        
        tf_helper_data = load_data('tf_helper')
        tf_E_interp   = tf_helper_data['tf_E']
        # hep_lb_interp = tf_helper_data['hep_lb']
        
        nntf_data = load_model('dep_nntf', verbose=verbose)
        hep_nntf = nntf_data['hep_p12']
        prp_nntf = nntf_data['hep_s11']
        lee_nntf = nntf_data['lee']
        lep_tf   = nntf_data['lep']
        
        nntf_data = load_model('ics_nntf', verbose=verbose)
        ics_thomson_ref_tf = nntf_data['ics_thomson'].TransFuncAtRedshift()
        engloss_ref_tf     = nntf_data['ics_engloss'].TransFuncAtRedshift()
        ics_rel_ref_tf     = nntf_data['ics_rel'].TransFuncAtRedshift()

        dlnz = 0.001
        
    else:
        raise ValueError('Invalid transfer function mode (tf_mode)!')
    

    #===== initialize physics =====
    if init_cond:
        xH_init, xHe_init, Tm_init = init_cond
    else:
        xH_init  = phys.xHII_std(start_rs)
        xHe_init = phys.xHeII_std(start_rs)
        Tm_init  = phys.Tm_std(start_rs)

    initial_state = dict(rs=start_rs, Tm=Tm_init, xHII=xH_init, xHeII=xHe_init, phot_spec=None)
    
    
    #===== injection =====
    USE_IN_SPEC_FUNC = False

    if DM_process == 'swave':
        if mDM < eleceng[1]:
            in_spec_elec = pppc.get_pppc_spec(1, eleceng, primary, 'elec') * 0
        else:
            in_spec_elec = pppc.get_pppc_spec(mDM, eleceng, primary, 'elec')
        in_spec_phot = pppc.get_pppc_spec(mDM, photeng, primary, 'phot')
        in_spec_elec.rs = start_rs
        in_spec_phot.rs = start_rs
        in_spec_elec.switch_spec_type('N')
        in_spec_phot.switch_spec_type('N')

        if struct_boost is None:
            struct_boost = lambda rs: 1
        def rate_func_N(rs, **kwargs):
            return phys.inj_rate('swave', rs, mDM=mDM, sigmav=sigmav) * struct_boost(rs) / (2 * mDM)
        def rate_func_eng(rs, **kwargs):
            return phys.inj_rate('swave', rs, mDM=mDM, sigmav=sigmav) * struct_boost(rs)

    elif DM_process == 'decay':
        struct_boost = lambda rs: 1
        in_spec_elec = pppc.get_pppc_spec(mDM, eleceng, primary, 'elec', decay=True)
        in_spec_phot = pppc.get_pppc_spec(mDM, photeng, primary, 'phot', decay=True)
        in_spec_elec.rs = start_rs
        in_spec_phot.rs = start_rs
        in_spec_elec.switch_spec_type('N')
        in_spec_phot.switch_spec_type('N')

        def rate_func_N(rs, **kwargs):
            return phys.inj_rate('decay', rs, mDM=mDM, lifetime=lifetime) / mDM
        def rate_func_eng(rs, **kwargs):
            return phys.inj_rate('decay', rs, mDM=mDM, lifetime=lifetime) 
        
    elif callable(in_spec_phot) and callable(in_spec_elec):
        
        USE_IN_SPEC_FUNC = True
        
        in_spec_elec_func = in_spec_elec
        in_spec_phot_func = in_spec_phot

        # xHeIII not used. No previous photon spectrum.
        in_spec_elec = in_spec_elec_func(start_rs, state=initial_state)
        in_spec_phot = in_spec_phot_func(start_rs, state=initial_state)
        in_spec_elec.rs = start_rs
        in_spec_phot.rs = start_rs
        in_spec_elec.switch_spec_type('N')
        in_spec_phot.switch_spec_type('N')
        
        if not (np.allclose(in_spec_elec.eng, eleceng) and np.allclose(in_spec_phot.eng, photeng)):
            logging.warning('rebinning in_spec_elec and in_spec_phot to config.eleceng and config.photeng respectively.')
            in_spec_elec.rebin(eleceng)
            in_spec_phot.rebin(photeng)
            
        if struct_boost is None:
            struct_boost = lambda rs: 1
        # User must define rate_func_N and rate_func_eng consistently.
        
    else: # custom injection spectrum with fixed spectral shape
        pass # User must define rate_func_N and rate_func_eng consistently.


    rs = start_rs
    dt = dlnz * coarsen_factor / phys.hubble(rs)

    def norm_fac(rs, dt, state=None):
        """Normalization to convert from per injection event to per baryon per dlnz step."""
        return rate_func_N(rs, state=state) * (dt / (phys.nB * rs**3))

    def rate_func_eng_unclustered(rs, state=None):
        """The rate excluding structure formation for s-wave annihilation. This is the correct normalization for f_c(z)."""
        if struct_boost is not None:
            return rate_func_eng(rs, state=state)/struct_boost(rs)
        else:
            return rate_func_eng(rs, state=state)

    elec_processes = (in_spec_elec.totN() > 0)
    if elec_processes:
        # High-Energy Electrons

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

    # SOFTPHOT EDIT
    softphot_point_inj_z = 3000
    softphot_point_inj_injected = False
    softphot_hist = SoftPhotonHistory(init_spec=SoftPhotonSpectralDistortion(z=rs-1))
    photoheat_rate_func = [lambda rs: 0., lambda rs: 0., lambda rs: 0.]
    
    #===== initialize trackers =====
    pbar = tqdm(total=int(np.ceil((np.log(rs) - np.log(end_rs))/dlnz/coarsen_factor)), position=0)

    x_arr  = np.array([[xH_init, xHe_init]])
    Tm_arr = np.array([Tm_init])

    out_highengphot_specs = Spectra([], spec_type='N')
    out_lowengphot_specs  = Spectra([], spec_type='N')
    out_lowengelec_specs  = Spectra([], spec_type='N')

    f_low  = np.empty((0,5))
    f_high = np.empty((0,5))
    highengdep_grid = np.empty((0,4))

    MEDEA_interp = make_interpolator(interp_type='2D', cross_check=cross_check)
    
    #===== Evolution loop =====
    while rs > end_rs:

        pbar.update(1)
        
        #=== First Step Special Cases ===
        if rs == start_rs:
            # Initialize the electron and photon arrays. 
            # These will carry the spectra produced by applying the
            # transfer function at rs to high-energy photons.
            highengphot_spec_at_rs = in_spec_phot*0
            lowengphot_spec_at_rs  = in_spec_phot*0
            lowengelec_spec_at_rs  = in_spec_elec*0
            highengdep_at_rs       = np.zeros(4)

        state = dict(rs=rs, Tm=Tm_arr[-1], xHII=x_arr[-1][0], xHeII=x_arr[-1][1], phot_spec=highengphot_spec_at_rs)
        
        if USE_IN_SPEC_FUNC and rs != start_rs:
            # Except for first step, remake in_spec_elec/phot if necessary
            in_spec_phot = in_spec_phot_func(rs, state=state)
            in_spec_elec = in_spec_elec_func(rs, state=state)
            in_spec_elec.rs = rs
            in_spec_phot.rs = rs
            in_spec_elec.switch_spec_type('N')
            in_spec_phot.switch_spec_type('N')

            # Rebin if necessary
            if not (np.allclose(in_spec_elec.eng, eleceng) and
                    np.allclose(in_spec_phot.eng, photeng)):
                logging.warning('rebinning in_spec_elec and in_spec_phot to config.eleceng and config.photeng respectively.')
                in_spec_elec.rebin(eleceng)
                in_spec_phot.rebin(photeng)

        #=== Electron Cooling ===
        # Get the transfer functions corresponding to electron cooling. 
        # These are \bar{T}_\gamma, \bar{T}_e and \bar{R}_c. 
        if elec_processes:

            if backreaction:
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
            elec_processes_lowengelec_spec = elec_processes_lowengelec_tf.sum_specs(in_spec_elec)

            lowengelec_spec_at_rs += elec_processes_lowengelec_spec * norm_fac(rs, dt, state=state)

            # High-energy deposition into ionization, excitation, heating, numerical error
            # *per baryon in this step*. 
            deposited_ion  = np.dot(deposited_ion_arr,  in_spec_elec.N * norm_fac(rs, dt, state=state))
            deposited_exc  = np.dot(deposited_exc_arr,  in_spec_elec.N * norm_fac(rs, dt, state=state))
            deposited_heat = np.dot(deposited_heat_arr, in_spec_elec.N * norm_fac(rs, dt, state=state))
            deposited_ICS  = np.dot(deposited_ICS_arr,  in_spec_elec.N * norm_fac(rs, dt, state=state))

            #=== Photons from Injected Electrons ===
            # ICS secondary photon spectrum after electron cooling, 
            # per injection event.
            ics_phot_spec = ics_sec_phot_tf.sum_specs(in_spec_elec)

            # Get the spectrum from positron annihilation, per injection event.
            # Only half of in_spec_elec is positrons!
            positronium_phot_spec = pos.weighted_photon_spec(photeng) * (in_spec_elec.totN()/2)
            positronium_phot_spec.switch_spec_type('N')

        # Add injected photons + photons from injected electrons
        # to the photon spectrum that got propagated forward. 
        if elec_processes:
            highengphot_spec_at_rs += (in_spec_phot + ics_phot_spec + positronium_phot_spec) * norm_fac(rs, dt, state=state)
        else:
            highengphot_spec_at_rs += in_spec_phot * norm_fac(rs, dt, state=state)
        # Set the redshift correctly. 
        highengphot_spec_at_rs.rs = rs

        #===== Save the Spectra =====
        # At this point, highengphot_at_rs, lowengphot_at_rs and 
        # lowengelec_at_rs have been computed for this redshift.
        out_highengphot_specs.append(highengphot_spec_at_rs)
        out_lowengphot_specs.append(lowengphot_spec_at_rs)
        out_lowengelec_specs.append(lowengelec_spec_at_rs)

        #===== Compute f_c(z) =====
        if elec_processes: # High-energy deposition from input electrons. 
            highengdep_at_rs += np.array([deposited_ion, deposited_exc, deposited_heat, deposited_ICS]) / dt

        if backreaction:
            x_vec_for_f = np.array([1-x_arr[-1, 0], phys.chi-x_arr[-1, 1], x_arr[-1, 1]])
        else: # Use baseline values if no backreaction.
            x_vec_for_f = np.array([1-phys.xHII_std(rs), phys.chi-phys.xHeII_std(rs), phys.xHeII_std(rs)])

        f_raw = compute_fs(
            MEDEA_interp, lowengelec_spec_at_rs, lowengphot_spec_at_rs,
            x_vec_for_f, rate_func_eng_unclustered(rs, state=state), dt,
            highengdep_at_rs, method=compute_fs_method, cross_check=cross_check
        )
        f_low  = np.concatenate((f_low,  [f_raw[0]]))
        f_high = np.concatenate((f_high, [f_raw[1]]))

        # Save CMB upscattered rate and high-energy deposition rate.
        highengdep_grid = np.concatenate((highengdep_grid, [highengdep_at_rs]))

        # Compute f for TLA: sum of low and high. 
        f_H_ion = f_raw[0][0] + f_raw[1][0]
        f_exc   = f_raw[0][2] + f_raw[1][2]
        f_heat  = f_raw[0][3] + f_raw[1][3]

        if compute_fs_method == 'old': # The old method neglects helium.
            f_He_ion = 0. 
        else:
            f_He_ion = f_raw[0][1] + f_raw[1][1]
        
        #=== AFTER THIS, COMPUTE QUANTITIES FOR NEXT STEP ===
        # Define the next redshift step. 
        next_rs = np.exp(np.log(rs) - dlnz * coarsen_factor)

        #=== TLA Integration ===
        # Initial conditions for the TLA, (Tm, xHII, xHeII, xHeIII). 
        # This is simply the last set of these variables. 
        init_cond_TLA = np.array([Tm_arr[-1], x_arr[-1,0], x_arr[-1,1], 0])

        # Solve the TLA for x, Tm for the *next* step.
        new_vals = tla.get_history(
            np.array([rs, next_rs]), init_cond=init_cond_TLA, 
            f_H_ion=f_H_ion, f_H_exc=f_exc, f_heating=f_heat,
            injection_rate=lambda rs: rate_func_eng_unclustered(rs, state=state),
            reion_switch=reion_switch, reion_rs=reion_rs,
            photoion_rate_func=photoion_rate_func,
            photoheat_rate_func=photoheat_rate_func,
            xe_reion_func=xe_reion_func, helium_TLA=helium_TLA,
            f_He_ion=f_He_ion, mxstep=mxstep, rtol=rtol
        )

        #=== Photon Cooling Transfer Functions ===
        # Get the transfer functions for this step.
        if not backreaction:
            # Interpolate using the baseline solution.
            xHII_to_interp  = phys.xHII_std(rs)
            xHeII_to_interp = phys.xHeII_std(rs)
        else:
            # Interpolate using the current xHII, xHeII values.
            xHII_to_interp  = x_arr[-1,0]
            xHeII_to_interp = x_arr[-1,1]
        
        if tf_mode == 'table':
            rs_to_interp = rs

            highengphot_tf, lowengphot_tf, lowengelec_tf, highengdep_arr, _ = get_tf(
                rs, xHII_to_interp, xHeII_to_interp, dlnz, dep_tf_data, coarsen_factor=coarsen_factor
            )

            # Get the spectra for the next step by applying the 
            # transfer functions. 
            highengdep_at_rs = np.dot(np.swapaxes(highengdep_arr, 0, 1), out_highengphot_specs[-1].N)
            highengphot_spec_at_rs = highengphot_tf.sum_specs(out_highengphot_specs[-1])
            lowengphot_spec_at_rs  = lowengphot_tf.sum_specs (out_highengphot_specs[-1])
            lowengelec_spec_at_rs  = lowengelec_tf.sum_specs (out_highengphot_specs[-1])
        
        elif tf_mode == 'nn':
            
            rs_to_interp = np.exp(np.log(rs) - dlnz * coarsen_factor/2)
            
            # Predict transfer functions
            rsxHxHe_loc = (xHII_to_interp, xHeII_to_interp, rs_to_interp)
            rsxHxHe_key = { 'rs' : rs_to_interp,
                            'xH' : xHII_to_interp,
                            'xHe': xHeII_to_interp }
            hep_E, prp_E, lee_E, lep_E = tf_E_interp.get_val(*rsxHxHe_loc)
            hep_nntf.predict_TF(E_arr=hep_E, **rsxHxHe_key)
            prp_nntf.predict_TF(E_arr=prp_E, **rsxHxHe_key)
            lee_nntf.predict_TF(E_arr=lee_E, **rsxHxHe_key)
            lep_tf.predict_TF(E_arr=lep_E, **rsxHxHe_key)
            hed_arr = highengdep_interp.get_val(*rsxHxHe_loc)

            # Compound transfer functions
            lep_tf.TF = np.matmul( prp_nntf.TF, lep_tf.TF )
            lee_nntf.TF = np.matmul( prp_nntf.TF, lee_nntf.TF )
            hed_arr = np.matmul( prp_nntf.TF, hed_arr ) / coarsen_factor
            
            # Apply transfer functions
            highengphot_spec_at_rs = hep_nntf( out_highengphot_specs[-1] )
            lowengelec_spec_at_rs  = lee_nntf( out_highengphot_specs[-1] )
            lowengphot_spec_at_rs  = lep_tf( out_highengphot_specs[-1] )
            highengdep_at_rs = np.dot( np.swapaxes(hed_arr, 0, 1), out_highengphot_specs[-1].N )

        #=== Soft photons ===
        # SOFTPHOT EDIT
        if rs < 1 + softphot_point_inj_z and not softphot_point_inj_injected:
            print('Inject!')
            sd_inj = SoftPhotonSpectralDistortion()
            sd_inj.from_point_inj(x_cut=1e3, gamma=3.6, z=rs-1, rho_frac=1e-6)
            softphot_hist.update(sd_inj)
            softphot_point_inj_injected = True

            dTffdz = softphot_hist.spec.dTffdz(rs-1, state=state)
            softphot_hist.dTffdz_arr.append(dTffdz)

        softphot_hist.step(z=rs-1, dz=next_rs-rs, state=state)
        dTffdz = softphot_hist.spec.dTffdz(rs-1, state=state)
        softphot_hist.dTffdz_arr.append(dTffdz)

        def photoheat_rate_func0(rs):
            n_H = phys.nH * rs**3 # [1/cm^3]
            n_He = phys.nHe * rs**3 # [1/cm^3]
            n_e = n_H * (state['xHII'] + state['xHeII'])
            return dTffdz * (3/2) / phys.dtdz(rs) * n_H / (n_H + n_He + n_e) # FIX
        
        def photoheat_rate_func1(rs):
            n_H = phys.nH * rs**3 # [1/cm^3]
            n_He = phys.nHe * rs**3 # [1/cm^3]
            n_e = n_H * (state['xHII'] + state['xHeII'])
            return dTffdz * (3/2) / phys.dtdz(rs) * n_H / (n_H + n_He + n_e) # FIX

        def photoheat_rate_func2(rs):
            n_H = phys.nH * rs**3 # [1/cm^3]
            n_He = phys.nHe * rs**3 # [1/cm^3]
            n_e = n_H * (state['xHII'] + state['xHeII'])
            return dTffdz * (3/2) / phys.dtdz(rs) * 0 / (n_H + n_He + n_e) # FIX
        
        photoheat_rate_func = [photoheat_rate_func0, photoheat_rate_func1, photoheat_rate_func2]
        
        #=== Parameters for next step ===
        highengphot_spec_at_rs.rs = next_rs
        lowengphot_spec_at_rs.rs  = next_rs
        lowengelec_spec_at_rs.rs  = next_rs

        if next_rs > end_rs:
            # Only save if next_rs < end_rs, since these are the x, Tm
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

    #===== Evolution loop ends =====

    # f_to_return = (f_low, f_high)
    
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

    f = {'low': f_low_dict, 'high': f_high_dict}

    data = {
        'rs': out_highengphot_specs.rs,
        'x': x_arr,
        'Tm': Tm_arr, 
        'highengphot': out_highengphot_specs,
        'lowengphot': out_lowengphot_specs, 
        'lowengelec': out_lowengelec_specs,
        'f': f,
        'softphot_hist': softphot_hist,
    }

    if tf_mode == 'table' and clean_up_tf:
        del dep_tf_data, ics_tf_data
        del highengphot_tf_interp, lowengphot_tf_interp, lowengelec_tf_interp, highengdep_interp
        del ics_thomson_ref_tf, ics_rel_ref_tf, engloss_ref_tf
    gc.collect()

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


def get_tf(rs, xHII, xHeII, dlnz, dep_tf_data, coarsen_factor=1):
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
        electron, high-energy deposition, and if coarsening is not 1, 
        coarsened propagating photon transfer functions. 
    """
    
    highengphot_tf_interp = dep_tf_data['highengphot']
    lowengphot_tf_interp  = dep_tf_data['lowengphot']
    lowengelec_tf_interp  = dep_tf_data['lowengelec']
    highengdep_interp     = dep_tf_data['highengdep']
    
    if coarsen_factor > 1:
        #rs_to_interpolate = np.exp(np.log(rs) - dlnz * coarsen_factor/2)
        rs_to_interpolate = rs
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
    else:
        prop_tf = None
    
    return(
        highengphot_tf, lowengphot_tf,
        lowengelec_tf, highengdep_arr, prop_tf
    )