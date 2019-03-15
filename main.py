""" Module containing the main DarkHistory functions.

"""

import numpy as np
from numpy.linalg import matrix_power
import pickle

from scipy.interpolate import interp1d

import darkhistory.physics as phys
import darkhistory.utilities as utils

import darkhistory.spec.spectools as spectools
from darkhistory.spec.spectools import EnglossRebinData
import darkhistory.spec.transferfunclist as tflist
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec.spectra import Spectra
from darkhistory.spec import transferfunction as tf
from darkhistory.spec import pppc

import darkhistory.history.histools as ht
import darkhistory.history.tla as tla

from darkhistory.electrons.ics.ics_spectrum import ics_spec
from darkhistory.electrons.ics.ics_spectrum import nonrel_spec
from darkhistory.electrons.ics.ics_spectrum import rel_spec
from darkhistory.electrons.ics.ics_engloss_spectrum import engloss_spec
from darkhistory.electrons.ics.ics_cooling import get_ics_cooling_tf
from darkhistory.electrons.elec_cooling import get_elec_cooling_tf_fast
from darkhistory.electrons.elec_cooling import \
    get_elec_cooling_tf_fast_linalg

from darkhistory.electrons import positronium as pos

from darkhistory.low_energy.lowE_deposition import compute_fs
from darkhistory.low_energy.lowE_electrons import make_interpolator

from config import data_path, photeng, eleceng
from tf_data import *

def evolve(
    in_spec_elec=None, in_spec_phot=None,
    rate_func_N=None, rate_func_eng=None, 
    DM_process=None, mDM=None, sigmav=None, lifetime=None,
    primary=None, start_rs=3000, end_rs=4,
    ics_only=False, compute_fs_method='old', highengdep_switch = True, 
    separate_higheng=False, CMB_subtracted=False, helium_TLA=False,
    reion_switch=False, reion_rs = None,
    photoion_rate_func=None, photoheat_rate_func=None, xe_reion_func=None,
    struct_boost=None,
    init_cond=None,
    coarsen_factor=1, std_soln=False, xH_func=None, xHe_func=None, user=None,
    verbose=False, use_tqdm=False
):
    """
    Main function that computes the temperature and ionization history.

    Parameters
    ----------
    in_spec_elec : Spectrum, optional
        Spectrum per annihilation/decay into electrons. rs of this spectrum is the rs of the initial conditions.
        if in_spec_elec.totN() == 0, turn off electron processes.
    in_spec_phot : Spectrum, optional
        Spectrum per annihilation/decay into photons.
    rate_func_N : function, optional
        Function describing the rate of annihilation/decay, dN/(dV dt)
    rate_func_eng : function, optional
        Function describing the rate of annihilation/decay, dE/(dV dt)
    DM_process : {'swave', 'decay'}, optional
        The dark matter process to use. 
    sigmav : float, optional
        The thermally averaged cross section, if DM_process == 'swave'. 
    lifetime : float, optional
        The decay lifetime, if DM_process == 'decay'. 
    primary : string, optional
        The primary channel of annihilation/decay. Refer to darkhistory.spec.pppc.chan_list for complete list. 
    start_rs : float, optional
        Starting redshift (1+z) to evolve from. Default is 1+z = 3000. Specify only for use with dark matter variables, otherwise initialize in_spec_elec.rs and/or in_spec_phot.rs directly.
    end_rs : float, optional
        Final redshift (1+z) to evolve to. Default is 1+z = 4. 
    reion_switch : bool
        Reionization model included if true.
    ics_only : bool, optional
        If True, turns off atomic cooling for input electrons.
    compute_fs_method : {'old', 'helium'}
        The method to compute f's. 'helium' includes helium photoionization.
    highengdep_switch: bool, optional
        If False, turns off high energy deposition estimate.
    separate_higheng : bool, optional
        If True, reports the high and low f(z) separately.
    CMB_subtracted : bool
        ???
    helium_TLA : bool
        If True, the TLA is solved with helium.
    reion_rs : float, optional
        Redshift 1+z at which reionization effects turn on.
    photoion_rate_func : tuple of functions, optional
        Functions take redshift 1+z as input, return the photoionization rate in s^-1 of HI, HeI and HeII respectively. If not specified, defaults to `darkhistory.history.reionization.photoion_rate`.
    photoheat_rate_func : tuple of functions, optional
        Functions take redshift 1+z as input, return the photoheating rate in s^-1 of HI, HeI and HeII respectively. If not specified, defaults to `darkhistory.history.reionization.photoheat_rate`.
    xe_reion_func : function, optional
        Specifies a fixed ionization history after reion_rs.
    struct_boost : function, optional
        Energy injection boost factor due to structure formation
    init_cond : tuple of floats
        Specifies the initial (xH, xHe, Tm). Defaults to RECFAST if None.
    coarsen_factor : int
        Coarsening to apply to the transfer function matrix.
    std_soln : bool
        If true, uses the standard TLA solution for f(z).
    xH_func : function, optional
        If provided, fixes xH to the output of this function (which takes redshift as its sole argument). Superceded by xe_reion_func past reion_rs. std_soln must be True.
    xHe_func : function, optional
        If provided, fixes xHe to the output of this function (which takes
        redshift as its sole argument). Superceded by xe_reion_func past
        reion_rs. std_soln must be True.
    user : str
        specify which user is accessing the code, so that the standard solution can be downloaded.  Must be changed!!!
    use_tqdm : bool, optional
        Uses tqdm if true.
    """

    ################################
    # Initialization
    ################################

    # Handle the case where a DM process is specified. 
    if DM_process == 'swave':
        if sigmav is None or primary is None:
            raise InputError('both sigmav and primary must be specified.')
        # Get input spectra from PPPC. 
        # in_spec_elec = pppc.get_pppc_spec(mDM, eleceng, primary, 'elec')
        # in_spec_phot = pppc.get_pppc_spec(mDM, photeng, primary, 'phot')
        # Define the rate functions. 
        def rate_func_N(rs):
            return (
                phys.inj_rate('swave', rs, mDM=mDM, sigmav=sigmav)
                * struct_boost(rs) / (2*mDM)
            )
        def rate_func_eng(rs):
            return (
                phys.inj_rate('swave', rs, mDM=mDM, sigmav=sigmav) 
                * struct_boost(rs)
            )

    if DM_process == 'decay':
        if lifetime is None or primary is None:
            raise InputError('both lifetime and primary must be specified.')
        # Get spectra from PPPC.
        in_spec_elec = pppc.get_pppc_spec(
            mDM, eleceng, primary, 'elec', decay=True
        )
        in_spec_phot = pppc.get_pppc_spec(
            mDM, photeng, primary, 'phot', decay=True
        )
        # Define the rate functions. 
        def rate_func_N(rs):
            return (
                phys.inj_rate('decay', rs, mDM=mDM, lifetime=lifetime) / mDM
            )
        def rate_func_eng(rs):
            return phys.inj_rate('swave', rs, mDM=mDM, lifetime=lifetime) 
                
    
    # Electron and Photon abscissae should be the default abscissae. 
    if in_spec_elec.eng != eleceng or in_spec_elec.eng != photeng:
        raise InputError('in_spec_elec and in_spec_phot must use config.photeng and config.eleceng respectively as abscissa.')

    # Initialize the next spectrum as None.
    next_highengphot_spec = None
    next_lowengphot_spec  = None
    next_lowengelec_spec  = None

    if (
        highengphot_tf_interp.dlnz    != lowengphot_tf_interp.dlnz
        or highengphot_tf_interp.dlnz != lowengelec_tf_interp.dlnz
        or lowengphot_tf_interp.dlnz  != lowengelec_tf_interp.dlnz
    ):
        raise TypeError('TransferFuncInterp objects must all have the same dlnz.')

    if in_spec_elec.rs != in_spec_phot.rs:
        raise TypeError('Input spectra must have the same rs.')

    start_rs = in_spec_elec.rs

    if CMB_subtracted and np.any(lowengphot_tf_interp._log_interp):
        raise TypeError('Cannot log interp over negative numbers')

    # Load the standard TLA and standard initializations.
    xH_std  = phys.xH_std
    xHe_std = phys.xHe_std
    Tm_std  = phys.Tm_std

    xH_init_std  = phys.xH_std(start_rs)
    xHe_init_std = phys.xHe_std(start_rs)
    Tm_init_std  = phys.Tm_std(start_rs) 

    # Initialize if not specified for std_soln.
    if std_soln:
        xH_init  = xH_init_std
        xHe_init = xHe_init_std
        Tm_init  = Tm_init_std

    # Initialize to std_soln if unspecified.
    if init_cond is None:
        xH_init  = xH_init_std
        xHe_init = xHe_init_std
        Tm_init  = Tm_init_std 
    else:
        xH_init  = init_cond[0]
        xHe_init = init_cond[1]
        Tm_init  = init_cond[2]

    if not std_soln and (xH_func is not None or xHe_func is not None):
        raise TypeError(
            'std_soln must be True if xH_func or xHe_func is specified.'
        )

    # If functions are specified, initialize according to the functions.
    # xH_std and xHe_std are reassigned to the functions, if they exist.
    if xH_func is not None:
        xH_std = xH_func
        xH_init = xH_std(in_spec_phot.rs)
    if xHe_func is not None:
        xHe_std  = xHe_func
        xHe_init = xHe_std(in_spec_phot.rs)


    x_arr  = np.array([[xH_init, xHe_init]])
    Tm_arr = np.array([Tm_init])

    # Redshift/timestep related quantities.
    dlnz = highengphot_tf_interp.dlnz[-1]
    prev_rs = None
    rs = in_spec_phot.rs
    dt = dlnz * coarsen_factor / phys.hubble(rs)

    # tqdm related stuff.
    if use_tqdm:
        from tqdm import tqdm_notebook as tqdm
        pbar = tqdm(
            total=np.ceil((np.log(rs) - np.log(end_rs))/dlnz/coarsen_factor)
        )

    ################################
    # Subroutines
    ################################

    # Function that changes the normalization
    # from per annihilation to per baryon in the step.
    # rate_func_N converts from per annihilation per volume per time,
    # other factors do the rest of the conversion.
    def norm_fac(rs):
        return rate_func_N(rs) * (
            dlnz * coarsen_factor / phys.hubble(rs) / (phys.nB * rs**3)
        )

    # If in_spec_elec is empty, turn off electron processes.
    elec_processes = False
    if in_spec_elec.totN() > 0:
        elec_processes = True

        if (
            ics_thomson_ref_tf is None or ics_rel_ref_tf is None
            or engloss_ref_tf is None
        ):
            raise TypeError('Must specify transfer functions for electron processes')

    if elec_processes:
        if ics_only:
            (
                ics_sec_phot_tf, elec_processes_lowengelec_tf,
                continuum_loss, deposited_ICS_arr
            ) = get_ics_cooling_tf(
                    ics_thomson_ref_tf, ics_rel_ref_tf, engloss_ref_tf,
                    eleceng, photeng, rs, fast=True
                )
        else:
            # Compute the (normalized) collisional ionization spectra.
            coll_ion_sec_elec_specs = (
                phys.coll_ion_sec_elec_spec(eleceng, eleceng, species='HI'),
                phys.coll_ion_sec_elec_spec(eleceng, eleceng, species='HeI'),
                phys.coll_ion_sec_elec_spec(eleceng, eleceng, species='HeII')
            )
            # Compute the (normalized) collisional excitation spectra.
            id_mat = np.identity(eleceng.size)

            coll_exc_sec_elec_tf_HI = tf.TransFuncAtRedshift(
                np.squeeze(id_mat[:, np.where(eleceng > phys.lya_eng)]),
                in_eng = eleceng, rs = rs*np.ones_like(eleceng),
                eng = eleceng[eleceng > phys.lya_eng] - phys.lya_eng,
                dlnz = -1, spec_type = 'N'
            )

            coll_exc_sec_elec_tf_HeI = tf.TransFuncAtRedshift(
                np.squeeze(
                    id_mat[:, np.where(eleceng > phys.He_exc_eng['23s'])]
                ),
                in_eng = eleceng, rs = rs*np.ones_like(eleceng),
                eng = (
                    eleceng[eleceng > phys.He_exc_eng['23s']] 
                    - phys.He_exc_eng['23s']
                ), 
                dlnz = -1, spec_type = 'N'
            )

            coll_exc_sec_elec_tf_HeII = tf.TransFuncAtRedshift(
                np.squeeze(id_mat[:, np.where(eleceng > 4*phys.lya_eng)]),
                in_eng = eleceng, rs = rs*np.ones_like(eleceng),
                eng = eleceng[eleceng > 4*phys.lya_eng] - 4*phys.lya_eng,
                dlnz = -1, spec_type = 'N'
            )

            coll_exc_sec_elec_tf_HI.rebin(eleceng)
            coll_exc_sec_elec_tf_HeI.rebin(eleceng)
            coll_exc_sec_elec_tf_HeII.rebin(eleceng)

            coll_exc_sec_elec_specs = (
                coll_exc_sec_elec_tf_HI.grid_vals,
                coll_exc_sec_elec_tf_HeI.grid_vals,
                coll_exc_sec_elec_tf_HeII.grid_vals
            )

            # Store the ICS rebinning data for speed.
            ics_engloss_data = EnglossRebinData(eleceng, photeng, eleceng)

            # REMEMBER TO CHANGE xHe WHEN USING THE CORRECT PRESCRIPTION!!
            (
                ics_sec_phot_tf, elec_processes_lowengelec_tf,
                deposited_ion_arr, deposited_exc_arr, deposited_heat_arr,
                continuum_loss, deposited_ICS_arr
            ) = get_elec_cooling_tf_fast(
                    ics_thomson_ref_tf, ics_rel_ref_tf, engloss_ref_tf,
                    coll_ion_sec_elec_specs, coll_exc_sec_elec_specs,
                    eleceng, photeng, rs,
                    x_arr[-1,0], xHe=x_arr[-1,1],
                    linalg=True, ics_engloss_data=ics_engloss_data
                )

        # Quantities are still per annihilation.
        ics_phot_spec = ics_sec_phot_tf.sum_specs(in_spec_elec)

        elec_processes_lowengelec_spec = (
            elec_processes_lowengelec_tf.sum_specs(in_spec_elec)
        )

        if not ics_only:
            deposited_ion  = np.dot(
                deposited_ion_arr,  in_spec_elec.N*norm_fac(rs)
            )
            deposited_exc  = np.dot(
                deposited_exc_arr,  in_spec_elec.N*norm_fac(rs)
            )
            deposited_heat = np.dot(
                deposited_heat_arr, in_spec_elec.N*norm_fac(rs)
            )

        else:

            deposited_ion  = 0.
            deposited_exc  = 0.
            deposited_heat = 0.

        deposited_ICS  = np.dot(
            deposited_ICS_arr,  in_spec_elec.N*norm_fac(rs)
        )

        positronium_phot_spec = pos.weighted_photon_spec(photeng) * (
            in_spec_elec.totN()/2
        )
        if positronium_phot_spec.spec_type != 'N':
            positronium_phot_spec.switch_spec_type()

        positronium_phot_spec.rs = rs

        # The initial input dN/dE per annihilation to per baryon per dlnz,
        # based on the specified rate.
        # dN/(dN_B d lnz dE) = dN/dE * (dN_ann/(dV dt)) * dV/dN_B * dt/dlogz
        init_inj_spec = (
            (in_spec_phot + ics_phot_spec + positronium_phot_spec)
            * norm_fac(rs)
        )

    else:
        init_inj_spec = in_spec_phot * norm_fac(rs)

    # Initialize the Spectra object that will contain all the
    # output spectra during the evolution.
    out_highengphot_specs = Spectra(
        [init_inj_spec], spec_type=init_inj_spec.spec_type
    )
    out_lowengphot_specs  = Spectra(
        [in_spec_phot*0], spec_type=in_spec_phot.spec_type
    )
    if elec_processes:
        out_lowengelec_specs  = Spectra(
            [elec_processes_lowengelec_spec*norm_fac(rs)],
            spec_type=init_inj_spec.spec_type
        )
    else:
        out_lowengelec_specs = Spectra(
            [in_spec_elec*0], spec_type=init_inj_spec.spec_type
        )

    if separate_higheng:
        f_low = np.zeros((1,5))
        f_high = np.zeros((1,5))
    else:
        f_arr = np.zeros((1,5))

    # Define these methods for speed.
    append_highengphot_spec = out_highengphot_specs.append
    append_lowengphot_spec  = out_lowengphot_specs.append
    append_lowengelec_spec  = out_lowengelec_specs.append
    #print('starting...\n')

    rate_func_eng_unclustered = rate_func_eng
    cmbloss_grid = np.zeros(1)
    highengdep_grid = np.zeros((1,4))

    MEDEA_interp = make_interpolator()

    if not highengdep_switch:
        highengdep_fac = 0
    else:
        highengdep_fac = 1

    if elec_processes:
        # Add energy deposited in atomic processes. Rescale to
        # energy per baryon per unit time.
        highengdep_grid += np.array([[
            deposited_ion/dt,
            deposited_exc/dt,
            deposited_heat/dt,
            deposited_ICS/dt
        ]])
        cmbloss_grid += np.array([
            np.dot(continuum_loss/dt, in_spec_elec.N*norm_fac(rs))
        ])

        if std_soln:
            f_raw = compute_fs(
                MEDEA_interp, out_lowengelec_specs[0],
                out_lowengphot_specs[0],
                np.array([
                    1.-xH_std(rs), 
                    phys.chi-xHe_std(rs), xHe_std(rs)
                ]),
                rate_func_eng_unclustered(rs), dt,
                highengdep_fac*highengdep_grid[-1], cmbloss_grid[-1],
                separate_higheng=separate_higheng, method=compute_fs_method
            )
        else:
            f_raw = compute_fs(
                MEDEA_interp, out_lowengelec_specs[0],
                out_lowengphot_specs[0],
                np.array([
                    1.-x_arr[-1,0], 
                    phys.chi-x_arr[-1,1], x_arr[-1,1]
                ]),
                rate_func_eng_unclustered(rs), dt,
                highengdep_fac*highengdep_grid[-1], cmbloss_grid[-1],
                separate_higheng=separate_higheng, method=compute_fs_method
            )

        if separate_higheng:
            f_low[0]  = f_raw[0]
            f_high[0] = f_raw[1]

        else:
            f_arr[0] = f_raw


    ######################################################
    # Loop while we are still at a redshift above end_rs.
    ######################################################


    while rs > end_rs:

        if use_tqdm:
            pbar.update(1)

        # dE/dVdt_inj without structure formation
        # should be passed into compute_fs
        if struct_boost is not None:
            if struct_boost(rs) == 1:
                rate_func_eng_unclustered = rate_func_eng
            else:
                def rate_func_eng_unclustered(rs):
                    return rate_func_eng(rs)/struct_boost(rs)

        # If prev_rs exists, calculate xe and T_m.
        if prev_rs is not None:
            # f_H_ion, f_He_ion, f_exc, f_heat, f_continuum

            if std_soln:
                f_raw = compute_fs(
                    MEDEA_interp, next_lowengelec_spec, next_lowengphot_spec,
                    np.array([
                        1.-xH_std(rs), 
                        phys.chi-xHe_std(rs), xHe_std(rs)
                    ]),
                    rate_func_eng_unclustered(rs), dt,
                    highengdep_fac*highengdep_grid[-1], cmbloss_grid[-1],
                    separate_higheng=separate_higheng, 
                    method=compute_fs_method
                )
            else:
                f_raw = compute_fs(
                    MEDEA_interp, next_lowengelec_spec, next_lowengphot_spec,
                    np.array([
                        1.-x_arr[-1,0], 
                        phys.chi-x_arr[-1,1], x_arr[-1,1]
                    ]),
                    rate_func_eng_unclustered(rs), dt,
                    highengdep_fac*highengdep_grid[-1], cmbloss_grid[-1],
                    separate_higheng=separate_higheng, 
                    method=compute_fs_method
                )

            if separate_higheng:
                f_low  = np.append(f_low, [f_raw[0]], axis=0)
                f_high = np.append(f_high, [f_raw[1]], axis=0)

                # Compute the f's for the TLA: sum low and high.
                f_H_ion  = f_raw[0][0] + f_raw[1][0]
                if compute_fs_method == 'old':
                    f_He_ion = 0.
                else:
                    f_He_ion = f_raw[0][1] + f_raw[1][1]
                f_exc    = f_raw[0][2] + f_raw[1][2]
                f_heat   = f_raw[0][3] + f_raw[1][3]
            else:
                f_arr = np.append(f_arr, [f_raw], axis=0)
                # Compute the f's for the TLA.
                f_H_ion  = f_raw[0]
                if compute_fs_method == 'old':
                    f_He_ion = 0.
                else:
                    f_He_ion = f_raw[1]
                f_exc    = f_raw[2]
                f_heat   = f_raw[3]

            init_cond_new = np.array(
                [Tm_arr[-1], x_arr[-1,0], x_arr[-1,1], 0]
            )

            new_vals = tla.get_history(
                np.array([prev_rs, rs]), init_cond=init_cond_new, 
                f_H_ion=f_H_ion, f_H_exc=f_exc, f_heating=f_heat,
                dm_injection_rate=rate_func_eng_unclustered,
                reion_switch=reion_switch, reion_rs=reion_rs,
                photoion_rate_func=photoion_rate_func,
                photoheat_rate_func=photoheat_rate_func,
                xe_reion_func=xe_reion_func, helium_TLA=helium_TLA,
                f_He_ion=f_He_ion
            )

            Tm_arr = np.append(Tm_arr, new_vals[-1,0])

            if helium_TLA:
                # Append the output of xHe to the array.
                x_arr  = np.append(
                    x_arr,  [[new_vals[-1,1], new_vals[-1,2]]], axis=0
                )
            else:
                # Append the standard solution value. 
                x_arr  = np.append(
                    x_arr,  [[ new_vals[-1,1], xHe_std(rs) ]], axis=0
                )

        #print('x_e at '+str(rs)+': '+ str(xe_arr[-1]))
        #print('Standard x_e at '+str(rs)+': '+str(xe_std(rs)))
        #print('T_m at '+str(rs)+': '+ str(Tm_arr[-1]))
        #print('Standard T_m at '+str(rs)+': '+str(Tm_std(rs)))
        #if prev_rs is not None:
        #    print('Back Reaction f_ionH, f_ionHe, f_exc, f_heat, f_cont: ', f_raw)

        # if std_soln:
        #     highengphot_tf = highengphot_tf_interp.get_tf(xH_std(rs), rs)
        #     lowengphot_tf  = lowengphot_tf_interp.get_tf(xH_std(rs), rs)
        #     lowengelec_tf  = lowengelec_tf_interp.get_tf(xH_std(rs), rs)
        #     cmbloss_arr    = CMB_engloss_interp.get_val(xH_std(rs), rs)
        #     highengdep_arr = highengdep_interp.get_val(xH_std(rs), rs)
        # else:
        #     highengphot_tf = highengphot_tf_interp.get_tf(x_arr[-1,0], rs)
        #     lowengphot_tf  = lowengphot_tf_interp.get_tf(x_arr[-1,0], rs)
        #     lowengelec_tf  = lowengelec_tf_interp.get_tf(x_arr[-1,0], rs)
        #     cmbloss_arr    = CMB_engloss_interp.get_val(x_arr[-1,0], rs)
        #     highengdep_arr = highengdep_interp.get_val(x_arr[-1,0], rs)

        if std_soln:
            highengphot_tf = highengphot_tf_interp.get_tf(
                xH_std(rs), xHe_std(rs), rs
            )
            lowengphot_tf  = lowengphot_tf_interp.get_tf(
                xH_std(rs), xHe_std(rs), rs
            )
            lowengelec_tf  = lowengelec_tf_interp.get_tf(
                xH_std(rs), xHe_std(rs), rs
            )
            cmbloss_arr    = CMB_engloss_interp.get_val(
                xH_std(rs), xHe_std(rs), rs
            )
            highengdep_arr = highengdep_interp.get_val(
                xH_std(rs), xHe_std(rs), rs)
        else:
            highengphot_tf = highengphot_tf_interp.get_tf(
                x_arr[-1,0], x_arr[-1,1], rs
            )
            lowengphot_tf  = lowengphot_tf_interp.get_tf(
                x_arr[-1,0], x_arr[-1,1], rs
            )
            lowengelec_tf  = lowengelec_tf_interp.get_tf(
                x_arr[-1,0], x_arr[-1,1], rs
            )
            cmbloss_arr    = CMB_engloss_interp.get_val(
                x_arr[-1,0], x_arr[-1,1], rs
            )
            highengdep_arr = highengdep_interp.get_val(
                x_arr[-1,0], x_arr[-1,1], rs
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
            cmbloss_arr = np.matmul(prop_tf, cmbloss_arr)/coarsen_factor
            highengdep_arr = (
                np.matmul(prop_tf, highengdep_arr)/coarsen_factor
            )

        cmbloss = np.dot(cmbloss_arr, out_highengphot_specs[-1].N)
        if CMB_subtracted:
            cmbloss = 0
        highengdep = np.dot(
            np.swapaxes(highengdep_arr, 0, 1),
            out_highengphot_specs[-1].N
        )

        next_highengphot_spec = highengphot_tf.sum_specs(
            out_highengphot_specs[-1]
        )
        next_lowengphot_spec  = lowengphot_tf.sum_specs(
            out_highengphot_specs[-1]
        )
        next_lowengelec_spec  = lowengelec_tf.sum_specs(
            out_highengphot_specs[-1]
        )

        # Re-define existing variables.
        prev_rs = rs
        rs = np.exp(np.log(rs) - dlnz * coarsen_factor)

        dt = dlnz * coarsen_factor/phys.hubble(rs)
        next_highengphot_spec.rs = rs
        next_lowengphot_spec.rs  = rs
        next_lowengelec_spec.rs  = rs

        # Add the next injection spectrum to next_highengphot_spec
        if elec_processes:
            if ics_only:
                (
                    ics_sec_phot_tf, elec_processes_lowengelec_tf,
                    continuum_loss, deposited_ICS_arr
                ) = get_ics_cooling_tf(
                        ics_thomson_ref_tf, ics_rel_ref_tf, engloss_ref_tf,
                        eleceng, photeng, rs, fast=True
                    )
            else:
                if std_soln:
                    xH_elec_cooling = xH_std(rs)
                else:
                    xH_elec_cooling = x_arr[-1,0]

                if std_soln:
                    xHe_elec_cooling = xHe_std(rs)
                else:
                    xHe_elec_cooling = x_arr[-1, 1]
                # NOTE TO GREG: ics_sec_phot_tf -= continuum_loss in the correct treatment
                # for the Tracy-consistent treatment, subtract dE/dt * 1/V / dE/dV/dt from f_cont, where dE/dt is derived from continuum loss
                (
                    ics_sec_phot_tf, elec_processes_lowengelec_tf,
                    deposited_ion_arr, deposited_exc_arr, deposited_heat_arr,
                    continuum_loss, deposited_ICS_arr
                ) = get_elec_cooling_tf_fast(
                        ics_thomson_ref_tf, ics_rel_ref_tf, engloss_ref_tf,
                        coll_ion_sec_elec_specs, coll_exc_sec_elec_specs,
                        eleceng, photeng, rs, 
                        xH_elec_cooling, xHe=xHe_elec_cooling,
                        linalg=True, ics_engloss_data=ics_engloss_data
                    )

            ics_phot_spec = ics_sec_phot_tf.sum_specs(in_spec_elec)
            elec_processes_lowengelec_spec = (
                elec_processes_lowengelec_tf.sum_specs(in_spec_elec)
            )

            if not ics_only:

                deposited_ion  = np.dot(
                    deposited_ion_arr,  in_spec_elec.N*norm_fac(rs)
                )
                deposited_exc  = np.dot(
                    deposited_exc_arr,  in_spec_elec.N*norm_fac(rs)
                )
                deposited_heat = np.dot(
                    deposited_heat_arr, in_spec_elec.N*norm_fac(rs)
                )

            else:

                deposited_ion  = 0
                deposited_exc  = 0
                deposited_heat = 0

            deposited_ICS  = np.dot(
                deposited_ICS_arr,  in_spec_elec.N*norm_fac(rs)
            )

            # Add energy deposited in atomic processes. Rescale to
            # energy per baryon per unit time.
            highengdep += np.array([
                deposited_ion/dt,
                deposited_exc/dt,
                deposited_heat/dt,
                deposited_ICS/dt
            ])

            cmbloss += np.dot(
                continuum_loss/dt, in_spec_elec.N*norm_fac(rs)
            )

            next_inj_phot_spec = (
                (in_spec_phot + ics_phot_spec + positronium_phot_spec)
                *norm_fac(rs)
            )
            # Add prompt low-energy electrons for the next step.
            next_lowengelec_spec += (
                elec_processes_lowengelec_spec*norm_fac(rs)
            )

        else:
            next_inj_phot_spec = in_spec_phot * norm_fac(rs)

        # This keeps the redshift.
        next_highengphot_spec.N += next_inj_phot_spec.N

        append_highengphot_spec(next_highengphot_spec)
        append_lowengphot_spec(next_lowengphot_spec)
        append_lowengelec_spec(next_lowengelec_spec)
        cmbloss_grid = np.append(cmbloss_grid, cmbloss)
        highengdep_grid = np.concatenate(
            (highengdep_grid, np.array([highengdep]))
        )

        if verbose:
            print("completed rs: ", prev_rs)

    if use_tqdm:
        pbar.close()

    if separate_higheng:
        f_to_return = (f_low, f_high)
    else:
        f_to_return = f_arr

    return (
        x_arr, Tm_arr,
        out_highengphot_specs, out_lowengphot_specs, out_lowengelec_specs,
        cmbloss_grid, f_to_return
    )
