""" Module containing the main DarkHistory functions.

"""

import numpy as np
from numpy.linalg import matrix_power
import pickle

from scipy.interpolate import interp1d

import darkhistory.physics as phys
import darkhistory.utilities as utils
import darkhistory.spec.spectools as spectools
import darkhistory.spec.transferfunclist as tflist
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec.spectra import Spectra
import darkhistory.history.histools as ht
import darkhistory.history.tla as tla

from darkhistory.electrons.ics.ics_spectrum import ics_spec
from darkhistory.electrons.ics.ics_spectrum import nonrel_spec
from darkhistory.electrons.ics.ics_spectrum import rel_spec
from darkhistory.electrons.ics.ics_engloss_spectrum import engloss_spec
from darkhistory.electrons.ics.ics_cooling import get_ics_cooling_tf
from darkhistory.electrons.elec_cooling import get_elec_cooling_tf_fast

from darkhistory.electrons import positronium as pos

from darkhistory.low_energy.lowE_deposition import compute_fs
from darkhistory.low_energy.lowE_electrons import make_interpolator

import os
cwd = os.getcwd()
abspath = os.path.abspath(__file__)
dir_path = os.path.dirname(abspath)

def load_trans_funcs(direc):
    # Load in the transferfunctions
    #!!! Should be a directory internal to DarkHistory
    print('Loading transfer functions...')
    highengphot_tflist_arr = pickle.load(open(direc+"tfunclist_photspec_60eV_complete.raw", "rb"))
    #highengphot_tflist_arr = pickle.load(open(direc+"tfunclist_photspec_60eV_injE_complete_rs_30_xe_2pts.raw", "rb"))
    print('Loaded high energy photons...')

    lowengphot_tflist_arr  = pickle.load(open(direc+"tfunclist_lowengphotspec_60eV_complete.raw", "rb"))
    #lowengphot_tflist_arr  = pickle.load(open(direc+"tfunclist_lowengphotspec_60eV_injE_complete_rs_30_xe_2pts.raw", "rb"))
    print('Low energy photons...')

    lowengelec_tflist_arr  = pickle.load(open(direc+"tfunclist_lowengelecspec_60eV_complete.raw", "rb"))
    #lowengelec_tflist_arr  = pickle.load(open(direc+"tfunclist_lowengelecspec_60eV_injE_complete_rs_30_xe_2pts.raw", "rb"))
    print('Low energy electrons...')

    highengdep_arr = pickle.load(open(direc+"highdeposited_60eV_complete.raw", "rb"))
    #highengdep_arr = pickle.load(open(direc+"highdeposited_60eV_injE_complete_rs_30_xe_2pts.raw", "rb"))
    highengdep_arr = np.swapaxes(highengdep_arr, 1, 2)
    print('high energy deposition.\n')

    CMB_engloss_arr = pickle.load(open(direc+"CMB_engloss_60eV_complete.raw", "rb"))
    #CMB_engloss_arr = pickle.load(open(direc+"CMB_engloss_60eV_injE_complete_rs_30_xe_2pts.raw", "rb"))
    CMB_engloss_arr = np.swapaxes(CMB_engloss_arr, 1, 2)
    print('CMB losses.\n')

    xes = 0.5 + 0.5*np.tanh([-5., -4.1, -3.2, -2.3, -1.4, -0.5, 0.4, 1.3, 2.2, 3.1, 4])
    #xes = 0.5 + 0.5*np.tanh([-5., -4.1])
    photeng = highengphot_tflist_arr[0].eng
    eleceng = lowengelec_tflist_arr[0].eng
    rs_list = highengphot_tflist_arr[0].rs

    #Split photeng into high and low energy.
    photeng_high = photeng[photeng > 60]
    photeng_low  = photeng[photeng <= 60]

    # Split eleceng into high and low energy.
    eleceng_high = eleceng[eleceng > 3000]
    eleceng_low  = eleceng[eleceng <= 3000]


    print('Padding tflists with zeros...')
    for highengphot_tflist in highengphot_tflist_arr:
        for tf in highengphot_tflist:
            # Pad with zeros so that it becomes photeng x photeng.
            tf._grid_vals = np.pad(tf.grid_vals, ((photeng_low.size, 0), (0, 0)), 'constant')
            tf._N_underflow = np.pad(tf._N_underflow, (photeng_low.size, 0), 'constant')
            tf._eng_underflow = np.pad(tf._eng_underflow, (photeng_low.size, 0), 'constant')
            tf._in_eng = photeng
            tf._eng = photeng
            tf._rs = tf.rs[0]*np.ones_like(photeng)

        highengphot_tflist._eng = photeng
        highengphot_tflist._in_eng = photeng
        highengphot_tflist._grid_vals = np.atleast_3d(
            np.stack([tf.grid_vals for tf in highengphot_tflist._tflist])
        )
    print("high energy photons...")

    # lowengphot_tflist.in_eng set to photeng_high
    for lowengphot_tflist in lowengphot_tflist_arr:
        for tf in lowengphot_tflist:
            # Pad with zeros so that it becomes photeng x photeng.
            tf._grid_vals = np.pad(tf.grid_vals, ((photeng_low.size,0), (0,0)), 'constant')
            # Photons in the low energy bins should be immediately deposited.
            tf._grid_vals[0:photeng_low.size, 0:photeng_low.size] = np.identity(photeng_low.size)
            tf._N_underflow = np.pad(tf._N_underflow, (photeng_low.size, 0), 'constant')
            tf._eng_underflow = np.pad(tf._eng_underflow, (photeng_low.size, 0), 'constant')
            tf._in_eng = photeng
            tf._eng = photeng
            tf._rs = tf.rs[0]*np.ones_like(photeng)

        lowengphot_tflist._eng = photeng
        lowengphot_tflist._in_eng = photeng
        lowengphot_tflist._grid_vals = np.atleast_3d(
            np.stack([tf.grid_vals for tf in lowengphot_tflist._tflist])
        )
    print("low energy photons...")

    # lowengelec_tflist.in_eng set to photeng_high
    for lowengelec_tflist in lowengelec_tflist_arr:
        for tf in lowengelec_tflist:
            # Pad with zeros so that it becomes photeng x eleceng.
            tf._grid_vals = np.pad(tf.grid_vals, ((photeng_low.size,0), (0,0)), 'constant')
            tf._N_underflow = np.pad(tf._N_underflow, (photeng_low.size, 0), 'constant')
            tf._eng_underflow = np.pad(tf._eng_underflow, (photeng_low.size, 0), 'constant')
            tf._in_eng = photeng
            tf._eng = eleceng
            tf._rs = tf.rs[0]*np.ones_like(photeng)

        lowengelec_tflist._eng = eleceng
        lowengelec_tflist._in_eng = photeng
        lowengelec_tflist._grid_vals = np.atleast_3d(
            np.stack([tf.grid_vals for tf in lowengelec_tflist._tflist])
        )
    print("low energy electrons...\n")

    tmp = np.zeros((len(xes),len(rs_list),len(photeng), 4))
    for i, highdep in enumerate(highengdep_arr):
        tmp[i] = np.pad(highdep, ((0,0),(photeng_low.size, 0),(0,0)), 'constant')
    highengdep_arr = tmp.copy()
    print("high energy deposition.\n")

    tmp = np.zeros((len(xes),len(rs_list),len(photeng)))
    for i, engloss in enumerate(CMB_engloss_arr):
        tmp[i] = np.pad(engloss, ((0,0),(photeng_low.size, 0)), 'constant')
    CMB_engloss_arr = tmp.copy()
    print("CMB losses.\n")

    print("Generating TransferFuncInterp objects for each tflist...")
    # interpolate over xe
    highengphot_tf_interp = tflist.TransferFuncInterp(xes, highengphot_tflist_arr, log_interp = False)
    lowengphot_tf_interp  = tflist.TransferFuncInterp(xes, lowengphot_tflist_arr, log_interp = False)
    lowengelec_tf_interp  = tflist.TransferFuncInterp(xes, lowengelec_tflist_arr, log_interp = False)
    highengdep_interp     = ht.IonRSInterp(xes, rs_list, highengdep_arr, in_eng = photeng, logInterp=True)
    CMB_engloss_interp    = ht.IonRSInterp(xes, rs_list, CMB_engloss_arr, in_eng = photeng, logInterp=True)
    print("Done.\n")

    return highengphot_tf_interp, lowengphot_tf_interp, lowengelec_tf_interp, highengdep_interp, CMB_engloss_interp

def load_ics_data():
    Emax = 1e20
    Emin = 1e-8
    nEe = 5000
    nEp  = 5000

    dlnEp = np.log(Emax/Emin)/nEp
    lowengEp_rel = Emin*np.exp((np.arange(nEp)+0.5)*dlnEp)

    dlnEe = np.log(Emax/Emin)/nEe
    lowengEe_rel = Emin*np.exp((np.arange(nEe)+0.5)*dlnEe)

    Emax = 1e10
    Emin = 1e-8
    nEe = 5000
    nEp  = 5000

    dlnEp = np.log(Emax/Emin)/nEp
    lowengEp_nonrel = Emin*np.exp((np.arange(nEp)+0.5)*dlnEp)

    dlnEe = np.log(Emax/Emin)/nEe
    lowengEe_nonrel = Emin*np.exp((np.arange(nEe)+0.5)*dlnEe)

    print('********* Thomson regime scattered photon spectrum *********')
    ics_thomson_ref_tf = nonrel_spec(lowengEe_nonrel, lowengEp_nonrel, phys.TCMB(400))
    print('********* Relativistic regime scattered photon spectrum *********')
    ics_rel_ref_tf = rel_spec(lowengEe_rel, lowengEp_rel, phys.TCMB(400), inf_upp_bound=True)
    print('********* Thomson regime energy loss spectrum *********')
    engloss_ref_tf = engloss_spec(lowengEe_nonrel, lowengEp_nonrel, phys.TCMB(400), nonrel=True)
    return ics_thomson_ref_tf, ics_rel_ref_tf, engloss_ref_tf

def load_std(xe_init, Tm_init, rs):
    """
    Load the std free electron fraction (xe) and matter temperature (Tm) histories.
    If xe_init and/or Tm_init aren't initialized, set them to their standard values.
    """
    os.chdir(dir_path)
    soln = pickle.load(open("darkhistory/history/std_soln.p", "rb"))
    xe_std  = interp1d(soln[0,:], soln[2,:])
    Tm_std = interp1d(soln[0,:], soln[1,:])
    # soln = np.loadtxt(open("darkhistory/history/recfast_standard.txt", "rb"))
    # xe_std = interp1d(soln[:,0], soln[:,2])
    # Tm_std = interp1d(soln[:,0], soln[:,1])
    os.chdir(cwd)
    #def xe_std(rs):
    #    return 0.00027458
    if xe_init is None:
        xe_init = xe_std(rs)
    if Tm_init is None:
        Tm_init = Tm_std(rs)
    return xe_std, Tm_std, xe_init, Tm_init

def evolve(
    in_spec_elec, in_spec_phot,
    rate_func_N, rate_func_eng, end_rs,
    highengphot_tf_interp, lowengphot_tf_interp, lowengelec_tf_interp,
    highengdep_interp, CMB_engloss_interp,
    ics_thomson_ref_tf=None, ics_rel_ref_tf=None, engloss_ref_tf=None,
    ics_only=False, highengdep_switch = True, separate_higheng=False,
    reion_switch=False, reion_rs = None, 
    photoion_rate_func=None, photoheat_rate_func=None, xe_reion_func=None,
    struct_boost=None,
    xe_init=None, Tm_init=None,
    coarsen_factor=1, std_soln=False, xe_func=None, user=None, 
    verbose=False, use_tqdm=False
):
    """
    Main function that computes the temperature and ionization history.

    Parameters
    ----------
    in_spec_elec : Spectrum
        Spectrum per annihilation/decay into electrons. rs of this spectrum is the rs of the initial conditions.
        if in_spec_elec.totN() == 0, turn off electron processes.
    in_spec_phot : Spectrum
        Spectrum per annihilation/decay into photons.
    rate_func_N : function
        Function describing the rate of annihilation/decay, dN/(dV dt)
    rate_func_eng : function
        Function describing the rate of annihilation/decay, dE/(dV dt)
    end_rs : float
        Final redshift to evolve to.
    reion_switch : bool
        Reionization model included if true.
    highengphot_tf_interp : TransFuncInterp
        high energy photon transfer function interpolation object.
    lowengphot_tf_interp : TransFuncInterp
        low energy photon transfer function interpolation object.
    lowengelec_tf_interp : TransFuncInterp
        low energy electron transfer function interpolation object.
    highengdep_interp : IonRSInterp
        energy deposition from high energy particles, interpolation object
    CMB_engloss_interp : IonRSInterp
        energy losses to CMB, interpolation object
    ics_thomson_ref_tf : Transferfunction
        ICS Thomson regime scattered photon transfer function. 
    ics_rel_ref_tf : Transferfunction
        ICS relativistic regime scattered photon transfer function.
    engloss_ref_tf : Transferfunction
        ICS energy loss scattered photon transfer function.
    ics_only : bool, optional
        If True, turns off atomic cooling for input electrons.
    highengdep_switch: bool, optional
        If False, turns off high energy deposition estimate.
    separate_higheng : bool, optional
        If True, reports the high and low f(z) separately.
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
    xe_init : float
        xe at the initial redshift.
    Tm_init : float
        Matter temperature at the initial redshift.
    coarsen_factor : int
        Coarsening to apply to the transfer function matrix.
    std_soln : bool
        If true, uses the standard TLA solution for f(z).
    xe_func : function, optional
        If provided, fixes the ionization history to the output of this function (which takes redshift as its sole argument). Superceded by xe_reion_func past reion_rs. std_soln must be true.
    user : str
        specify which user is accessing the code, so that the standard solution can be downloaded.  Must be changed!!!
    use_tqdm : bool, optional
        Uses tqdm if true.
    """

    # CODE UP f(z) FOR ARBITRARY INPUT xe

    # Electron and Photon abscissae
    eleceng = in_spec_elec.eng
    photeng = in_spec_phot.eng

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

    # Load the standard TLA solution and set xe/Tm initialize conditions if necessary.
    if std_soln or xe_init == None or Tm_init == None: 
        xe_std, Tm_std, xe_init, Tm_init = load_std(
            xe_init, Tm_init, in_spec_phot.rs
        )
    if std_soln and xe_func is not None:
        xe_std = xe_func
        xe_init = xe_std(in_spec_phot.rs)


    # Initialize the xe and T array that will store the solutions.
    xe_arr  = np.array([xe_init])
    Tm_arr = np.array([Tm_init])

    # redshift/timestep related quantities.
    dlnz = highengphot_tf_interp.dlnz
    prev_rs = None
    rs = in_spec_phot.rs
    dt = dlnz * coarsen_factor / phys.hubble(rs)

    # tqdm related stuff. 
    if use_tqdm:
        from tqdm import tqdm_notebook as tqdm
        pbar = tqdm(total=np.floor((np.log(rs) - np.log(end_rs))/dlnz))

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
                ics_sec_phot_tf, ics_sec_elec_tf, 
                continuum_loss, deposited_ICS_arr
            ) = get_ics_cooling_tf(
                    ics_thomson_ref_tf, ics_rel_ref_tf, engloss_ref_tf,
                    eleceng, photeng, rs, fast=True
                )
        else:
            (
                ics_sec_phot_tf, ics_sec_elec_tf, 
                deposited_ion_arr, deposited_exc_arr, deposited_heat_arr,
                continuum_loss, deposited_ICS_arr
            ) = get_elec_cooling_tf_fast(
                    ics_thomson_ref_tf, ics_rel_ref_tf, engloss_ref_tf,
                    eleceng, photeng, rs, xe_arr[-1], xHe=0
                )

        # Quantities are still per annihilation.            
        ics_phot_spec = ics_sec_phot_tf.sum_specs(in_spec_elec)
        # NOTE: THIS NEEDS TO BE ADDED TO lowengelec!!
        ics_lowengelec_spec = ics_sec_elec_tf.sum_specs(in_spec_elec)

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
    out_lowengelec_specs  = Spectra(
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
    if elec_processes:
        # Add energy deposited in atomic processes. Rescale to
        # energy per baryon per unit time.
        highengdep_grid += np.array([[
            deposited_ion/dt,
            deposited_exc/dt,
            deposited_heat/dt,
            deposited_ICS/dt
        ]])
        

    MEDEA_interp = make_interpolator()

    # Loop while we are still at a redshift above end_rs.
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

            if not highengdep_switch:
                highengdep_fac = 0
            else:
                highengdep_fac = 1

            if std_soln:
                f_raw = compute_fs(
                    MEDEA_interp, next_lowengelec_spec, next_lowengphot_spec,
                    np.array([1-xe_std(rs), 0, 0]), 
                    rate_func_eng_unclustered(rs), dt,
                    highengdep_fac*highengdep_grid[-1], cmbloss_grid[-1],
                    separate_higheng=separate_higheng
                )
            else:
                f_raw = compute_fs(
                    MEDEA_interp, next_lowengelec_spec, next_lowengphot_spec,
                    np.array([1-xe_arr[-1], 0, 0]), 
                    rate_func_eng_unclustered(rs), dt,
                    highengdep_fac*highengdep_grid[-1], cmbloss_grid[-1],
                    separate_higheng=separate_higheng
                )
            if separate_higheng:
                f_low  = np.append(f_low, [f_raw[0]], axis=0)
                f_high = np.append(f_high, [f_raw[1]], axis=0)

                # Compute the f's for the TLA: sum low and high.
                f_H_ion = f_raw[0][0] + f_raw[1][0]
                f_exc   = f_raw[0][2] + f_raw[1][2]
                f_heat  = f_raw[0][3] + f_raw[1][3]
            else:
                f_arr = np.append(f_arr, [f_raw], axis=0)
                # Compute the f's for the TLA.
                f_H_ion = f_raw[0]
                f_exc   = f_raw[2]
                f_heat  = f_raw[3]

            init_cond = np.array([Tm_arr[-1], xe_arr[-1], 0, 0])



            new_vals = tla.get_history(
                init_cond, f_H_ion, f_exc, f_heat,
                rate_func_eng_unclustered, np.array([prev_rs, rs]),
                reion_switch=reion_switch, reion_rs=reion_rs,
                photoion_rate_func=photoion_rate_func, 
                photoheat_rate_func=photoheat_rate_func,
                xe_reion_func=xe_reion_func
            )

            Tm_arr = np.append(Tm_arr, new_vals[-1,0])
            xe_arr  = np.append(xe_arr,  new_vals[-1,1])

        #print('x_e at '+str(rs)+': '+ str(xe_arr[-1]))
        #print('Standard x_e at '+str(rs)+': '+str(xe_std(rs)))
        #print('T_m at '+str(rs)+': '+ str(Tm_arr[-1]))
        #print('Standard T_m at '+str(rs)+': '+str(Tm_std(rs)))
        #if prev_rs is not None:
        #    print('Back Reaction f_ionH, f_ionHe, f_exc, f_heat, f_cont: ', f_raw)

        if std_soln:
            highengphot_tf = highengphot_tf_interp.get_tf(rs, xe_std(rs))
            lowengphot_tf  = lowengphot_tf_interp.get_tf(rs, xe_std(rs))
            lowengelec_tf  = lowengelec_tf_interp.get_tf(rs, xe_std(rs))
            cmbloss_arr = CMB_engloss_interp.get_val(xe_std(rs), rs)
            highengdep_arr = highengdep_interp.get_val(xe_std(rs), rs)
        else:
            highengphot_tf = highengphot_tf_interp.get_tf(rs, xe_arr[-1])
            lowengphot_tf  = lowengphot_tf_interp.get_tf(rs, xe_arr[-1])
            lowengelec_tf  = lowengelec_tf_interp.get_tf(rs, xe_arr[-1])
            cmbloss_arr = CMB_engloss_interp.get_val(xe_arr[-1], rs)
            highengdep_arr = highengdep_interp.get_val(xe_arr[-1], rs)

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
        if elec_processes:
            next_lowengelec_spec  = (
                lowengelec_tf.sum_specs(out_highengphot_specs[-1]) 
                + ics_lowengelec_spec*norm_fac(rs)
            )
        else:
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
                    ics_sec_phot_tf, ics_sec_elec_tf, 
                    continuum_loss, deposited_ICS_arr
                ) = get_ics_cooling_tf(
                        ics_thomson_ref_tf, ics_rel_ref_tf, engloss_ref_tf,
                        eleceng, photeng, rs, fast=True
                    )
            else:
                if std_soln:
                    xe_elec_cooling = xe_std(rs)
                else:
                    xe_elec_cooling = xe_arr[-1]
                (
                    ics_sec_phot_tf, ics_sec_elec_tf, 
                    deposited_ion_arr, deposited_exc_arr, deposited_heat_arr,
                    continuum_loss, deposited_ICS_arr
                ) = get_elec_cooling_tf_fast(
                        ics_thomson_ref_tf, ics_rel_ref_tf, engloss_ref_tf,
                        eleceng, photeng, rs, xe_elec_cooling, xHe=0
                    )

            ics_phot_spec = ics_sec_phot_tf.sum_specs(in_spec_elec)
            ics_lowengelec_spec = ics_sec_elec_tf.sum_specs(in_spec_elec)

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

            next_inj_spec = (
                (in_spec_phot + ics_phot_spec + positronium_phot_spec)
                *norm_fac(rs)
            )
        else:
            next_inj_spec = in_spec_phot * norm_fac(rs)

        # This keeps the redshift.
        next_highengphot_spec.N += next_inj_spec.N

        append_highengphot_spec(next_highengphot_spec)
        append_lowengphot_spec(next_lowengphot_spec)
        append_lowengelec_spec(next_lowengelec_spec)
        cmbloss_grid = np.append(cmbloss_grid, cmbloss)
        highengdep_grid = np.concatenate(
            (highengdep_grid, np.array([highengdep]))
        )

        if verbose:
            print("completed rs: ", prev_rs)

    # f_arr = np.reshape(f_arr,(int(len(f_arr)/5), 5))

    if use_tqdm:
        pbar.close()

    if separate_higheng:
        f_to_return = (f_low, f_high)
    else:
        f_to_return = f_arr

    return (
        xe_arr, Tm_arr,
        out_highengphot_specs, out_lowengphot_specs, out_lowengelec_specs,
        f_to_return
    )
