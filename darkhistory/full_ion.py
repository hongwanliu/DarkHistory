import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d, RegularGridInterpolator
from copy import copy, deepcopy
from tqdm import tqdm_notebook as tqdm

import h5py
import darkhistory.config as config
import darkhistory.physics as phys
from darkhistory.physics import ymu_distortion
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec.spectra import Spectra
from darkhistory.spec.spectools import rebin_N_arr, get_log_bin_width
from darkhistory.electrons.ics.ics_spectrum import ics_spec
import wq_ih_comptonscattering as CS

import matplotlib.pyplot as plt

save_dir_default = config.data_path+'/'
xH_min = 4.53978687e-05 # minimum xH value used in photon transfer functions

def save_h5_dict(fn, d):
    """Save a dictionary to an HDF5 file."""
    with h5py.File(fn, 'w') as hf:
        for key, item in d.items():
            hf.create_dataset(key, data=item)

### FOR OBTAINING THE LOW REDSHIFT EVOLUTION OF A GIVEN ENERGY INJECTION MODEL
def evolve_elecdecay_at_lowz(
    mDM, lifetime=None, sigmav=None, start_rs=4.0, end_rs=1.0, dlnz=0.008,
    save_dir=None, verbose=True, include_heating=True, tworegime=True,
    include_photoion=False, include_PP=False, include_positrons=False
):
    """
    Main function computing spectra due to injection of e+e- pairs from decaying dark matter.

    Parameters
    ----------
    save_dir: string
        Directory at which to save the file

    Returns
    -------
    ndarrays
        tbd
    """
    ### PRELIMINARIES
    # Redshift list to evolve over
    rs_list = np.exp(np.arange(np.log(start_rs), np.log(end_rs), -dlnz))

    # Kinetic energy of primary electrons
    Ee_init = mDM / 2 - phys.me

    # heating rate list
    if include_heating:
        heat_rate_list = np.zeros_like(rs_list)

    # Load photon/electron energy arrays and compton scattering tables.
    # If the tables don't exist, build and save them.
    if save_dir is None:
        save_dir = save_dir_default

    try:
        phot_table = config.load_h5_dict(save_dir+"compton_scattering_phot_matrix.hdf5")
        eng_phot, cs_phot_matrix = phot_table['eng'], phot_table['table']
        del phot_table

        elec_table = config.load_h5_dict(save_dir+"compton_scattering_elec_matrix.hdf5")
        eng_elec, cs_elec_matrix = elec_table['eng'], elec_table['table']
        del elec_table
    except:
        if verbose:
            print("Compton scattering tables not found. Building them now...")
        eng_phot = 10**(np.linspace(-8, 12, 500))
        eng_elec = np.logspace(3, 12, num=100)
        cs_phot_matrix, cs_elec_matrix = build_CS_tables(eng_phot, eng_elec)
        if verbose:
            print("...done!")

    # Photon and electron bin widths
    bw_phot = eng_phot * get_log_bin_width(eng_phot)
    bw_elec = eng_elec * get_log_bin_width(eng_elec)

    # For a single photon, the sum of the photon scattering tables along its energy bin
    # gives its total Compton scattering rate
    cs_phot_rate_out = cs_phot_matrix.sum(axis=1)

    # Inverse compton scattering preliminaries
    ics_tf = config.load_data('ics_tf')
    ics_tf_cache = {
        rs: ics_spec(
            eng_elec, eng_phot, phys.TCMB(rs), T_ref=phys.TCMB(400),
            thomson_tf=ics_tf['thomson'], rel_tf=ics_tf['rel']
        ).grid_vals
    for rs in rs_list}

    # Electron kinetic energy transfer function
    try:
        transfer_Ee = get_electron_tf(
            regenerate=False, save_dir=save_dir, file_name='eng_elec_tf', 
            interpolated=True, tworegime=tworegime
        )
    except:
        if verbose:
            print("Electron kinetic energy transfer function not found. Building it now...")
        # WQ: eventually make tworegime true and change default values
        transfer_Ee = get_electron_tf(
            regenerate=True, interpolated=True, tworegime=tworegime,
            save_dir=save_dir, file_name='eng_elec_tf', save_file=True,
            rs_high=10, rs_low=1, rs_steps=50,
            logE_low=3, logE_fast=8, logE_high=12, 
            logE_steps=25, logE_steps_slow=25, logE_steps_fast=25, 
            dlnz_steps=int(3e4), dlnz_steps_slow=int(1000), dlnz_steps_fast=int(3e4)
        )
        if verbose:
            print("...done!")

    # Initialize quantities for output
    # Photon spectra
    phot_spec = Spectrum(eng_phot, np.zeros_like(eng_phot), rs=1, spec_type='dNdE')
    # Electron spectra
    elec_spec_at_rs = np.zeros((len(rs_list), len(eng_elec))) # total electron spectrum at each time step

    ### START OF MAIN CALCULATION
    # First loop is over each electron injection
    if verbose:
        print("Starting main calculation...")
    for ind_inj, rs_inj in enumerate(tqdm(rs_list)):
        # Deposition only happens at subsequent redshifts
        dep_rs_list = rs_list[rs_list <= rs_inj]

        # Create spectra for tracking deposition due to cascade of this injection step
        sec_phot_spec = Spectrum(eng_phot, np.zeros(len(eng_phot)), rs=rs_inj, spec_type='N')
        sec_elec_spec = np.zeros((len(dep_rs_list), len(eng_phot))) 

        # Calculate the electrons injected per baryon at this step
        dt_inj = dlnz / phys.hubble(rs_inj)
        ne_inj = (
            2 * dt_inj / mDM * phys.inj_rate(
                'decay', rs_inj, mDM=mDM, lifetime=lifetime, sigmav=sigmav
            ) 
        ) / (phys.nB * rs_inj**3)

        # Use transfer function to get evolution of electron energy
        pts_grid = np.meshgrid([rs_inj], [Ee_init], dep_rs_list, indexing='ij')
        pts_list = np.reshape(pts_grid, (3, -1), order='C').T
        Ee_history = transfer_Ee(pts_list)

        # Add heating (per baryon) from these electrons
        if include_heating:
            heat_rate_list[ind_inj:] += ne_inj * phys.elec_heating_engloss_rate(Ee_history, 1, dep_rs_list) # [eV/s]

        # Find indices of electron spectrum corresponding to Ee_history
        # Add ne_inj to corresponding bins
        inds_rs = np.arange(ind_inj, ind_inj+len(dep_rs_list))
        inds_Ee = np.digitize(Ee_history, eng_elec)
        if rs_inj > end_rs:
            elec_spec_at_rs[inds_rs,inds_Ee] += ne_inj
            sec_elec_spec[range(len(dep_rs_list)), inds_Ee] += ne_inj

        ### Start cascade loop
        for ind_dep, rs_dep in enumerate(dep_rs_list):
            # Only consider steps where there is a non-negligible electron energy and number
            dep_rs_future = dep_rs_list[dep_rs_list < rs_dep]
            nonzero_ee = np.where(sec_elec_spec[ind_dep] > 0)[0]
            if (Ee_history[ind_dep] > np.min(eng_elec)) and (len(nonzero_ee) > 0):
                # abundance of background electrons assuming full ionization
                dt_dep = dlnz / phys.hubble(rs_dep)
                ne_BG = (phys.nH + 2 * phys.nHe)*rs_dep**3 

                # Prefactor to convert Compton scattering tables into usual units
                cs_unit_conv = (3/8) * phys.me**2 * phys.thomson_xsec * phys.c * ne_BG * dt_dep # eV^2

                # CMB spectrum at this cascade time step
                CMB_N = phys.CMB_spec(eng_phot, phys.TCMB(rs_dep)) * bw_phot

                # Separate fast and slow electrons
                nonzero_ee_low = np.where(eng_elec[nonzero_ee] < 1e9)[0]
                nonzero_ee_high = np.where(eng_elec[nonzero_ee] >= 1e9)[0]

                ne_bins_sub = sec_elec_spec[ind_dep, nonzero_ee]   

                # This keeps track of changes to photon spectrum at step. 
                # Bring in last step's photons if relevant.
                N_phot_new = np.zeros((len(nonzero_ee), len(eng_phot)))
                if ind_dep != 0:
                    sec_phot_spec.redshift(rs_dep)
                    N_phot_new[0] = sec_phot_spec.N

                ### INVERSE COMPTON SCATTERING
                # Deal with slow electrons
                N_phot_new[nonzero_ee_low] += (
                    ics_tf_cache[rs_dep][nonzero_ee[nonzero_ee_low]] 
                    * dt_dep * ne_bins_sub[nonzero_ee_low, np.newaxis] * bw_phot
                )

                # High energy electrons lose energy fast
                # Divide up this time step and use faster approx for ICS at high energies
                n_fine = 1000
                rs_fine = np.exp(np.linspace(np.log(rs_dep), np.log(rs_dep) - dlnz, n_fine))
                dt_fine = dlnz / n_fine / phys.hubble(rs_fine)
                
                # For every high energy bin
                for ef, eng_fine in enumerate(eng_elec[nonzero_ee[nonzero_ee_high]]):
                    # Get finer electron transfer function
                    pts_fine = np.meshgrid([rs_dep], [eng_fine], rs_fine, indexing='ij')
                    pts_fine = np.reshape(pts_fine, (3, -1), order='C').T
                    Ee_fine = transfer_Ee(pts_fine)

                    # Use fast relativistic ICS secondary expression
                    # and sum over finer transfer function/time steps
                    for rf, rsf in enumerate(rs_fine):
                        temp_N = CS.ICS_spectrum_scaled(
                            rsf, 1 + Ee_fine[rf] / phys.me, eng_phot
                        ) * ne_bins_sub[nonzero_ee_high[ef]] * dt_fine[rf] * bw_phot
                        N_phot_new[nonzero_ee_high[ef]] += temp_N

                # Subtract CMB photons upscattered by each electron bin
                N_phot_new -= phys.thomson_xsec * phys.c * dt_dep * ne_bins_sub[:, np.newaxis] * CMB_N

                ### POSITRON ANNIHILATION (WQ : needs testing)
                if include_positrons:
                    ind_pos = np.argmin((eng_phot - phys.me)**2)
                    N_phot_new[0, ind_pos] += ne_inj

                ### PAIR PRODUCTION OFF OF GAS (WQ : needs testing)
                if include_PP:
                    N_PP = (
                        (
                            xsec_PP_nuc(eng_phot) / (1 - phys.YHe)
                            + xsec_PP_ele(eng_phot) * (1 + 2 * phys.chi) 
                        ) * phys.c * dt_dep * phys.nH * rs_dep**3
                        * N_phot_new.sum(axis=0)
                    )
                    
                    Ne_PP_total = np.zeros_like(eng_elec)
                    for ii, E_phot in enumerate(eng_phot):
                        if E_phot > 2 * phys.me:
                            Ne_PP_total += Ne_PP_single(E_phot, eng_elec, bw_elec) * N_PP[ii]
                    
                    valid_PP = (Ne_PP_total > 1e-40)
                    if len(dep_rs_future) > 0 and np.any(valid_PP):
                        nonzero_PP = np.where(valid_PP)[0]
                        Ee_PP_vals = eng_elec[nonzero_PP]   # energies on photon grid, shifted by rydberg
                        Ne_PP_vals = Ne_PP_total[nonzero_PP]   # counts per baryon

                        for ff, rs_fut in enumerate(dep_rs_future):
                            abs_idx =ind_inj + ind_dep + 1 + ff

                            pts_PP = np.column_stack([
                                np.full(len(nonzero_PP), rs_dep),
                                Ee_PP_vals,
                                np.full(len(nonzero_PP), rs_fut)
                            ])
                            Ee_PP_future = transfer_Ee(pts_PP)

                            if include_heating:
                                heat_rate_list[abs_idx] += np.sum(Ne_PP_vals * phys.elec_heating_engloss_rate(Ee_PP_future, 1, rs_fut))

                            inds_PP_ff = np.digitize(Ee_PP_future, eng_elec)
                            inds_PP_ff[inds_PP_ff == len(eng_elec)] = 0
                            np.add.at(elec_spec_at_rs[abs_idx], inds_PP_ff, Ne_PP_vals)
                            np.add.at(sec_elec_spec[ind_dep + 1 + ff], inds_PP_ff, Ne_PP_vals)
      
                ### FOR TESTING AGAINST DH (WQ : needs testing)
                ### include some photoionizations due to photon transfer functions not going to full ionization
                if include_photoion:
                    photo_ion_xsec_list = phys.photo_ion_xsec(eng_phot, species='HI') # cm^2
                    Ne_photoion = N_phot_new * photo_ion_xsec_list * phys.c * dt_dep * xH_min * phys.nH * rs_dep**3 # per_baryon
                    N_phot_new -= Ne_photoion
                    Ne_photoion = Ne_photoion.sum(axis=0)
                    Ee_photoion = eng_phot - phys.rydberg

                    # Add photoionization electrons (same propagation pattern as Compton secondaries)
                    valid_photo = (Ee_photoion > 3e3) & (Ne_photoion > 1e-40)
                    if len(dep_rs_future) > 0 and np.any(valid_photo):
                        nonzero_photo = np.where(valid_photo)[0]
                        Ee_photo_vals = Ee_photoion[nonzero_photo]   # energies on photon grid, shifted by rydberg
                        Ne_photo_vals = Ne_photoion[nonzero_photo]   # counts per baryon

                        for ff, rs_fut in enumerate(dep_rs_future):
                            abs_idx =ind_inj + ind_dep + 1 + ff

                            pts_photo = np.column_stack([
                                np.full(len(nonzero_photo), rs_dep),
                                Ee_photo_vals,
                                np.full(len(nonzero_photo), rs_fut)
                            ])
                            Ee_photo_future = transfer_Ee(pts_photo)

                            if include_heating:
                                heat_rate_list[abs_idx] += np.sum(Ne_photo_vals * phys.elec_heating_engloss_rate(Ee_photo_future, 1, rs_fut))

                            inds_photo_ff = np.digitize(Ee_photo_future, eng_elec)
                            inds_photo_ff[inds_photo_ff == len(eng_elec)] = 0
                            np.add.at(elec_spec_at_rs[abs_idx], inds_photo_ff, Ne_photo_vals)
                            np.add.at(sec_elec_spec[ind_dep + 1 + ff],        inds_photo_ff, Ne_photo_vals)

                ### COMPTON SCATTERING
                # Change to photon spectrum
                N_scatter_out = N_phot_new * cs_phot_rate_out * cs_unit_conv # photons scattered out
                N_scatter_in  = (N_phot_new * cs_unit_conv) @ cs_phot_matrix # photons scattered in
                N_final       = N_phot_new - N_scatter_out + N_scatter_in

                # Secondary electrons, summed over all primary electron bins
                sec_elec_ICS = (
                    (N_scatter_out / cs_phot_rate_out.clip(min=1e-300)) @ cs_elec_matrix
                ).sum(axis=0)

                if len(dep_rs_future) > 0 and np.any(sec_elec_ICS > 0):
                    nonzero_sec  = np.where(sec_elec_ICS > 1e-40)[0]
                    Ee_secondary = eng_elec[nonzero_sec]

                    for ind_fut, rs_fut in enumerate(dep_rs_future):
                        abs_idx =ind_inj + ind_dep + 1 + ind_fut

                        pts = np.column_stack([
                            np.full(len(nonzero_sec), rs_dep),
                            Ee_secondary,
                            np.full(len(nonzero_sec), rs_fut)
                        ])
                        Ee_future = transfer_Ee(pts)
                        finite_mask = Ee_future > 1e-6

                        # Add heat from secondary electrons
                        if include_heating:
                            heat_rate_list[abs_idx] += np.sum(
                                sec_elec_ICS[nonzero_sec][finite_mask]
                                * phys.elec_heating_engloss_rate(Ee_future[finite_mask], 1, rs_fut)
                            )

                        # Add electrons to spectrum
                        inds_Ee_fut = np.digitize(Ee_future, eng_elec)
                        inds_Ee_fut[inds_Ee_fut == len(eng_elec)] = 0
                        np.add.at(elec_spec_at_rs[abs_idx], inds_Ee_fut, sec_elec_ICS[nonzero_sec])
                        np.add.at(sec_elec_spec[ind_dep + 1 + ind_fut], inds_Ee_fut, sec_elec_ICS[nonzero_sec]) 

                ### Redshift and accumulate distortion
                sec_phot_spec = Spectrum(eng_phot, N_final.sum(axis=0), rs=rs_dep, spec_type='N')

        sec_phot_spec.redshift(1)
        phot_spec.dNdE += sec_phot_spec.dNdE
                    
    ### Get y-type distortion
    if include_heating:
        if verbose:
            print("getting y-distortion...")

        # Calculate change to temperature
        heat_rate_interp = interp1d(rs_list, heat_rate_list, bounds_error=False, fill_value=0)

        def dTdz(rs, T): # in eV
            return - heat_rate_interp(rs) * 2 / 3 / (2 + phys.chi) / phys.hubble(rs) / rs # 2 + 3*phys.chi?
        delTsol = solve_ivp(dTdz, [start_rs, end_rs], [0.0], dense_output=True)

        # Calculate y-parameter
        def heat_integrand(rs):
            delT = delTsol.sol(rs)
            return - (
                delT * phys.thomson_xsec * phys.c * (phys.nH * rs**3) / phys.me # s^-1
                / phys.hubble(rs) / rs # s
            )
        
        ypar = quad(heat_integrand, start_rs, end_rs)[0]

        # Get y-type distortion
        y_dist = phys.ymu_distortion(eng_phot, ypar, 1., 'y')
        phot_spec.dNdE += y_dist.dNdE

        f_heat = heat_rate_list / (
            phys.inj_rate(
                'decay', rs_list, mDM=mDM, lifetime=lifetime, sigmav=sigmav
            ) / (phys.nB * rs_list**3)
        )

    if verbose:
        print("...done!")

    res = {
        'rs' : rs_list,
        'elec_eng' : eng_elec,
        'elec_spec' : elec_spec_at_rs,
        'distortion' : phot_spec
    }
    
    if include_heating:
        res['y'] = ypar
        res['f_heat'] = f_heat

    return res

### FOR GENERATING ELECTRON KINETIC ENERGY TRANSFER FUNCTION
### ASSUMING CONTINUOUS ENERGY LOSS

def electron_energy_loss_rate(rs, Ee, xe):
    """
    The energy loss rate of an electron, assuming we can treat
    the loss rate as continuous. Since we do not assume the
    electrons lose their energy quickly, this function accounts for 
    redshifting, inverse compton scattering, and heating.

    Parameters
    ----------
    rs : float
        Redshift
    Ee : float
        Electron kinetic energy in eV.
    xe : float
        Free electron fraction.

    Returns
    -------
    ndarray
        dE / dt in units of [eV/s]
    """
    # Calculate some preliminary quantities
    gamma = 1 + Ee / phys.me
    beta = np.sqrt(1 - 1 / gamma**2)

    # Calculate the individual cooling terms
    adiabatic = 2 * phys.hubble(rs) * Ee # [eV / s]
    ICS = (4/3) * phys.thomson_xsec * phys.c * gamma**2 * beta**2 * phys.CMB_eng_density(phys.TCMB(rs)) # [eV / s]
    heating = phys.elec_heating_engloss_rate(Ee, xe, rs) # [eV / s]
    return - (adiabatic + ICS + heating)

def generate_lowz_electron_energy_tf(
    rs_high=10, rs_low=1, rs_steps=50,
    logE_low=3, logE_high=12, logE_steps=25, dlnz_steps=int(3e4), 
    save_file=False, save_dir=None, file_name=None, return_tf=False
):
    """
    Generates the electron kinetic energy transfer function
    as a function of injection redshift, electron energy, and evolution redshift.

    Parameters
    ----------
    rs_high : float
        Highest redshift at which to generate transfer function
    rs_low : float
        Lowest redshift at which to generate transfer function
    rs_steps : int
        Number of steps to use for injection redshift array
    logE_low : float
        Lowest electron kinetic energy at which to generate transfer function
    logE_high : float
        Highest electron kinetic energy at which to generate transfer function
    logE_steps : int
        Number of steps to use for electron kinetic energy array
    dlnz_steps : int
        Number of steps to use for evolution redshift array
    save_file : bool
        Option to save the transfer function as a file
    save_dir: string
        Directory at which to save the file
    file_name : string
        Save file name
    return_tf : bool
        Option to return the transfer function dictionary

    Returns
    -------
    None or dict
        Depends on the parameter 'return_tf'
    """

    # Sanity checks
    if logE_low > logE_high:
        raise ValueError("lowest energy for electron transfer function is higher than highest energy")

    # List of injection redshifts and electron kinetic energies to solve
    inj_rs_list = np.geomspace(rs_high, rs_low, num=rs_steps)
    Ee_list = np.logspace(logE_low, logE_high, num=logE_steps)

    # d lnz after injection
    dlnz_list = -np.linspace(0, np.log(rs_high-rs_low), num=dlnz_steps)

    # Stopping event for scipy handler (if electron energy goes below zero)
    def check_energy_sign(t, E):
        return E[0] - 1
    check_energy_sign.terminal = True

    transfer_Ee_array = np.zeros((len(inj_rs_list), len(Ee_list), len(dlnz_list)))

    # For each injection redshift
    for ii, inj_rs in enumerate(tqdm(inj_rs_list)):
        # For each injection electron energy
        lnz_span = np.log(inj_rs) + dlnz_list
        for jj, Ee_init in enumerate(Ee_list):
            # Solve ODE for energy evolution
            def dE_dlnz(lnz, Ee):
                rs = np.exp(lnz)
                xe = 1 + 2*phys.chi # assume fully ionized at late redshifts
                dE_dt = electron_energy_loss_rate(rs, Ee, xe) # [eV / s]
                return - dE_dt / phys.hubble(rs) # [eV / dlnz]

            Ee_sol = solve_ivp(
                dE_dlnz, 
                (lnz_span[0], lnz_span[-1]), 
                [Ee_init],
                t_eval=lnz_span,
                rtol=1e-15,
                events=check_energy_sign
            )
            transfer_Ee_array[ii, jj, :len(Ee_sol.t)] = Ee_sol.y[0]

    # Make dictionary with all the arrays
    elec_tf = {
        'rs_inj' : inj_rs_list,
        'eng_elec' : Ee_list,
        'dlnz' : dlnz_list,
        'tf' : transfer_Ee_array
    }

    # Save the transfer function array, if relevant
    if save_file:
        if save_dir is None:
            save_dir = save_dir_default

        if file_name is None:
            raise ValueError("'save_file' is true but no save file name is given")
        else:
            save_h5_dict(save_dir+file_name+".hdf5", elec_tf)
            
    if return_tf:
        return elec_tf
    else:
        return 

def generate_lowz_electron_tf_tworegime(
    rs_high=10, rs_low=1, rs_steps=50,
    logE_low=3, logE_fast=8, logE_high=12, 
    logE_steps_slow=25, logE_steps_fast=25, 
    dlnz_steps_slow=int(1000), dlnz_steps_fast=int(3e4), 
    save_file=False, save_dir=None, file_name=None, return_tf=False
):
    """
    Generates two electron kinetic energy transfer functions,
    for slow and fast energy loss regimes.

    Parameters
    ----------
    rs_high : float
        Highest redshift at which to generate transfer function
    rs_low : float
        Lowest redshift at which to generate transfer function
    rs_steps : int
        Number of steps to use for injection redshift array
    logE_low : float
        Lowest electron kinetic energy at which to generate transfer function
    logE_fast : float
        Electron energy threshold for fast cooling.
    logE_high : float
        Highest electron kinetic energy at which to generate transfer function
    logE_steps_low : int
        Number of steps to use for slow electron kinetic energy array
    logE_steps_fast : int
        Number of steps to use for fast electron kinetic energy array
    dlnz_steps_slow : int
        Number of steps to use for slow evolution redshift array
    dlnz_steps_fast : int
        Number of steps to use for fast evolution redshift array
    save_file : bool
        Option to save the transfer function as a file
    save_dir: string
        Directory at which to save the file
    file_name : string
        Save file name
    return_tf : bool
        Option to return the transfer function dictionary

    Returns
    -------
    None or dict
        Depends on the parameter 'return_tf'
    """
    tf = {}
    tf['fast'] = generate_lowz_electron_energy_tf(
        rs_high=rs_high, rs_low=rs_low, rs_steps=rs_steps,
        logE_low=logE_fast, logE_high=logE_high, 
        logE_steps=logE_steps_fast, dlnz_steps=dlnz_steps_fast, 
        save_file=False, return_tf=True
    )
    tf['slow'] = generate_lowz_electron_energy_tf(
        rs_high=rs_high, rs_low=rs_low, rs_steps=rs_steps,
        logE_low=logE_low, logE_high=logE_fast, 
        logE_steps=logE_steps_slow, dlnz_steps=dlnz_steps_slow, 
        save_file=False, return_tf=True
    )

    # Save the transfer function array, if relevant
    if save_file:
        if save_dir is None:
            save_dir = save_dir_default
            
        if file_name is None:
            raise ValueError("'save_file' is true but no save file name is given")
        else:
            save_h5_dict(save_dir+file_name+"_slow.hdf5", tf['slow'])
            save_h5_dict(save_dir+file_name+"_fast.hdf5", tf['fast'])
            
    if return_tf:
        return tf
    else:
        return 

def make_electron_tf_interpolator(elec_tf):
    """
    Given electron transfer function dictionary, creates interpolating function.

    Parameters
    ----------
    elec_tf : dict
        Dictionary contain injection redshift list, electron kinetic energy list, 
        dlnz list, and electron kinetic energies.

    Returns
    -------
    interp
        Interpolating function
    """
    # Interpolate over log quantities
    logz_inj_list = np.log10(elec_tf['rs_inj'])
    log_Ee_list = np.log10(elec_tf['eng_elec'])
    log_transfer_Ee_interp = RegularGridInterpolator(
        (logz_inj_list, log_Ee_list, elec_tf['dlnz']), np.log10(elec_tf['tf']),
        fill_value=None, bounds_error=False
    )
    # Define a function that will map from quantities to their log
    # and obtain the interpolated transfer function
    def transfer_Ee_log(pars_in):
        pars_in = np.atleast_2d(pars_in)
        logz_inj = np.log10(pars_in[:, 0])
        log_Ee   = np.log10(pars_in[:, 1])
        lnz_dep  = np.log(pars_in[:, 2])
        dlnz     = lnz_dep - np.log(10**logz_inj)
        pars_out = np.stack([logz_inj, log_Ee, dlnz], axis=-1)
        return 10**log_transfer_Ee_interp(pars_out)
    return transfer_Ee_log

def get_electron_tf(
    regenerate=False, save_dir=None, file_name=None, save_file=False,
    interpolated=True, tworegime=True,
    rs_high=10, rs_low=1, rs_steps=50,
    logE_low=3, logE_fast=8, logE_high=12, 
    logE_steps=25, logE_steps_slow=25, logE_steps_fast=25, 
    dlnz_steps=int(3e4), dlnz_steps_slow=int(1000), dlnz_steps_fast=int(3e4)
):
    """
    Obtains 

    Parameters
    ----------
    regenerate : bool
        Option to regenerate the electron kinetic energy transfer functions.
        Otherwise, load from file.
    save_dir: string
        Directory at which to save the file
    file_name : string
        File name for loading transfer function
    interpolated : bool
        Option to return transfer function as interpolating function. 
        Otherwise, returns as dictionary.
    tworegime : bool
        Option to separately treat fast and slow electrons.

    Returns
    -------
    dict or interpolated function
        Depends on the parameter 'interpolated'
    """
    # Regenerate transfer function if necessary
    if regenerate:
        if tworegime:
            elec_tf = generate_lowz_electron_tf_tworegime(
                rs_high=rs_high, rs_low=rs_low, rs_steps=rs_steps,
                logE_low=logE_low, logE_fast=logE_fast, logE_high=logE_high, 
                logE_steps_slow=logE_steps_slow, logE_steps_fast=logE_steps_fast, 
                dlnz_steps_slow=dlnz_steps_slow, dlnz_steps_fast=dlnz_steps_fast, 
                save_file=save_file, file_name=file_name, return_tf=True
            )
        else:
            elec_tf = generate_lowz_electron_energy_tf(
                rs_high=rs_high, rs_low=rs_low, rs_steps=rs_steps,
                logE_low=logE_low, logE_high=logE_high,
                logE_steps=logE_steps, dlnz_steps=dlnz_steps,
                save_file=save_file, save_dir=save_dir, file_name=file_name,
                return_tf=True
            )
    # Otherwise, load from file
    else:
        # Load transfer function
        if save_dir is None:
            save_dir = save_dir_default

        if file_name is None:
            raise ValueError("Loading electron tf, but no save file name is given")
        else:
            if tworegime:
                elec_tf = {
                    'slow' : config.load_h5_dict(save_dir+file_name+"_slow.hdf5"),
                    'fast' : config.load_h5_dict(save_dir+file_name+"_fast.hdf5")
                }
            else:
                elec_tf = config.load_h5_dict(save_dir+file_name+".hdf5")

    # Choose whether to return transfer function as interpolated function or dictionary
    if interpolated:
        if tworegime:
            tf_interp = {
                'fast' : make_electron_tf_interpolator(elec_tf['fast']),
                'slow' : make_electron_tf_interpolator(elec_tf['slow'])
            }
        else:
            tf_interp = make_electron_tf_interpolator(elec_tf)
        return tf_interp
    else:
        return elec_tf

### COMPTON AND INVERSE COMPTON SCATTERING (some translated from Tracy's IDL code)

# ### ICS

# # Distribution function of ICS photons
# # in terms of normalized final energy, E_f = E_final / (4 E_init gamma^2)
# def f_ICS(E_f):
#     lnE = np.log(E_f)
#     if np.isscalar(E_f):
#         if E_f == 0:
#             lnE = 0
#     else:
#         lnE[E_f == 0] = 0
#     return 2 * E_f * lnE + E_f + 1 - 2 * E_f**2

# # Spectrum per time of photons from ICS of CMB,
# # in terms of dN / dt / dE_final
# def ICS_spectrum(rs, gamma, E_final):
#     ECMB_list = np.linspace(1e-3*phys.TCMB(rs), 10*phys.TCMB(rs), num=500)
#     dE = ECMB_list[1] - ECMB_list[0]
#     E_f = E_final / (4 * ECMB_list * gamma**2)
#     mask = (E_f >= 0) & (E_f <= 1)

#     CMB_integral = np.sum(phys.CMB_spec(ECMB_list[mask], phys.TCMB(rs)) / ECMB_list[mask] * f_ICS(E_f[mask]) * dE) # [1 / eV / cm^3]
#     return(
#         2 * np.pi * phys.ele_rad**2 * phys.c / gamma**2 # [cm^3 / s]
#         *  CMB_integral # [1 / eV / cm^3]
#     ) # final units [ 1 / s / eV]

# # Calculate ICS spectrum across huge range of energies to use scaling relations
# eng_list = 10**(np.linspace(-8, 10, 300))
# gamma_ref = 1e3
# rs_ref = 4
# compspec_ref = np.zeros_like(eng_list)

# for jj, eng in enumerate(eng_list):
#     compspec_ref[jj] = ICS_spectrum(rs_ref, gamma_ref, eng)
# compspec_ref = interp1d(eng_list, compspec_ref, bounds_error=False, fill_value=(compspec_ref[0], 0))

# # Define function to take advantage of scaling relation
# def ICS_spectrum_scaled(
#     rs, gamma, E_final, 
#     spec_ref=compspec_ref, rs_ref=rs_ref, gamma_ref=gamma_ref
# ):
#     rs_fac = rs / rs_ref
#     g_fac = rs_fac * gamma / gamma_ref
#     return rs_fac**4 / g_fac**2 * spec_ref(E_final * rs_fac / g_fac**2)

# ### Compton scattering
# ### Translated from Tracy Slatyer's old IDL code

# def secondaryspec(spec, inbins, outbins, secbins):
#     """
#     Python/Numpy translation of IDL secondaryspec.pro

#     Parameters
#     ----------
#     spec : array_like
#         Spectrum describing dN in `outbins` from a unit in `inbins`.
#         In IDL this is (n_inbins x n_outbins). It may also be 1D
#         (length n_outbins); in that case it is reshaped to (n_inbins, n_outbins).
#     inbins : array_like
#         Incoming energies (same units as outbins).
#         Length n_inbins. Scalars are allowed (treated as length-1).
#     outbins : array_like
#         Outgoing photon energies, length n_outbins.
#     secbins : array_like
#         Secondary (electron) energy binning (kinetic), length n_secbins.

#     Returns
#     -------
#     secspec : ndarray, shape (n_inbins, n_secbins)
#         Secondary spectrum in secbins for each inbin.
#     """
#     inbins = np.atleast_1d(np.asarray(inbins, dtype=float))
#     outbins = np.asarray(outbins, dtype=float)
#     secbins = np.asarray(secbins, dtype=float)

#     n_in = inbins.size
#     n_out = outbins.size
#     n_sec = secbins.size

#     spec = np.asarray(spec, dtype=float)
#     if spec.ndim < 2:
#         spec = spec.reshape(n_in, n_out)

#     secspec = np.zeros((n_in, n_sec), dtype=float)

#     conversion = inbins[:, None] - outbins[None, :]

#     # -------------------------------------------------
#     # newindices = interpol([-2, -1, 0..n_sec-1], [-max(outbins), 0, secbins], conversion)
#     # -------------------------------------------------
#     # Map energies -> "index-like" positions in secbins.
#     # Use np.interp to mimic IDL INTERPOL behavior.
#     x_table = np.concatenate(([-np.max(outbins)], [0.0], secbins))
#     y_table = np.concatenate(([-2.0, -1.0], np.arange(n_sec, dtype=float)))

#     # Flatten -> interp -> reshape back
#     flat_conv = conversion.ravel()
#     flat_newidx = np.interp(flat_conv, x_table, y_table, left=-2.0, right=float(n_sec - 1))
#     newindices = flat_newidx.reshape(conversion.shape)

#     # -------------------------------------------------
#     # Loop over outgoing bins, distribute into secbins
#     # -------------------------------------------------
#     for j in range(n_out):
#         col = newindices[:, j]

#         # IDL: ind = where(newindices[*,j] GT 0 and LT n_elements(secbins)-1, ngood)
#         mask_good = (col > 0.0) & (col < (n_sec - 1))
#         if not np.any(mask_good):
#             continue

#         ind = np.where(mask_good)[0]
#         x = col[ind]  # fractional indices into secbins

#         # Linear split between floor(x) and ceil(x)
#         floor_x = np.floor(x).astype(int)
#         ceil_x = np.ceil(x).astype(int)

#         w_low = (ceil_x - x) * spec[ind, j]
#         w_high = (x - floor_x) * spec[ind, j]

#         secspec[ind, floor_x] += w_low
#         secspec[ind, ceil_x] += w_high

#         # IDL: whole = where(x EQ round(x), nwhole)
#         # Handle exactly integer indices (deposit full spec once)
#         whole_mask = (x == np.round(x))
#         if np.any(whole_mask):
#             x_int = np.round(x[whole_mask]).astype(int)
#             ind_whole = ind[whole_mask]
#             secspec[ind_whole, x_int] += spec[ind_whole, j]

#     return secspec

# def ih_comptonscattering(photengbins=None,
#                          elecengbins=None,
#                          calcsec=False):
#     """
#     Python/Numpy translation of IDL ih_comptonscattering.

#     Parameters
#     ----------
#     photengbins : dict or None
#         If None, construct a default logarithmic photon grid.
#         If provided, expected keys:
#             - 'nbins'     : int, number of photon energy bins
#             - 'bins'      : 1D array of bin edges, length nbins+1
#             - 'photoneng' : 1D array of bin-centered energies, length nbins
#     elecengbins : dict or None
#         If None, construct default logarithmic electron energy bins.
#         If provided, expected keys:
#             - 'nbins'  : int, number of electron energy bins
#             - 'dlneng' : float, log spacing
#             - 'bins'   : 1D array of bin edges, length nbins+1
#     calcsec : bool
#         If True, also compute the secondary electron rate table.

#     Returns
#     -------
#     integratedrate0 : ndarray, shape (nphoteng, nphoteng)
#         Integrated Compton scattering rate between photon bins.
#         Axis 0: incoming photon bin index i1
#         Axis 1: outgoing photon bin index.
#     secondaryrate0 : ndarray or None, shape (nphoteng, neng)
#         If calcsec=True, the secondary electron rate table.
#         Otherwise, None.
#     """
#     # --------------------------------------------
#     # Photon energy bins
#     # --------------------------------------------
#     if photengbins is None:
#         nphoteng = 500
#         dlnphoteng = np.log(1e13 / 1e-2) / nphoteng

#         idx = np.arange(nphoteng, dtype=float)
#         photenglow = 1e-2 * np.exp(idx * dlnphoteng)
#         photenghigh = 1e-2 * np.exp((idx + 1.0) * dlnphoteng)

#         photeng = np.sqrt(photenglow * photenghigh)  # geometric mean
#         photbinwidth = photenghigh - photenglow
#     else:
#         nphoteng = int(photengbins["nbins"])
#         photbins = np.asarray(photengbins["bins"], dtype=float)

#         photenglow = photbins[:-1]
#         photenghigh = photbins[1:]
#         photeng = np.asarray(photengbins["photoneng"], dtype=float)

#         dlnphoteng = np.log(photbins[1] / photbins[0])
#         photbinwidth = photenghigh - photenglow

#     # --------------------------------------------
#     # Electron energy bins
#     # --------------------------------------------
#     if elecengbins is None:
#         neng = 500
#         dlneng = np.log(1e13) / neng

#         idx = np.arange(neng + 1, dtype=float)
#         elecbins = phys.me + np.exp(idx * dlneng)

#         englow = elecbins[:-1]
#         enghigh = elecbins[1:]
#         eng = phys.me + np.sqrt((englow - phys.me) * (enghigh - phys.me))
#         elecbinwidth = enghigh - englow
#     else:
#         neng = int(elecengbins["nbins"])
#         dlneng = float(elecengbins["dlneng"])
#         elecbins = np.asarray(elecengbins["bins"], dtype=float)

#         englow = elecbins[:-1]
#         enghigh = elecbins[1:]
#         eng = phys.me + np.sqrt((englow - phys.me) * (enghigh - phys.me))
#         elecbinwidth = enghigh - englow

#     # print("building integrated rate table for Compton scattering")

#     integratedrate0 = np.zeros((nphoteng, nphoteng), dtype=float)
#     secondaryrate0 = np.zeros((nphoteng, neng), dtype=float) if calcsec else None

#     # Sub-bin sampling
#     nsub = 200
#     dsub = 1.0 / nsub
#     subshift = (np.arange(nsub, dtype=float) + 0.5) * dsub

#     # --------------------------------------------
#     # Main loops in IDL:
#     # for i1 = 1L, nphoteng-2
#     # --------------------------------------------
#     for i1 in tqdm(range(1, nphoteng - 1)):  # 1 .. nphoteng-2
#         for i2 in range(nsub):
#             Ein = photeng[i1] * np.exp((subshift[i2] - 0.5) * dlnphoteng)

#             lowspectrumlimit = Ein / (1.0 + 2.0 * Ein / phys.me)

#             mask = (photenglow <= Ein) & (photenghigh > lowspectrumlimit)
#             if not np.any(mask):
#                 continue

#             ind = np.where(mask)[0]

#             Elow = photenglow[ind].copy()
#             Ehigh = photenghigh[ind].copy()

#             Elow[0] = lowspectrumlimit
#             Ehigh[-1] = Ein

#             term1 = (Ehigh - Elow) * (
#                 Elow * Ehigh * (2 * phys.me + (4.0 + Elow / phys.me + Ehigh / phys.me) * Ein)
#                 + 2.0 * phys.me * Ein**2
#             )
#             term2 = (
#                 2.0
#                 * np.log(Ehigh / Elow)
#                 * Elow
#                 * Ehigh
#                 * Ein
#                 * (-2 * phys.me + Ein * (-2.0 + Ein / phys.me))
#             )
#             row = (term1 + term2) / (2.0 * Elow * Ehigh * Ein**2 * Ein**2)

#             integratedrate0[i1, ind] += dsub * row

#             if calcsec:
#                 # secbins here are kinetic electron energies: eng - phys.me
#                 secspec_2d = secondaryspec(dsub * row,
#                                            inbins=Ein,
#                                            outbins=photeng[ind],
#                                            secbins=eng - phys.me)
#                 # Only one incoming bin used, so take the first row
#                 secondaryrate0[i1, :] += secspec_2d[0, :]

#     return integratedrate0, secondaryrate0

def build_CS_tables(
    eng_phot, eng_elec, save_dir=None
):
    """
    Creates and saves tables for spectra of scattered photons and 
    secondary electrons due to Compton scattering.

    Parameters
    ----------
    eng_phot: ndarray
        Photon energies
    eng_elec: ndarray
        Photon energies
    save_dir: string
        Directory at which to save the file

    Returns
    -------
    ndarrays
        Tables for photon scattering and secondary electrons
    """
    ### Build bin-edge dicts for ih_comptonscattering from the notebook's energy grids
    # Photon bins: eng_phot are geometric-mean bin centersx
    dlnE_phot = np.log(eng_phot[1] / eng_phot[0])
    phot_edges = np.concatenate([
        [eng_phot[0] * np.exp(-dlnE_phot / 2)],
        np.sqrt(eng_phot[:-1] * eng_phot[1:]),
        [eng_phot[-1] * np.exp(dlnE_phot / 2)]
    ])
    photengbins = {'nbins': len(eng_phot), 'bins': phot_edges, 'photoneng': eng_phot}

    # Electron bins: eng_elec are kinetic energies (eV); ih_comptonscattering uses total energy = phys.me + KE
    dlnE_elec = np.log(eng_elec[1] / eng_elec[0])
    KE_edges = np.concatenate([
        [eng_elec[0] * np.exp(-dlnE_elec / 2)],
        np.sqrt(eng_elec[:-1] * eng_elec[1:]),
        [eng_elec[-1] * np.exp(dlnE_elec / 2)]
    ])
    elecengbins = {'nbins': len(eng_elec), 'bins': phys.me + KE_edges, 'dlneng': dlnE_elec}

    ### Use translated version of Tracy's IDL code to get CS tables
    cs_phot_matrix, cs_elec_matrix = CS.ih_comptonscattering(
        photengbins=photengbins, elecengbins=elecengbins, calcsec=True
    )
    # cs_phot_matrix[i, j]: rate for photons from bin i to scatter into bin j
    # cs_elec_matrix[i, k]:  rate for photons from bin i to produce secondary electrons in bin k
    # Both are proportional to dσ/dE' — multiply by (normalization * n_e * c * dt) to get physical rates.

    ### Fix noisy low-energy bins
    # Find where photon energy is less than 10 eV and zero out table
    ind_noise = np.where(eng_phot > 10)[0][0]
    cs_phot_matrix[:ind_noise, :ind_noise] = 0.0

    # Scattering transfer function should just be diagonal
    # The table has weird units, so this is what you need to get the Thomson scattering limit
    thomson_limit_val = 8 / (3 * phys.me**2)  # σ_T / (π α²) in eV^{-2}
    np.fill_diagonal(cs_phot_matrix[:ind_noise, :ind_noise], thomson_limit_val)

    ### This table took a long time to calculate---save it and load later
    cs_phot_dict = {
        'eng' : eng_phot,
        'table' : cs_phot_matrix
    }
    cs_elec_dict = {
        'eng' : eng_elec,
        'table' : cs_elec_matrix
    }

    if save_dir is None:
        save_dir = save_dir_default

    save_h5_dict(save_dir+"compton_scattering_phot_matrix.hdf5", cs_phot_dict)
    save_h5_dict(save_dir+"compton_scattering_elec_matrix.hdf5", cs_elec_dict)
    return cs_phot_matrix, cs_elec_matrix

### FOR PAIR PRODUCTION
def xsec_PP_nuc(E_phot):
    if np.isscalar(E_phot):
        if E_phot > 2 * phys.me:
            E_norm = E_phot / phys.me
            return phys.alpha * phys.ele_rad**2 * (
                28 / 9 * np.log(2*E_norm) - 218/27
            ) # cm^2
    else:
        xsec = np.zeros_like(E_phot)
        E_norm = E_phot / phys.me
        pp_mask = E_phot > 2 * phys.me
        xsec[pp_mask] = phys.alpha * phys.ele_rad**2 * (
            28 / 9 * np.log(2*E_norm[pp_mask]) - 218/27
        )
        return xsec
    
def xsec_PP_ele(E_phot):
    if np.isscalar(E_phot):
        if E_phot > 2 * phys.me:
            E_norm = E_phot / phys.me
            return phys.alpha * phys.ele_rad**2 * (
                28 / 9 * np.log(2*E_norm) - 100/9
            ) # cm^2
    else:
        xsec = np.zeros_like(E_phot)
        E_norm = E_phot / phys.me
        pp_mask = E_phot > 2 * phys.me
        xsec[pp_mask] = phys.alpha * phys.ele_rad**2 * (
            28 / 9 * np.log(2*E_norm[pp_mask]) - 100/9
        )
        return xsec
    
# Spectrum of electrons per PP event. Approximated as flat
def Ne_PP_single(eng_phot, eng_elec, bw_elec):
    dNdE = np.zeros_like(eng_elec)
    Ee_max_PP = eng_phot - 2 * phys.me
    dNdE[eng_elec < Ee_max_PP] = 2 / (Ee_max_PP) # this normalization results in 2 electrons per PP event
    return dNdE * bw_elec
                        
### FOR CREATING LOW REDSHIFT SPECTRAL DISTORTION TRANSFER FUNCTIONS

def generate_lowz_spectral_distortion_tf():
    return