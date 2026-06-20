import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d, RegularGridInterpolator
from copy import copy, deepcopy
from tqdm import tqdm_notebook as tqdm

import h5py
import darkhistory.config as config
# from darkhistory.config import load_data
from darkhistory.electrons.elec_cooling import get_elec_cooling_tf
import darkhistory.physics as phys
from darkhistory.physics import ymu_distortion
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec.spectra import Spectra
from darkhistory.spec.spectools import rebin_N_arr, get_log_bin_width
from darkhistory.electrons.ics.ics_spectrum import ics_spec
import wq_ih_comptonscattering as CS # rename this eventually

save_dir_default = config.data_path+'/'
xH_min = 4.53978687e-05 # minimum xH value used in photon transfer functions

def save_h5_dict(fn, d):
    """Save a dictionary to an HDF5 file."""
    with h5py.File(fn, 'w') as hf:
        for key, item in d.items():
            hf.create_dataset(key, data=item)

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
            save_h5_dict(save_dir+file_name, elec_tf)
            
    if return_tf:
        return elec_tf
    else:
        return 

def generate_lowz_electron_tf_tworegime(
    rs_high=10, rs_low=1, rs_steps=50,
    logE_low=3, logE_fast=8, logE_high=12, 
    logE_steps_slow=25, logE_steps_fast=25, 
    dlnz_steps_slow=1000, dlnz_steps_fast=3e4, 
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
            save_h5_dict(save_dir+file_name, tf)
            
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
    regenerate=False, save_dir=None, file_name=None, 
    interpolated=True, tworegime=True,
    rs_high=10, rs_low=1, rs_steps=50,
    logE_low=3, logE_fast=8, logE_high=12, 
    logE_steps=25, logE_steps_slow=25, logE_steps_fast=25, 
    dlnz_steps=1000, dlnz_steps_slow=1000, dlnz_steps_fast=3e4
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
                save_file=False, return_tf=True
            )
        else:
            elec_tf = generate_lowz_electron_energy_tf(
                rs_high=rs_high, rs_low=rs_low, rs_steps=rs_steps,
                logE_low=logE_low, logE_high=logE_high, 
                logE_steps=logE_steps, dlnz_steps=dlnz_steps, 
                save_file=False, return_tf=True
            )
        save_h5_dict(save_dir+file_name, elec_tf)
    # Otherwise, load from file
    else:
        # Load transfer function
        if save_dir is None:
            save_dir = save_dir_default

        if file_name is None:
            raise ValueError("Loading electron tf, but no save file name is given")
        else:
            elec_tf = config.load_h5_dict(save_dir+file_name)

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

### FOR GENERATING COMPTON SCATTERING TABLES

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
    # Use low energy cross-section ? 
    # WQ: check that this is right when I have internet
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
                        
### FOR OBTAINING THE LOW REDSHIFT EVOLUTION OF A GIVEN ENERGY INJECTION MODEL

def evolve_elecdecay_at_lowz(
    mDM, lifetime, start_rs=4.0, end_rs=1.0, dlnz=0.008,
    save_dir=None, verbose=True,
    include_photoion=False, include_PP=False, include_positrons=False,
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

    # Load photon/electron energy arrays and compton scattering tables.
    # If the tables don't exist, build and save them.
    if save_dir is None:
        save_dir = save_dir_default

    try:
        temp = config.load_h5_dict(save_dir+"compton_scattering_phot_matrix.hdf5")
        eng_phot = temp['eng']
        cs_phot_matrix = temp['table']

        temp = config.load_h5_dict(save_dir+"compton_scattering_elec_matrix.hdf5")
        eng_elec = temp['eng']
        cs_elec_matrix = temp['table']
        # free(temp) # WQ: fix this when I have internet
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
    # WQ : check that this is right when I have internet
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
            regenerate=False, save_dir=save_dir, file_name='eng_elec_tf.hdf5', 
            interpolated=True, tworegime=False, # WQ: eventually make true
        )
    except:
        if verbose:
            print("Electron kinetic energy transfer function not found. Building it now...")
        # WQ: eventually make tworegime true and change default values
        transfer_Ee = get_electron_tf(
            regenerate=True, save_dir=save_dir, file_name='eng_elec_tf.hdf5', 
            interpolated=True, tworegime=False,
            rs_high=5, rs_low=1, rs_steps=50, dlnz_steps=int(3e4),
            logE_low=3, logE_high=12, logE_steps=25
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
            2 * dt_inj * phys.inj_rate('decay', rs_inj, mDM=mDM, lifetime=lifetime) / mDM
        ) / (phys.nB * rs_inj**3)

        # Use transfer function to get evolution of electron energy
        pts_grid = np.meshgrid([rs_inj], [Ee_init], dep_rs_list, indexing='ij')
        pts_list = np.reshape(pts_grid, (3, -1), order='C').T
        Ee_history = transfer_Ee(pts_list)

        # Find indices of electron spectrum corresponding to Ee_history
        # Add ne_inj to corresponding bins
        inds_rs = np.arange(ind_inj, ind_inj+len(dep_rs_list))
        inds_Ee = np.digitize(Ee_history, eng_elec)
        if rs_inj > end_rs:
            # WQ: which of the two indexing methods below is correct? Fix!!
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
                # WQ: add more explanation when I have internet
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

                            inds_PP_ff = np.digitize(Ee_PP_future, eng_elec)
                            inds_PP_ff[inds_PP_ff == len(eng_elec)] = 0
                            np.add.at(elec_spec_at_rs[abs_idx], inds_PP_ff, Ne_PP_vals)
                            np.add.at(sec_elec_spec[ind_dep + 1 + ff],        inds_PP_ff, Ne_PP_vals)
      
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

                        inds_Ee_fut = np.digitize(Ee_future, eng_elec)
                        inds_Ee_fut[inds_Ee_fut == len(eng_elec)] = 0
                        np.add.at(elec_spec_at_rs[abs_idx], inds_Ee_fut, sec_elec_ICS[nonzero_sec])
                        np.add.at(sec_elec_spec[ind_dep + 1 + ind_fut], inds_Ee_fut, sec_elec_ICS[nonzero_sec])

                ### Redshift and accumulate distortion
                sec_phot_spec = Spectrum(eng_phot, N_final.sum(axis=0), rs=rs_dep, spec_type='N')

        sec_phot_spec.redshift(1)
        phot_spec.dNdE += sec_phot_spec.dNdE

    if verbose:
        print("...done!")

    # Get the y-distortion
    # y_val = y_interp([mDM, lifetime])[0]
    # y_dist = phys.ymu_distortion(eng_phot, y_val, 1., 'y')
    # phot_spec.dNdE += y_dist.dNdE
    return {
        'rs' : rs_list,
        'elec_eng' : eng_elec,
        'elec_spec' : elec_spec_at_rs,
        'distortion' : phot_spec
    }

### FOR CREATING LOW REDSHIFT SPECTRAL DISTORTION TRANSFER FUNCTIONS

def generate_lowz_spectral_distortion_tf():
    return