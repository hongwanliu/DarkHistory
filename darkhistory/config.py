""" Configuration and defaults."""

import os
import numpy as np
import json
import h5py

from scipy.interpolate import interp1d, PchipInterpolator, pchip_interpolate, RegularGridInterpolator


#===== SET DATA PATH HERE =====#
# or set the environment variable DH_DATA_DIR.
data_path = None

if data_path is None and 'DH_DATA_DIR' in os.environ.keys():
    data_path = os.environ['DH_DATA_DIR']
#==============================#


# Global variables for data.
glob_binning_data   = None
glob_dep_tf_data    = None
glob_ics_tf_data    = None
glob_struct_data    = None
glob_hist_data      = None
glob_pppc_data      = None
glob_f_data         = None
glob_exc_data       = None
glob_reion_data     = None
glob_bnd_free_data  = None

class PchipInterpolator2D:

    """ 2D interpolation over PPPC4DMID raw data, using the PCHIP method.

    Parameters
    -----------
    coords_data : ndarray, size (M,N)


    values_data : ndarray
    pri : string
        Specifies primary annihilation channel. See :func:`.get_pppc_spec` for the full list.
    sec : {'elec', 'phot'}
        Specifies which secondary spectrum to obtain (electrons/positrons or photons).

    Attributes
    ----------
    pri : string
        Specifies primary annihilation channel. See :func:`.get_pppc_spec` for the full list.
    sec : {'elec', 'phot'}
        Specifies which secondary spectrum to obtain (electrons/positrons or photons).
    get_val : function
        Returns the interpolation value at (coord, value) based

    Notes
    -------
    PCHIP stands for piecewise cubic hermite interpolating polynomial. This class was built to mimic the Mathematica interpolation of the PPPC4DMID data.

    """

    def __init__(self, coords_data, values_data, pri, sec):
        if sec == 'elec':
            i = 0
            # fac is used to multiply the raw electron data by 2 to get the
            # e+e- spectrum that we always use in DarkHistory.
            fac = 2.
        elif sec == 'phot':
            i = 1
            fac = 1.
        else:
            raise TypeError('invalid final state.')

        self.pri = pri
        self.sec = sec

        # To compute the spectrum of 'e', we average over 'e_L' and 'e_R'.
        # We do the same thing for 'mu', 'tau', 'W' and 'Z'.
        # To avoid thinking too much, all spectra are split into two parts.
        # self._weight gives the weight of each half.

        if pri == 'e' or pri == 'mu' or pri == 'tau':
            pri_1 = pri + '_L'
            pri_2 = pri + '_R'
            self._weight = [0.5, 0.5]
        elif pri == 'W' or pri == 'Z':
            # 2 transverse pol., 1 longitudinal.
            pri_1 = pri + '_T'
            pri_2 = pri + '_L'
            self._weight = [2/3, 1/3]
        else:
            pri_1 = pri
            pri_2 = pri
            self._weight = [0.5, 0.5]

        idx_list_data = {
            'e_L': 0, 'e_R': 1, 'mu_L': 2, 'mu_R': 3, 'tau_L': 4, 'tau_R': 5,
            'q': 6, 'c': 7, 'b': 8, 't': 9,
            'W_L': 10, 'W_T': 11, 'Z_L': 12, 'Z_T': 13,
            'g': 14, 'gamma': 15, 'h': 16,
            'nu_e': 17, 'nu_mu': 18, 'nu_tau': 19,
            'VV_to_4e': 20, 'VV_to_4mu': 21, 'VV_to_4tau': 22
        }

        # Compile the raw data.
        mDM_in_GeV_arr_1 = np.array(
            coords_data[i, idx_list_data[pri_1], 0]
        )
        log10x_arr_1     = np.array(
            coords_data[i, idx_list_data[pri_1], 1]
        )
        values_arr_1     = np.array(values_data[i, idx_list_data[pri_1]])

        mDM_in_GeV_arr_2 = np.array(
            coords_data[i, idx_list_data[pri_2], 0]
        )
        log10x_arr_2     = np.array(
            coords_data[i, idx_list_data[pri_2], 1]
        )
        values_arr_2     = np.array(values_data[i, idx_list_data[pri_2]])

        self._mDM_in_GeV_arrs = [mDM_in_GeV_arr_1, mDM_in_GeV_arr_2]
        self._log10x_arrs     = [log10x_arr_1,     log10x_arr_2]

        # Save the 1D PCHIP interpolator over mDM_in_GeV. Multiply the
        # electron spectrum by 2 by adding np.log10(2).
        self._interpolators = [
            PchipInterpolator(
                mDM_in_GeV_arr_1, values_arr_1 + np.log10(fac),
                extrapolate=False
            ),
            PchipInterpolator(
                mDM_in_GeV_arr_2, values_arr_2 + np.log10(fac),
                extrapolate=False
            )
        ]

    def get_val(self, mDM_in_GeV, log10x):

        if (
            mDM_in_GeV < self._mDM_in_GeV_arrs[0][0]
            or mDM_in_GeV < self._mDM_in_GeV_arrs[1][0]
            or mDM_in_GeV > self._mDM_in_GeV_arrs[0][-1]
            or mDM_in_GeV > self._mDM_in_GeV_arrs[1][-1]
        ):
            raise TypeError('mDM lies outside of the interpolation range.')

        # Call the saved interpolator at mDM_in_GeV,
        # then use PCHIP 1D interpolation at log10x.
        result1 = pchip_interpolate(
            self._log10x_arrs[0], self._interpolators[0](mDM_in_GeV), log10x
        )
        # Set all values outside of the log10x interpolation range to
        # (effectively) zero.
        result1[log10x >= self._log10x_arrs[0][-1]] = -100.
        result1[log10x <= self._log10x_arrs[0][0]]  = -100.

        result2 = pchip_interpolate(
            self._log10x_arrs[1], self._interpolators[1](mDM_in_GeV), log10x
        )
        result2[log10x >= self._log10x_arrs[1][-1]] = -100.
        result2[log10x <= self._log10x_arrs[1][0]]  = -100.

        # Combine the two spectra.
        return np.log10(
            self._weight[0]*10**result1 + self._weight[1]*10**result2
        )

def load_h5_dict(file_path):
    def recursive_load(h5_obj):
        data_dict = {}
        for key, item in h5_obj.items():
            if isinstance(item, h5py.Group):
                data_dict[key] = recursive_load(item)
            elif isinstance(item, h5py.Dataset):
                data_dict[key] = item[()]
        return data_dict

    with h5py.File(file_path, 'r') as h5_file:
        return recursive_load(h5_file)

def load_data(data_type, verbose=1):
    """ Loads data from downloaded files.

    Parameters
    ----------
    data_type : {'binning', 'dep_tf', 'ics_tf', 'struct', 'hist', 'f', 'pppc', 'exc'}
        Type of data to load. The options are:

        - *'binning'* -- Default binning for all transfer functions;

        - *'dep_tf'* -- Transfer functions for propagating photons and deposition into low-energy photons, low-energy electrons, high-energy deposition and upscattered CMB energy rate;

        - *'ics_tf'* -- Transfer functions for ICS for scattered photons in the Thomson regime, relativistic regime, and scattered electron energy-loss spectrum;

        - *'struct'* -- Structure formation boosts;

        - *'hist'* -- Baseline ionization and temperature histories;

        - *'f'* -- :math:`f_c(z)` fractions without backreaction; and

        - *'pppc'* -- Data from PPPC4DMID for annihilation spectra. Specify the primary channel in *primary*.

        - *'exc'* -- cross-sections for e- H(1s) -> e- H(2s) or e- H(np) where n is within 2 through 10.

    verbose : {0, 1}
        Set verbosity.

    Returns
    --------
    dict
        A dictionary of the data requested.

    See Also
    ---------
    :func:`.get_pppc_spec`

    """

    global data_path

    global glob_binning_data, glob_dep_tf_data, glob_ics_tf_data
    global glob_struct_data,  glob_hist_data, glob_f_data, glob_pppc_data, glob_exc_data, glob_reion_data
    global glob_bnd_free_data

    if data_path is None or not os.path.isdir(data_path):
        raise ValueError('Please set data directory in darkhistory.config or to `DH_DATA_DIR` environment variable.')

    if data_type == 'binning':
        if glob_binning_data is None:
            try:
                glob_binning_data = load_h5_dict(data_path+'/binning.h5')
            except FileNotFoundError as err:
                print(type(err).__name__, ':', err)
                raise FileNotFoundError('Please update your dataset! See README.md for instructions.')
        return glob_binning_data

    elif data_type == 'dep_tf':
        from darkhistory.spec.transferfunclist import TransferFuncInterp
        from darkhistory.history.histools import IonRSInterp
        # prevent Spectrum -> physics -> load_data -> TransferFuncInterp -> Spectrum ciruclar import
        if glob_dep_tf_data is None:
            if verbose >= 1:
                print('****** Loading transfer functions... ******')
                print(f'Using data at {data_path}')
                print('    for propagating photons... ', end =' ', flush=True)
            highengphot_tf_interp = TransferFuncInterp(load_h5_dict(data_path+'/highengphot.h5'))
            if verbose >= 1:
                print(' Done!')
                print('    for low-energy photons... ', end=' ', flush=True)
            lowengphot_tf_interp  = TransferFuncInterp(load_h5_dict(data_path+'/lowengphot.h5'))
            if verbose >= 1:
                print('Done!')
                print('    for low-energy electrons... ', end=' ', flush=True)
            lowengelec_tf_interp  = TransferFuncInterp(load_h5_dict(data_path+'/lowengelec.h5'))
            if verbose >= 1:
                print('Done!')
                print('    for high-energy deposition... ', end=' ', flush=True)
            highengdep_interp     = IonRSInterp(load_h5_dict(data_path+'/highengdep.h5'))
            if verbose >= 1:
                print('Done!')
                print('    for total upscattered CMB energy rate... ', end=' ', flush=True)
            CMB_engloss_interp    = IonRSInterp(load_h5_dict(data_path+'/CMB_engloss.h5'))
            if verbose >= 1:
                print('Done!')
                print('****** Loading complete! ******', flush=True)

            glob_dep_tf_data = {
                'highengphot' : highengphot_tf_interp,
                'lowengphot'  : lowengphot_tf_interp,
                'lowengelec'  : lowengelec_tf_interp,
                'highengdep'  : highengdep_interp,
                'CMB_engloss' : CMB_engloss_interp
            }
        return glob_dep_tf_data

    elif data_type == 'ics_tf':
        from darkhistory.spec.transferfunction import TransFuncAtRedshift
        if glob_ics_tf_data is None:
            if verbose >= 1:
                print('****** Loading transfer functions... ******')
                print('    for inverse Compton (Thomson)... ', end=' ', flush=True)
            ics_thomson_ref_tf = TransFuncAtRedshift(load_h5_dict(data_path+'/ics_thomson_ref.h5'))
            if verbose >= 1:
                print('Done!')
                print('    for inverse Compton (relativistic)... ', end=' ', flush=True)
            ics_rel_ref_tf     = TransFuncAtRedshift(load_h5_dict(data_path+'/ics_rel_ref.h5'))
            if verbose >= 1:
                print('Done!')
                print('    for inverse Compton (energy loss)... ', end=' ', flush=True)
            engloss_ref_tf     = TransFuncAtRedshift(load_h5_dict(data_path+'/ics_engloss_ref.h5'))
            if verbose >= 1:
                print('Done!')
                print('****** Loading complete! ******')
            glob_ics_tf_data = {
                'thomson' : ics_thomson_ref_tf,
                'rel'     : ics_rel_ref_tf,
                'engloss' : engloss_ref_tf
            }
        return glob_ics_tf_data

    elif data_type == 'struct':
        if glob_struct_data is None:
            boost_data = np.loadtxt(data_path+'/boost_data.txt')
            #einasto_subs = np.loadtxt(open(data_path+'/boost_Einasto_subs.txt', 'rb'))
            glob_struct_data = {
                'einasto_subs'    : boost_data[:,[0,1]],
                'einasto_no_subs' : boost_data[:,[0,2]],
                'NFW_subs'        : boost_data[:,[0,3]],
                'NFW_no_subs'     : boost_data[:,[0,4]]
            }
        return glob_struct_data

    elif data_type == 'hist':
        if glob_hist_data is None:
            glob_hist_data = load_h5_dict(data_path+'/std_soln_He.h5')
        return glob_hist_data

    elif data_type == 'f':
        if glob_f_data is None:

            ln_rs = np.array([np.log(3000) - 0.001*i for i in np.arange(6620)])
            ln_rs_phot_pwave = np.array([np.log(3000) - 0.004*i for i in np.arange(1655)])
            ln_rs_elec_pwave = np.array([np.log(3000) - 0.032*i for i in np.arange(207)])
            def get_rs_arr(label):
                if   label == 'phot_pwave_NFW':
                    return ln_rs_phot_pwave
                elif label == 'elec_pwave_NFW':
                    return ln_rs_elec_pwave
                else:
                    return ln_rs

            log10eng0 = 3.6989700794219966
            log10eng = np.array([log10eng0 + 0.23252559*i for i in np.arange(40)])
            log10eng[-1] = 12.601505994846297

            labels = ['phot_decay', 'elec_decay',
              'phot_swave_noStruct', 'elec_swave_noStruct',
              'phot_swave_einasto', 'elec_swave_einasto',
              'phot_swave_NFW', 'elec_swave_NFW',
              'phot_pwave_NFW', 'elec_pwave_NFW']

            f_data = load_h5_dict(data_path+'/f_std_with_pwave_09_19_2019.h5')

            glob_f_data = {label : RegularGridInterpolator(
                (log10eng, np.flipud(get_rs_arr(label))), np.flip(np.log(f_data[label]),1)
                ) for label in labels}


        return glob_f_data

    elif data_type == 'pppc':
        if glob_pppc_data is None:

            coords_data = np.array(json.load(open(data_path+'/dlNdlxIEW_coords_table.json')), dtype=object)
            # coords_data is a (2, 23, 2) array.
            # axis 0: stable SM secondaries, {'elec', 'phot'}
            # axis 1: annihilation primary channel.
            # axis 2: {mDM in GeV, np.log10(K/mDM)}, K is the energy of
            # the secondary.
            # Each element is a 1D array.

            values_data = np.array(json.load(open(data_path+'/dlNdlxIEW_values_table.json')), dtype=object)
            # values_data is a (2, 23) array, d log_10 N / d log_10 (K/mDM).
            # axis 0: stable SM secondaries, {'elec', 'phot'}
            # axis 1: annihilation primary channel.
            # Each element is a 2D array indexed by {mDM in GeV, np.log10(K/mDM)}
            # as saved in coords_data.

            # Compile a dictionary of all of the interpolators.
            dlNdlxIEW_interp = {'elec':{}, 'phot':{}}

            chan_list = [
                'e_L','e_R', 'e', 'mu_L', 'mu_R', 'mu',
                'tau_L', 'tau_R', 'tau',
                'q',  'c',  'b', 't',
                'W_L', 'W_T', 'W', 'Z_L', 'Z_T', 'Z', 'g',  'gamma', 'h',
                'nu_e', 'nu_mu', 'nu_tau',
                'VV_to_4e', 'VV_to_4mu', 'VV_to_4tau'
            ]

            for pri in chan_list:
                dlNdlxIEW_interp['elec'][pri] = PchipInterpolator2D(
                    coords_data, values_data, pri, 'elec'
                )
                dlNdlxIEW_interp['phot'][pri] = PchipInterpolator2D(
                    coords_data, values_data, pri, 'phot'
                )

            glob_pppc_data = dlNdlxIEW_interp

        return glob_pppc_data

    elif data_type == 'exc':
        if glob_exc_data == None:
            species_list = ['HI', 'HeI']
            state_list = [
                '2s', '2p',
                '3s', '3p', '3d',
                '4s', '4p', '4d', '4f',
                '5p', '6p', '7p', '8p', '9p', '10p'
            ]

            KimRudd_list = ['2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p', '10p']
            KimRudd_data = {'HI': load_h5_dict(data_path+'/H_exc_xsec_data.h5'),
                    'HeI': load_h5_dict(data_path+'/He_exc_xsec_data.h5')
                    }

            CCC_states = ['2s','3s','3d','4s','4d','4f']
            CCC_data = load_h5_dict(data_path+'/H_exc_xsec_data_CCC.h5')


            def make_interpolator(species,state):
                if (species=='HI') and (state in CCC_states):
                    x = CCC_data['eng']
                    y = CCC_data[state]
                elif not (state in CCC_states):
                    x = KimRudd_data[species]['eng_'+state[-1]]
                    # The cross-section is currently in units of Angstrom^2
                    y = KimRudd_data[species][state]*1e-16
                else:
                    x,y = None,None

                if (x is None) or (y is None):
                    return None
                else:
                    return interp1d(x,y, kind='linear', bounds_error=False, fill_value=(0,0))

            glob_exc_data = {species:
                {state : make_interpolator(species, state) for state in state_list}
            for species in species_list}

        return glob_exc_data

    elif data_type == 'exc_AcharyaKhatri':
        if glob_exc_data == None:
            #CCC cross-sections in units of cm^2
            species_list = ['HI', 'HeI']
            exc_data = {'HI': load_h5_dict(data_path+'/H_exc_xsec_data_CCC.h5'),
                    'HeI': load_h5_dict(data_path+'/He_exc_xsec_data_CCC.h5')
                    }

            state_list = ['2s', '2p', '3p']

            def make_interpolator(x,y):
                if (x is None) or (y is None):
                    return None
                else:
                    return interp1d(x,y, kind='linear', bounds_error=False, fill_value=(0,0))

            glob_exc_data = {species:
                {state : make_interpolator(exc_data[species]['eng_'+state[-1]], exc_data[species][state])
                for state in state_list}
            for species in species_list}

        return glob_exc_data

    elif data_type == 'reion':
        raise NotImplementedError('pickle files need to be updated.')
        if glob_reion_data == None:
            glob_reion_data = pickle.load(open(data_path+'/Onorbe_data.p','rb'))

        return glob_reion_data

    elif data_type == 'bnd_free':
        if glob_bnd_free_data == None:

            glob_bnd_free_data = {}

            # Contains a pre-computed dictionary indexed by [n][l][lp] of g values,
            # using the generate_g_table_dict function at the end of this module. See arXiv:0911.1359 Eq. (30) for definition.
            glob_bnd_free_data['g_table_dict']  = load_h5_dict(data_path+'/g_table_dict.h5')['g_table_dict']

            # Number of log-spaced large bins for kappa^2 = E_e / R, where E_e is the electron energy and R is the
            # ionization potential of hydrogen.

            glob_bnd_free_data['n_kap'] = 50

            # Generate the abscissa for kappa^2 and the spacing at each point for integration later. We will compute
            # coefficients up to n = 300 of the hydrogen atom.

            # first axis for n = 300 hydrogen states, second axis subdivides each large bin above
            # into 10 equally spaced intervals in kappa^2.
            glob_bnd_free_data['kappa2_bin_edges_ary'] = np.zeros((301,11*glob_bnd_free_data['n_kap']))
            glob_bnd_free_data['h_ary'] = np.zeros((301,11*glob_bnd_free_data['n_kap']))

            # Fill the arrays accordingly.
            for n in 1 + np.arange(300):

                # Using the same boundaries as arXiv:0911.1359. However, we integrate over kappa^2.
                kappa2_big_bin_edges = np.logspace(
                    np.log10(1e-25/n**2), np.log10(4.96e8/n**2),
                    num=glob_bnd_free_data['n_kap']+1
                )

                for i,_ in enumerate(kappa2_big_bin_edges[:-1]):

                    low = kappa2_big_bin_edges[i]
                    upp = kappa2_big_bin_edges[i+1]
                    abscissa = np.linspace(low, upp, num=11)
                    glob_bnd_free_data['kappa2_bin_edges_ary'][n, 11*i:11*(i+1)] = abscissa
                    glob_bnd_free_data['h_ary'][n, 11*i:11*(i+1)] = np.ones(11) * (abscissa[1] - abscissa[0])

        return glob_bnd_free_data

    else:
        raise ValueError('invalid data_type.')


def test(date_str=None, end_rs=4, iter=2, std_only=False):
    """
    Runs a quick unit test of the code using some reference files.

    Parameters
    ----------
    date_str : string, optional
        A string for the date of the reference file.
    end_rs : float, optional
        Redshift :math:`1+z` to end the test at.
    iter : int, optional
        Number of iterations.
    std_only : bool, optional
        If *True*, runs only the case with no exotic injection.

    Returns
    -------
    None

    """

    import darkhistory.main as main

    if date_str is None:
        std_file_str = data_path+'/reference_20220822_std_result_n_10_high_rs_1555_coarsen_16_reion_False_rtol_1e-6_iter_'+str(iter)+'.p'

        DM_file_str = data_path+'/reference_20220822_mDM_1e8_elec_delta_decay_3e25_n_10_high_rs_1555_coarsen_16_reion_True_rtol_1e-6_iter_'+str(iter)+'.p'

    else:
        std_file_str = data_path+'/reference_'+date_str+'_std_result_n_10_high_rs_1555_coarsen_16_reion_False_rtol_1e-6_iter_'+str(iter)+'.p'
        DM_file_str = data_path+'/reference_'+date_str+'_mDM_1e8_elec_delta_decay_3e25_n_10_high_rs_1555_coarsen_16_reion_True_rtol_1e-6_iter_'+str(iter)+'.p'

    std_file_data = pickle.load(open(std_file_str, 'rb'))
    DM_file_data = pickle.load(open(DM_file_str, 'rb'))

    DM_options_dict = {
        'primary':'elec_delta', 'DM_process':'decay', 'mDM':1e8, 'lifetime':3e25,
        'start_rs': 3000, 'high_rs': 1.555e3, 'end_rs':end_rs,
        'reion_switch':True, 'reion_method':'Puchwein', 'heat_switch':True,
        'coarsen_factor':16, 'distort':True, 'fexc_switch': True,
        'MLA_funcs':None,
        'reprocess_distortion':True, 'nmax':10, 'rtol':1e-6, 'use_tqdm':True, 'iterations':iter
    }

    std_options_dict = {
        'primary':'elec_delta', 'DM_process':'decay', 'mDM':1e8, 'lifetime':3e40,
        'start_rs': 3000, 'high_rs': 1.555e3, 'end_rs':end_rs,
        'reion_switch':False, 'reion_method':'Puchwein', 'heat_switch':True,
        'coarsen_factor':16, 'distort':True, 'fexc_switch': True,
        'MLA_funcs':None,
        'reprocess_distortion':True, 'nmax':10, 'rtol':1e-6, 'use_tqdm':True, 'iterations':iter
    }

    print('Running main.evolve(...): ')

    try:

        print('******************************************')
        print('Testing solution with no DM: ')

        std_res = main.evolve(**std_options_dict)

        def max_rel_change(new_res, ref_res):
            if end_rs != 4:
                ref_res = ref_res[:len(new_res)]
            return np.nanmax(
                np.abs(
                    np.divide(
                        new_res, ref_res, out=np.ones_like(new_res)*np.nan,
                        where=(ref_res != 0)
                    ) - 1
                )
            )

        print(
            'The maximum relative change in xHI and xHeI is: ',
            max_rel_change(std_res[-1]['x'], std_file_data[-1]['x'])
        )
        print(
            'The maximum relative change in Tm is: ',
            max_rel_change(std_res[-1]['Tm'], std_file_data[-1]['Tm'])
        )
        print(
            'The maximum relative change in f_(H ion) is: ',
            max_rel_change(std_res[-1]['f']['H ion'], std_file_data[-1]['f']['H ion'])
        )
        print(
            'The maximum relative change in f_(H ion) is: ',
            max_rel_change(std_res[-1]['f']['H ion'], std_file_data[-1]['f']['H ion'])
        )
        print(
            'The maximum relative change in f_(He ion) is: ',
            max_rel_change(std_res[-1]['f']['He ion'], std_file_data[-1]['f']['He ion'])
        )
        print(
            'The maximum relative change in f_(Lya) is: ',
            max_rel_change(std_res[-1]['f']['Lya'], std_file_data[-1]['f']['Lya'])
        )
        print(
            'The maximum relative change in f_(heat) is: ',
            max_rel_change(std_res[-1]['f']['heat'], std_file_data[-1]['f']['heat'])
        )
        print(
            'The maximum relative change in f_(cont) is: ',
            max_rel_change(std_res[-1]['f']['cont'], std_file_data[-1]['f']['cont'])
        )
        print(
            'The maximum relative change in the MLA parameters is: ', max_rel_change(np.transpose(std_res[-1]['MLA'][1:]), np.transpose(std_file_data[-1]['MLA'][1:]))
        )

        pickle.dump(std_res, open(data_path+'/std_test_data.p', 'wb'))

        print('Pickled solution with no DM!')

        if not std_only:

            DM_res = main.evolve(**DM_options_dict)



            print('******************************************')
            print('Testing solution with DM: ')

            print(
                'The maximum relative change in xHI and xHeI is: ',
                max_rel_change(DM_res[-1]['x'], DM_file_data[-1]['x'])
            )
            print(
                'The maximum relative change in Tm is: ',
                max_rel_change(DM_res[-1]['Tm'], DM_file_data[-1]['Tm'])
            )
            print(
                'The maximum relative change in f_(H ion) is: ',
                max_rel_change(DM_res[-1]['f']['H ion'], DM_file_data[-1]['f']['H ion'])
            )
            print(
                'The maximum relative change in f_(H ion) is: ',
                max_rel_change(DM_res[-1]['f']['H ion'], DM_file_data[-1]['f']['H ion'])
            )
            print(
                'The maximum relative change in f_(He ion) is: ',
                max_rel_change(DM_res[-1]['f']['He ion'], DM_file_data[-1]['f']['He ion'])
            )
            print(
                'The maximum relative change in f_(Lya) is: ',
                max_rel_change(DM_res[-1]['f']['Lya'], DM_file_data[-1]['f']['Lya'])
            )
            print(
                'The maximum relative change in f_(heat) is: ',
                max_rel_change(DM_res[-1]['f']['heat'], DM_file_data[-1]['f']['heat'])
            )
            print(
                'The maximum relative change in f_(cont) is: ',
                max_rel_change(DM_res[-1]['f']['cont'], DM_file_data[-1]['f']['cont'])
            )
            print(
                'The maximum relative change in the MLA parameters is: ', max_rel_change(np.transpose(DM_res[-1]['MLA'][1:]), np.transpose(DM_file_data[-1]['MLA'][1:]))
            )

            pickle.dump(DM_res, open(data_path+'/DM_test_data.p', 'wb'))

            print('Pickled solution with DM!')

        print('Test complete!')

    except:

        raise RuntimeError('main.evolve(...) failed to complete.')

    return None
