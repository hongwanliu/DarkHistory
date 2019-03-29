""" PPPC4DMID [1]_ [2]_ Information and Functions. 

"""

import sys
import numpy as np
import json

from scipy.interpolate import PchipInterpolator
from scipy.interpolate import pchip_interpolate

from config import data_path

import darkhistory.physics as phys
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec.spectools import rebin_N_arr

# PPPC data.  

# Import data. 
coords_file_name = (
    data_path+'/dlNdlxIEW_coords_table.txt'
)
values_file_name = (
    data_path+'/dlNdlxIEW_values_table.txt'
)

if not 'pytest' in sys.modules: 

    with open(coords_file_name) as data_file:    
        coords_data = np.array(json.load(data_file))
    with open(values_file_name) as data_file:
        values_data = np.array(json.load(data_file))


# coords_data is a (2, 23, 2) array. 
# axis 0: stable SM secondaries, {'elec', 'phot'}
# axis 1: annihilation primary channel, given by chan_list_data below. 
# axis 2: {mDM in GeV, np.log10(K/mDM)} where K is the energy of 
# the secondary. 
# Each element is a 1D array.

# values_data is a (2, 23) array, storing d log_10 N / d log_10 (K/mDM). 
# axis 0: stable SM secondaries, {'elec', 'phot'}
# axis 1: annihilation primary channel, given by chan_list_data below. 
# Each element is a 2D array, indexed by {mDM in GeV, np.log10(K/mDM)}
# as saved in coords_data. 

# chan_list_data = [
#     'e_L', 'e_R', 'mu_L', 'mu_R', 'tau_L', 'tau_R',
#     'q',  'c',  'b', 't',
#     'W_L', 'W_T', 'Z_L', 'Z_T', 
#     'g',  'gamma', 'h',
#     'nu_e', 'nu_mu', 'nu_tau',
#     'VV_to_4e', 'VV_to_4mu', 'VV_to_4tau'
# ]

idx_list_data = {
    'e_L': 0, 'e_R': 1, 'mu_L': 2, 'mu_R': 3, 'tau_L': 4, 'tau_R': 5,
    'q': 6, 'c': 7, 'b': 8, 't': 9,
    'W_L': 10, 'W_T': 11, 'Z_L': 12, 'Z_T': 13, 
    'g': 14, 'gamma': 15, 'h': 16,
    'nu_e': 17, 'nu_mu': 18, 'nu_tau': 19,
    'VV_to_4e': 20, 'VV_to_4mu': 21, 'VV_to_4tau': 22
} 

class PchipInterpolator2D: 

    """ 2D interpolation over the raw data, using PCHIP method.

    Parameters
    -----------
    pri : string
        Specifies primary annihilation channel. See below for list.
    sec : {'elec', 'phot'}
        Specifies stable SM secondary to get the spectrum of.

    Attributes
    ----------
    pri : string
        Specifies primary annihilation channel. See below for list.
    sec : {'elec', 'phot'}
        Specifies stable SM secondary to get the spectrum of.
    get_val : function
        Returns the interpolation value at (mDM_in_GeV, log10(K/mDM)) where K is the kinetic energy of the secondary.

    """
    
    def __init__(self, pri, sec):
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

# This list includes 'e', 'mu', 'tau', 'W' and 'Z'. 
chan_list = [
    'e_L','e_R', 'e', 'mu_L', 'mu_R', 'mu', 'tau_L', 'tau_R', 'tau',
    'q',  'c',  'b', 't',
    'W_L', 'W_T', 'W', 'Z_L', 'Z_T', 'Z', 'g',  'gamma', 'h',
    'nu_e', 'nu_mu', 'nu_tau',
    'VV_to_4e', 'VV_to_4mu', 'VV_to_4tau'
]

# Mass threshold for mDM to annihilate into the primaries.
mass_threshold = {
    'elec_delta': phys.mass['e'], 'phot_delta': 0.,
    'e_L'   : phys.mass['e'],   'e_R': phys.mass['e'], 
    'e': phys.mass['e'],
    'mu_L'  : phys.mass['mu'], 'mu_R': phys.mass['mu'], 
    'mu': phys.mass['mu'],
    'tau_L' : phys.mass['tau'], 'tau_R' : phys.mass['tau'], 
    'tau': phys.mass['tau'],
    'q'     : 0.,             'c': phys.mass['c'],   
    'b'     : phys.mass['b'], 't': phys.mass['t'],
    'W_L': phys.mass['W'], 'W_T'   : phys.mass['W'], 'W': phys.mass['W'],
    'Z_L': phys.mass['Z'], 'Z_T'   : phys.mass['Z'], 'Z': phys.mass['Z'],
    'g': 0., 'gamma' : 0., 'h': phys.mass['h'],
    'nu_e': 0., 'nu_mu' : 0., 'nu_tau': 0.,
    'VV_to_4e'   : 2*phys.mass['e'], 'VV_to_4mu' : 2*phys.mass['mu'], 
    'VV_to_4tau' : 2*phys.mass['tau']
}

# Compile a dictionary of all of the interpolators.
dlNdlxIEW_interp = {'elec':{}, 'phot':{}}

if not 'pytest' in sys.modules: 
    for pri in chan_list:
        dlNdlxIEW_interp['elec'][pri] = PchipInterpolator2D(pri, 'elec')
        dlNdlxIEW_interp['phot'][pri] = PchipInterpolator2D(pri, 'phot')

def get_pppc_spec(mDM, eng, pri, sec, decay=False):

    """ Returns the PPPC4DMID spectrum. 

    This is the secondary spectrum to e+e-/photons normalized to one annihilation or decay event to the species specified in ``pri``. These results include electroweak corrections. The full list of allowed channels is: 

    - :math:`\\delta`\ -function injections: ``elec_delta, phot_delta``
    - Leptons: ``e_L, e_R, e, mu_L, mu_R, mu, tau_L, tau_R, tau``
    - Quarks:  ``q, c, b, t``
    - Gauge bosons: ``gamma, g, W_L, W_T, W, Z_L, Z_T, Z``
    - Higgs: ``h``

    ``elec_delta`` and ``phot_delta`` assumes annihilation or decay to two electrons and photons respectively with no EW corrections or ISR/FSR. 

    Variables with subscripts, e.g. ``e_L``, correspond to particles with different polarizations. These polarizations are suitably averaged to obtain the spectra returned in their corresponding variables without subscripts, e.g. ``e``. 

    Parameters
    ----------
    mDM : float
        The mass of the annihilating dark matter particle (in eV). 
    eng : ndarray
        The energy abscissa for the output spectrum (in eV). 
    pri : string
        One of the available channels (see Notes). 
    sec : {'elec', 'phot'}
        The secondary spectrum to obtain. 
    decay : bool, optional
        If ``True``, returns the result for decays.

    Returns
    -------
    Spectrum
        Output :class:`.Spectrum` object, ``spec_type == 'dNdE'``.
    
    """

    if decay:
        # Primary energies is for 1 GeV decay = 0.5 GeV annihilation.
        _mDM = mDM/2.
    else:
        _mDM = mDM

    if _mDM < mass_threshold[pri]:
        # This avoids the absurd situation where mDM is less than the 
        # threshold but we get a nonzero spectrum due to interpolation.
        raise ValueError('mDM is below the threshold to produce pri particles.')

    if pri == 'elec_delta':
        # Exact kinetic energy of each electron.
        if not decay:
            eng_elec = mDM - phys.me
        else:
            eng_elec = (mDM - 2*phys.me)/2
        # Find the correct bin in eleceng.
        eng_to_inj = eng[eng < eng_elec][-1]
        # Place 2*eng_elec worth of electrons into that bin. Use
        # rebinning to accomplish this.

        if sec == 'elec':
            return rebin_N_arr(
                np.array([2 * eng_elec / eng_to_inj]), 
                np.array([eng_to_inj]), 
                eng
            )

        elif sec == 'phot':
            return Spectrum(
                eng, np.zeros_like(eng), spec_type='dNdE'
            )

        else:
            raise ValueError('invalid sec.')

    if pri == 'phot_delta':
        # Exact kinetic energy of each photon. 
        if not decay:
            eng_phot = mDM
        else:
            eng_phot = mDM/2
        # Find the correct bin in photeng.
        eng_to_inj = eng[eng < eng_phot][-1]
        # Place 2*eng_phot worth of photons into that bin. Use
        # rebinning to accomplish this. 
        if sec == 'elec':
            return Spectrum(
                eng, np.zeros_like(eng), spec_type='dNdE'
            )

        elif sec == 'phot':
            return rebin_N_arr(
                np.array([2 * eng_phot / eng_to_inj]), 
                np.array([eng_to_inj]), 
                eng
            )

        else:
            raise ValueError('invalid sec.')
    
    log10x = np.log10(eng/_mDM)

    # Refine the binning so that the spectrum is accurate. 
    # Do this by checking that in the relevant range, there are at
    # least 50,000 bins. If not, double (unless an absurd number
    # of bins already). 

    if (
        log10x[(log10x < 1) & (log10x > 1e-9)].size > 0
        and log10x.size < 500000
    ):
        while log10x[(log10x < 1) & (log10x > 1e-9)].size < 50000:
            log10x = np.interp(
                np.arange(0, log10x.size-0.5, 0.5), 
                np.arange(log10x.size), 
                log10x
            )
    
    # Get the spectrum for the interpolator.
    dN_dlog10x = 10**dlNdlxIEW_interp[sec][pri].get_val(_mDM/1e9, log10x)
    
    # Recall that dN/dE = dN/dlog10x * dlog10x/dE
    x = 10**log10x
    spec = Spectrum(x*_mDM, dN_dlog10x/(x*_mDM*np.log(10)), spec_type='dNdE')
    
    # Rebin down to the original binning.

    # The highest bin of spec.eng should be the same as eng[-1], based on
    # the interpolation strategy above. However, sometimes a floating point
    # error is picked up. We'll get rid of this so that rebin doesn't
    # complain.
    spec.eng[-1] = eng[-1]
    spec.rebin(eng)
        
    return spec


