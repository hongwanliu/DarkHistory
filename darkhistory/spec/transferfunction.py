"""``transferfunction`` contains functions and classes for processing transfer functions."""

import numpy as np
from darkhistory import physics as phys
from darkhistory import utilities as utils
from darkhistory.spec import spectrum
from darkhistory.spec import spectra
from scipy import interpolate

from astropy.io import fits 
from tqdm import tqdm 

class TransferFunction(spectra.Spectra):
    """Collection of ``Spectrum`` objects for transfer functions.

    Parameters
    ---------- 
    spec_arr : list of ``Spectrum``
        List of ``Spectrum`` to be stored together.
    in_eng : float
        Injection energy of this transfer function. 
    rebin_eng : ndarray, optional
        New abscissa to rebin all of the ``Spectrum`` objects into. 

    Attributes
    ----------
    eng : ndarray
        Energy abscissa for the ``Spectrum``.
    rs : ndarray
        The redshifts of the ``Spectrum`` objects.
    grid_values : ndarray
        2D array with the spectra laid out in (rs, eng). 
    

    """
    def __init__(self, spec_arr, in_eng, rebin_eng=None):
        spectra.Spectra.__init__(self, spec_arr, rebin_eng)
        self.in_eng = in_eng

    def __iter__(self):
        return iter(self.spec_arr)

    def __getitem__(self,key):
        if np.issubdtype(type(key), int) or isinstance(key, slice):
            return self.spec_arr[key]
        else:
            raise TypeError("index must be int.")

    def __setitem__(self,key,value):
        if isinstance(key, int):
            if not isinstance(value, (list, tuple)):
                if np.issubclass_(type(value), Spectrum):
                    self.spec_arr[key] = value
                else:
                    raise TypeError("can only add Spectrum.")
            else:
                raise TypeError("can only add one spectrum per index.")
        elif isinstance(key, slice):
            if len(self.spec_arr[key]) == len(value):
                for i,spec in zip(key,value): 
                    if np.issubclass_(type(spec), Spectrum):
                        self.spec_arr[i] = spec
                    else: 
                        raise TypeError("can only add Spectrum.")
            else:
                raise TypeError("can only add one spectrum per index.")
        else:
            raise TypeError("index must be int.")


    def __add__(self, other): 
        
        if np.issubclass_(type(other), TransferFunction):

            if not util.array_equal(self.eng, other.eng):
                raise TypeError('abscissae are different for the two TransferFunction.')
            if not util.array_equal(self.rs, other.rs):
                raise TypeError('redshifts are different for the two TransferFunction.')
            if not self.in_eng == other.in_eng:
                raise TypeError('injection energies are different \
                    for the two TransferFunction.')

            return TransferFunction([spec1 + spec2 for spec1,spec2 in zip(self.spec_arr, other.spec_arr)], self.in_eng)

        else: raise TypeError('adding an object that is not of class TransferFunction.')


    def __radd__(self, other): 
        
        if np.issubclass_(type(other), TransferFunction):

            if not util.array_equal(self.eng, other.eng):
                raise TypeError('abscissae are different for the \
                    two TransferFunction.')
            if not util.array_equal(self.rs, other.rs):
                raise TypeError('redshifts are different for the \
                    two TransferFunction.')
            if self.in_eng != other.in_eng:
                raise TypeError('injection energies are different \
                    for the two TransferFunction.')

            return TransferFunction([spec1 + spec2 for spec1,spec2 in zip(
                self.spec_arr, other.spec_arr)], self.in_eng)

        else: raise TypeError('adding an object that is not of \
                    class TransferFunction.')

    def __sub__(self, other):
        
        return self + -1*other 

    def __rsub__(self, other):
          
        return other + -1*self

    def __neg__(self):
        
        return -1*self

    def __mul__(self, other):
       
        if (np.issubdtype(type(other), float) 
            or np.issubdtype(type(other), int)):
            return TransferFunction([other*spec for spec in self], 
                self.in_eng)
        elif np.issubclass_(type(other), TransferFunction):
            if self.rs != other.rs or self.eng != other.eng:
                raise TypeError("the two spectra do not have the same redshift or abscissae.")
            if self.in_eng != other.in_eng:
                raise TypeError('injection energies are different \
                    for the two TransferFunction.')
            return TransferFunction([spec1*spec2 
                for spec1,spec2 in zip(self, other)], self.in_eng)
        else:
            raise TypeError("can only multiply TransferFunction or scalars.")

    def __rmul__(self, other):
        
        if (np.issubdtype(type(other), float) 
            or np.issubdtype(type(other), int)):
            return TransferFunction([other*spec for spec in self])
        elif np.issubclass_(type(other), TransferFunction):
            if self.rs != other.rs or self.eng != other.eng:
                raise TypeError("the two spectra do not have the \
                    same redshift or abscissae.")
            if self.in_eng != other.in_eng:
                raise TypeError('injection energies are different \
                    for the two TransferFunction.')
            return TransferFunction([spec2*spec1 
                for spec1,spec2 in zip(self, other)], self.in_eng)
        else:
            raise TypeError("can only multiply TransferFunction or scalars.")

    def __truediv__(self,other):
        
        if np.issubclass_(type(other), TransferFunction):
            invSpec = TransferFunction([1./spec for spec in other])
            return self*invSpec
        else:
            return self*(1/other)

    def __rtruediv__(self,other):
        
        invSpec = TransferFunction([1./spec for spec in self])

        return other*invSpec

    def at_rs(self, out_rs, interp_type='val'):
        """Returns the interpolation spectrum at a given redshift.

        Interpolation is logarithmic.

        Parameters
        ----------
            out_rs : ndarray
                The redshifts (or redshift bin indices) at which to interpolate. 
            interp_type : {'val', 'bin'}
                The type of interpolation. 'bin' uses bin index, while 'val' uses the actual redshift. 

        Returns
        -------
        TransferFunction
            The interpolated spectra. 
        """

        interp = interpolate.interp2d(self.eng, np.log(self.rs), 
            self.grid_values)

        if interp_type == 'val':
            return TransferFunction(
                [spectrum.Spectrum(self.eng, interp(self.eng, np.log(rs)), rs) for rs in out_rs], self.in_eng
                )
        elif interp_type == 'bin':
            log_rs_value = np.interp(out_rs, np.arange(self.rs.size), np.log(self.rs))
            return TransferFunction(
                [spectrum.Spectrum(self.eng, interp(self.eng, log_rs_value), rs) for rs in out_rs], self.in_eng
                )
        else:
            raise TypeError("Invalid interp_type specified.")

    def coarsen(self, type='step', )

def process_raw_tf(file):
    """Processes raw data to return transfer functions.
    
    Parameters
    ----------
    file : str
        File to be processed. 

    Returns
    -------
    list of TransferFunction
        List indexed by injection energy. 


    """

    def get_out_eng_absc(in_eng):
        """ Returns the output energy abscissa for a given input energy. 

        Parameters
        ----------
        in_eng : float
            Input energy (in eV). 

        Returns
        -------
        ndarray
            Output energy abscissa. 
        """
        log_bin_width = np.log((phys.me + in_eng)/1e-4)/500
        bin_boundary = 1e-4 * np.exp(np.arange(501) * log_bin_width)
        bin_boundary_low = bin_boundary[0:500]
        bin_boundary_upp = bin_boundary[1:501]

        return np.sqrt(bin_boundary_low * bin_boundary_upp)

    #Redshift abscissa. In decreasing order.
    rs_step = 50
    rs_upp  = 31. 
    rs_low  = 4. 

    log_rs_absc = (np.log(rs_low) + (np.arange(rs_step) + 1)
                 *(np.log(rs_upp) - np.log(rs_low))/rs_step)
    log_rs_absc = np.flipud(log_rs_absc)

    # Input energy abscissa. 

    in_eng_step = 500
    low_in_eng_absc = 3e3 + 100.
    upp_in_eng_absc = 5e3 * np.exp(39 * np.log(1e13/5e3) / 40)
    in_eng_absc = low_in_eng_absc * np.exp((np.arange(in_eng_step)) * 
                  np.log(upp_in_eng_absc/low_in_eng_absc) / in_eng_step)

    # Output energy abscissa
    out_eng_absc_arr = np.array([get_out_eng_absc(in_eng) 
                                for in_eng in in_eng_absc])

    # Initial injected bin in output energy abscissa
    init_inj_eng_arr = [out_eng_absc[out_eng_absc < in_eng][-1] 
        for in_eng,out_eng_absc in zip(in_eng_absc, out_eng_absc_arr)
    ]

    # Import raw data. 

    tf_raw = np.load(file)
    tf_raw = np.swapaxes(tf_raw, 0, 1)
    tf_raw = np.swapaxes(tf_raw, 1, 2)
    tf_raw = np.swapaxes(tf_raw, 2, 3)
    tf_raw = np.flip(tf_raw, axis=0)

    # tf_raw has indices (redshift, xe, out_eng, in_eng), redshift in decreasing order.

    # Prepare the output.

    norm_fac = (in_eng_absc/init_inj_eng_arr)*2

    # The transfer function is expressed as a dN/dE spectrum as a result of injecting approximately 2 particles in out_eng_absc[-1]. The exact number is computed and the transfer function appropriately normalized to 1 particle injection (at energy out_eng_absc[-1]).

    tf_raw_list = [
        [spectrum.Spectrum(out_eng_absc_arr[i], tf_raw[j,0,:,i]/norm_fac[i], 
            np.exp(log_rs_absc[j])) for j in np.arange(tf_raw.shape[0])]
        for i in tqdm(np.arange(tf_raw.shape[-1]))
    ]

    transfer_func_table = [
        TransferFunction(spec_arr, init_inj_eng, rebin_eng = init_inj_eng_arr) for init_inj_eng, out_eng_absc, spec_arr in zip(
                init_inj_eng_arr, out_eng_absc_arr, tqdm(tf_raw_list))
    ]

    #Rebin to the desired abscissa, which is in_eng_absc.
    # for spec_list,out_eng_absc in zip(tqdm(tf_raw_list),out_eng_absc_arr):
    #     for spec in spec_list:
    #         spec.rebin(in_eng_absc)
    #     # Note that the injection energy is out_eng_absc[-1] due to our conventions in the high energy code.
    #     transfer_func_table.append(TransferFunction(spec_list, out_eng_absc[-1]))

    return transfer_func_table