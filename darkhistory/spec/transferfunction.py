"""Functions and classes for processing transfer functions."""

import numpy as np
from scipy import interpolate
from tqdm import tqdm_notebook as tqdm

import darkhistory.physics as phys
from darkhistory.spec.spectools import rebin_N_arr
from darkhistory.spec.spectra import Spectra
from darkhistory.spec.spectrum import Spectrum 


class TransFuncAtEnergy(Spectra):
    """Transfer function at a given injection energy. 

    Collection of Spectrum objects, each at different redshifts. 

    Parameters
    ----------
    spec_arr : list of Spectrum
        List of Spectrum to be stored together. 
    in_eng : float
        Injection energy of this transfer function. 
    dlnz : float
        The d ln(1+z) step for the transfer function. 
    rebin_eng : ndarray, optional
        New abscissa to rebin all of the Spectrum objects into.

    Attributes
    ----------
    eng : ndarray
        Energy abscissa for the Spectrum. 
    rs : ndarray
        Redshift abscissa for the Spectrum objects. 
    grid_values : ndarray
        2D array with the spectra laid out in (rs, eng).

    """
    def __init__(self, spec_arr, in_eng, dlnz, rebin_eng=None):

        if isinstance(in_eng, (list, tuple, np.ndarray)):
            raise TypeError("can only have a single injection energy.")
        self.in_eng = in_eng
        self.dlnz = dlnz
        super().__init__(spec_arr, rebin_eng)

    def at_rs(self, new_rs, interp_type='val'):
        """Interpolates the transfer function at a new redshift. 

        Interpolation is logarithmic. 

        Parameters
        ----------
        new_rs : ndarray
            The redshifts or redshift bin indices at which to interpolate. 
        interp_type : {'val', 'bin'}
            The type of interpolation. 'bin' uses bin index, while 'val' uses the actual redshift. 
        bounds_error : bool, optional
            See scipy.interpolate.interp1d.
        fill_value : array-like or (array-like, array-like) or "extrapolate", optional
            See scipy.interpolate.interp1d.
        """

        interp_func = interpolate.interp1d(
            np.log(self.rs), self.grid_values, axis=0, 
            bounds_error=bounds_error, fill_value=fill_value
        )

        if interp_type == 'val':
            
            new_spec_arr = [
                Spectrum(self.eng, interp_func(np.log(rs)), rs)
                    for rs in new_rs
            ]
            return TransFuncAtEnergy(
                new_spec_arr, self.in_eng, self.dlnz
            )

        elif interp_type == 'bin':
            
            log_new_rs = np.interp(
                np.log(new_rs), 
                np.arange(self.rs.size), 
                np.log(self.rs)
            )

            return self.at_rs(np.exp(log_new_rs))

        else:
            raise TypeError("invalid interp_type specified.")
        

class TransFuncAtRedshift(Spectra):
    """Transfer function at a given redshift. 

    Collection of Spectrum objects, each at different injection energies. 

    Parameters
    ----------
    spec_arr : list of Spectrum
        List of Spectrum to be stored together. 
    in_eng : ndarray
        Injection energies of this transfer function. 
    dlnz : float
        The d ln(1+z) step for the transfer function. 
    rebin_eng : ndarray, optional
        New abscissa to rebin all of the Spectrum objects into. 

    Attributes
    ----------
    eng : ndarray
        Energy abscissa for the Spectrum. 
    rs : float
        The redshift of the Spectrum objects. 
    grid_values : ndarray
        2D array with the spectra laid out in (in_eng, eng).

    """

    def __init__(self, spec_arr, in_eng, dlnz, rebin_eng=None):

        self.in_eng = in_eng
        self.dlnz = dlnz
        super().__init__(spec_arr, rebin_eng)
        if len(set([rs for rs in self.rs])) > 1:
            raise TypeError("all spectra must have identical redshifts.")
        self.rs = self.spec_arr[0].rs 

    def at_in_eng(self, new_eng, interp_type='val', bounds_error=None, fill_value=np.nan):
        """Interpolates the transfer function at a new injection energy. 

        Interpolation is logarithmic. 

        Parameters
        ----------
        new_eng : ndarray
            The injection energies or injection energy bin indices at which to interpolate. 
        interp_type : {'val', 'bin'}
            The type of interpolation. 'bin' uses bin index, while 'val' uses the actual injection energies.
        bounds_error : bool, optional
            See scipy.interpolate.interp1d.
        fill_value : array-like or (array-like, array-like) or "extrapolate", optional
            See scipy.interpolate.interp1d.

        Returns
        -------
        TransFuncAtRedshift
            New transfer function at the new injection energy. 
        """

        interp_func = interpolate.interp1d(
            np.log(self.in_eng), self.grid_values, axis=0, 
            bounds_error=bounds_error, fill_value=fill_value
        )

        if interp_type == 'val':
            new_spec_arr = [
                Spectrum(self.eng, interp_func(np.log(eng)), self.rs) 
                for eng in new_eng
            ]
            return TransFuncAtRedshift(
                new_spec_arr, new_eng, self.dlnz
            )

        elif interp_type == 'bin':

            log_new_eng = np.interp(
                np.log(new_eng),
                np.arange(self.in_eng.size),
                np.log(self.in_eng)
            )

            return self.at_in_eng(np.exp(log_new_eng))

    def at_eng(self, new_eng, interp_type='val', bounds_error=None, fill_value= np.nan):
        """Interpolates the transfer function at a new energy abscissa. 

        Interpolation is logarithmic. 

        Parameters
        ----------
        new_eng : ndarray
            The energy abscissa or energy abscissa bin indices at which to interpolate. 
        interp_type : {'val', 'bin'}
            The type of interpolation. 'bin' uses bin index, while 'val' uses the actual injection energies. 
        bounds_error : bool, optional
            See scipy.interpolate.interp1d.
        fill_value : array-like or (array-like, array-like) or "extrapolate", optional
            See scipy.interpolate.interp1d.

        Returns
        -------
        TransFuncAtRedshift
            New transfer function at the new energy abscissa. 
        """

        interp_func = interpolate.interp1d(
            np.log(self.eng), self.grid_values, axis=1, 
            bounds_error=bounds_error, fill_value=fill_value
        )

        if interp_type == 'val':
            new_grid_values = np.transpose(
                np.stack([interp_func(np.log(eng)) for eng in new_eng])
            )
            new_spec_arr = [
                Spectrum(new_eng, spec, self.rs) 
                for spec in new_grid_values
            ]
            return TransFuncAtRedshift(
                new_spec_arr, self.in_eng, self.dlnz
            )

        elif interp_type == 'bin':

            log_new_eng = np.interp(
                np.log(new_eng),
                np.arange(self.eng.size),
                np.log(self.eng)
            )

            return self.at_eng(np.exp(log_new_eng))

    def at_rs(self, new_eng, interp_type='val'):
        """ Removes the inherited Spectra.at_rs from this class.

        Parameters
        ----------
        new_rs : ndarray
            The redshifts or redshift bin indices at which to interpolate. 
        interp_type : {'val', 'bin'}
            The type of interpolation. 'bin' uses bin index, while 'val' uses the actual redshift. 
        """

        raise AttributeError("Not implemented for TransFuncAtRedshift class.")

    def plot(self, ax, ind=None, step=1, indtype='ind', 
        abs_plot=False, **kwargs):
        """Plots the contained `Spectrum` objects. 

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis handle of the figure to show the plot in.
        ind : int, float, tuple or ndarray, optional.
            Index or injected energy of Spectrum to plot, or a tuple of indices or injected energies providing a range of Spectrum to plot, or a list of indices or injected energies of Spectrum to plot.
        step : int, optional
            The number of steps to take before choosing one Spectrum to plot.
        indtype : {'ind', 'in_eng'}, optional
            Specifies whether ind is an index or an abscissa value.
        abs_plot :  bool, optional
            Plots the absolute value if true.
        **kwargs : optional
            All additional keyword arguments to pass to matplotlib.plt.plot. 

        Returns
        -------
        matplotlib.figure
        """
        
        if ind is None:
            return self.plot(
                ax, ind=np.arange(self.in_eng.size), 
                abs_plot=abs_plot, **kwargs
            )

        if indtype == 'ind':

            if np.issubdtype(type(ind), int):
                if abs_plot:
                    return ax.plot(
                        self.eng, 
                        np.abs(self.spec_arr[ind].dNdE), 
                        **kwargs
                    )
                else:
                    return ax.plot(
                        self.eng, 
                        self.spec_arr[ind].dNdE, 
                        **kwargs
                    )

            elif isinstance(ind, tuple):
                if abs_plot:
                    spec_to_plot = np.stack(
                        [np.abs(self.spec_arr[i].dNdE) 
                            for i in 
                                np.arange(ind[0], ind[1], step)
                        ], 
                        axis=-1
                    )
                else:
                    spec_to_plot = np.stack(
                        [self.spec_arr[i].dNdE 
                            for i in 
                                np.arange(ind[0], ind[1], step)
                        ], 
                        axis=-1
                    )
                return ax.plot(self.eng, spec_to_plot, **kwargs)
                
            
            elif isinstance(ind, np.ndarray):
                if abs_plot:
                    spec_to_plot = np.stack(
                        [np.abs(self.spec_arr[i].dNdE)
                            for i in ind
                        ], axis=-1
                    ) 
                else:
                    spec_to_plot = np.stack(
                        [self.spec_arr[i].dNdE
                            for i in ind
                        ], axis=-1
                    )
                return ax.plot(self.eng, spec_to_plot, **kwargs)
                

            else:
                raise TypeError("ind should be either int, tuple of int or ndarray.")

        if indtype == 'in_eng':

            if (np.issubdtype(type(ind),int) or 
                    np.issubdtype(type(ind), float)):
                return self.at_in_eng(
                        np.array([ind]), interp_type='val'
                    ).plot(
                    ax, ind=0, abs_plot=abs_plot, **kwargs
                )

            elif isinstance(ind, tuple):
                eng_to_plot = np.arange(ind[0], ind[1], step)
                return self.at_in_eng(
                        eng_to_plot, interp_type='val'
                    ).plot(
                    ax, abs_plot=abs_plot,**kwargs
                )

            elif isinstance(ind, np.ndarray):
                return self.at_in_eng(
                        ind, interp_type='val'
                    ).plot(
                    ax, abs_plot=abs_plot, **kwargs
                )

        else:
            raise TypeError("indtype must be either ind or in_eng.")


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

    from darkhistory.spec.transferfunclist import TransferFuncList

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
    init_inj_eng_arr = np.array([out_eng_absc[out_eng_absc < in_eng][-1] 
        for in_eng,out_eng_absc in zip(in_eng_absc, out_eng_absc_arr)
    ])

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
        [Spectrum(out_eng_absc_arr[i], tf_raw[j,0,:,i]/norm_fac[i], 
            np.exp(log_rs_absc[j])) for j in np.arange(tf_raw.shape[0])
        ]
        for i in np.arange(tf_raw.shape[-1])
    ]

    normfac2 = rebin_N_arr(np.ones(init_inj_eng_arr.size), 
        init_inj_eng_arr, init_inj_eng_arr
    ).dNdE
    # This rescales the transfer function so that it is now normalized to
    # dN/dE = 1. 
    

    transfer_func_table = TransferFuncList([
        TransFuncAtEnergy(spec_arr/N, init_inj_eng, 
            0.002, rebin_eng = init_inj_eng_arr
        ) for N, init_inj_eng, spec_arr in zip(normfac2,
            init_inj_eng_arr, tqdm(tf_raw_list)
        )
    ])

    # This further rescales the spectrum so that it is now the transfer
    # function for dN/dE = 1 in in_eng_absc. 


    #Rebin to the desired abscissa, which is in_eng_absc.
    # for spec_list,out_eng_absc in zip(tqdm(tf_raw_list),out_eng_absc_arr):
    #     for spec in spec_list:
    #         spec.rebin(in_eng_absc)
    #     # Note that the injection energy is out_eng_absc[-1] due to our conventions in the high energy code.
    #     transfer_func_table.append(TransferFunction(spec_list, out_eng_absc[-1]))

    return transfer_func_table
