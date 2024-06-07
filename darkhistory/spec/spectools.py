"""Functions useful for processing spectral data."""

import numpy as np

from darkhistory import physics as phys
from darkhistory import utilities as utils
from darkhistory.numpy_groupies import aggregate as agg
import matplotlib.pyplot as plt
import warnings

from scipy import integrate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline


def get_bin_bound(eng):
    """Returns the bin boundary of an abscissa.

    The bin boundaries are computed by taking the midpoint of the **log** of the abscissa. The first and last entries are computed by taking all of the bins to be symmetric with respect to the bin center.

    Parameters
    ----------
    eng : ndarray
        Abscissa from which the bin boundary is obtained.

    Returns
    -------
    ndarray
        The bin boundaries.
    """
    if eng.size <= 1:
        raise TypeError("There needs to be more than 1 bin to get a bin width.")

    log_bin_width_low = np.log(eng[1]) - np.log(eng[0])
    log_bin_width_upp = np.log(eng[-1]) - np.log(eng[-2])

    bin_boundary = np.zeros(eng.size + 1)

    bin_boundary[1:-1] = np.sqrt(eng[:-1] * eng[1:])

    low_lim = np.exp(np.log(eng[0]) - log_bin_width_low / 2)
    upp_lim = np.exp(np.log(eng[-1]) + log_bin_width_upp / 2)
    bin_boundary[0] = low_lim
    bin_boundary[-1] = upp_lim

    return bin_boundary

def get_log_bin_width(eng):
    """Return the log bin width of the abscissa.

    Returns
    -------
    ndarray

    """
    bin_boundary = get_bin_bound(eng)
    return np.diff(np.log(bin_boundary))

def get_bounds_between(eng, E1, E2=None, bound_type='inc'):
    """Returns the bin boundary of an abscissa between two energies.

    If set to inc(lusive), E1 and E2 are part of the returned bounds.
    If set to exc(lusive), the bin boundaries between E1 and E2 are returned.

    Parameters
    ----------
    eng : ndarray
        Abscissa from which the bin boundary is obtained.
    E1 : float
        Lower bound
    E2 : float, optional
        Upper bound.  If None, E2 = max(bound of eng)
    bound_type : {'inc', 'exc'}, optional
        if 'inc', E1 and E2 are part of the returned bounds. If 'exc', they are not.

    Returns
    -------
    ndarray
        The bin boundaries between E1 and E2.
    """
    bin_boundary = get_bin_bound(eng)
    left_bound = np.searchsorted(bin_boundary,E1)
    if E2 is not None:
        right_bound = np.searchsorted(bin_boundary,E2) - 1
        bin_boundary = bin_boundary[left_bound : right_bound]
    else:
        bin_boundary = bin_boundary[left_bound:]

    if(bound_type == 'inc'):
        if E2 is None:
            return np.insert(bin_boundary,0,E1)
        else:
            tmp = np.insert(bin_boundary,0,E1)
            return np.append(tmp, E2)
    else:
        return bin_boundary

def get_indx(eng, E):
    """Returns index of bin containing E.

    Parameters
    ----------
    eng : ndarray
        Energy abscissa
    E : float
        You would like to know the bin index containing this energy

    Returns
    -------
    float
        Index of bin that contains E
    """
    return np.searchsorted(get_bin_bound(eng),E)-1

def rebin_N_arr(
    N_arr, in_eng, out_eng=None, spec_type='dNdE', log_bin_width=None
):
    """Rebins an array of particle number with fixed energy.

    Returns an array or a `Spectrum` object. The rebinning conserves both total number and total energy.

    Parameters
    ----------
    N_arr : ndarray
        An array of number of particles in each bin.
    in_eng : ndarray
        An array of the energy abscissa for each bin. The total energy in each bin `i` should be `N_arr[i]*in_eng[i]`.
    out_eng : ndarray, optional
        The new abscissa to bin into. If unspecified, assumed to be in_eng.
    spec_type : {'N', 'dNdE'}, optional
        The spectrum type to be output. Default is 'dNdE'.
    log_bin_width : ndarray, optional
        The bin width of the output abscissa.

    Returns
    -------
    Spectrum
        The output `Spectrum` with appropriate dN/dE, with abscissa out_eng.

    Raises
    ------
    OverflowError
        The maximum energy in `out_eng` cannot be smaller than any bin in `self.eng`.

    Notes
    -----
    The total number and total energy is conserved by assigning the number of particles N in a bin of energy eng to two adjacent bins in new_eng, with energies eng_low and eng_upp such that eng_low < eng < eng_upp. Then dN_low_dE_low = (eng_upp - eng)/(eng_upp - eng_low)*(N/(E * dlogE_low)), and dN_upp_dE_upp = (eng - eng_low)/(eng_upp - eng_low)*(N/(E*dlogE_upp)).

    If a bin in `in_eng` is below the lowest bin in `out_eng`, then the total number and energy not assigned to the lowest bin are assigned to the underflow. Particles will only be assigned to the lowest bin if there is some overlap between the bin index with respect to `out_eng` bin centers is larger than -1.0.

    If a bin in `in_eng` is above the highest bin in `out_eng`, then an `OverflowError` is thrown.

    See Also
    --------
    spectrum.Spectrum.rebin
    """

    from darkhistory.spec.spectrum import Spectrum
    # This avoids circular dependencies.

    if N_arr.size != in_eng.size:
        raise TypeError("The array for number of particles has a different length from the abscissa.")

    if out_eng is None:
        if log_bin_width is None:
            log_bin_width = get_log_bin_width(in_eng)
        return Spectrum(in_eng, N_arr/(in_eng*log_bin_width))

    if not np.all(np.diff(out_eng) > 0):
        raise TypeError("new abscissa must be ordered in increasing energy.")
    if out_eng[-1] < in_eng[-1]:
        raise OverflowError("the new abscissa lies below the old one: this function cannot handle overflow (yet?).")
    # Get the bin indices that the current abscissa (self.eng) corresponds to in the new abscissa (new_eng). Can be any number between 0 and self.length-1. Bin indices are wrt the bin centers.

    # Add an additional bin at the lower end of out_eng so that underflow can be treated easily.

    first_bin_eng = np.exp(np.log(out_eng[0]) - (np.log(out_eng[1]) - np.log(out_eng[0])))
    new_eng = np.insert(out_eng, 0, first_bin_eng)

    # Find the relative bin indices for in_eng wrt new_eng. The first bin in new_eng has bin index -1.


    bin_ind_interp = interp1d(
        new_eng, np.arange(new_eng.size)-1,
        bounds_error = False, fill_value = (-2, new_eng.size)
    )

    bin_ind = bin_ind_interp(in_eng)

    # Locate where bin_ind is below 0, above self.length-1 and in between.
    ind_low = np.where(bin_ind < 0)
    ind_high = np.where(bin_ind == new_eng.size)
    ind_reg = np.where( (bin_ind >= 0) & (bin_ind <= new_eng.size - 1) )

    # if ind_high[0].size > 0:
    #     raise OverflowError("the new abscissa lies below the old one: this function cannot handle overflow (yet?).")

    # Get the total N and toteng in each bin
    toteng_arr = N_arr*in_eng

    N_arr_low = N_arr[ind_low]
    N_arr_high = N_arr[ind_high]
    N_arr_reg = N_arr[ind_reg]

    toteng_arr_low = toteng_arr[ind_low]

    # Bin width of the new array. Use only the log bin width, so that dN/dE = N/(E d log E)
    if log_bin_width is None:
        new_E_dlogE = new_eng * np.diff(np.log(get_bin_bound(new_eng)))
    else:
        new_log_bin_width = np.insert(
            log_bin_width, 0, log_bin_width[0]
        )
        new_E_dlogE = new_eng * new_log_bin_width

    # Regular bins first, done in a completely vectorized fashion.

    # reg_bin_low is the array of the lower bins to be allocated the particles in N_arr_reg, similarly reg_bin_upp. This should also take care of the fact that bin_ind is an integer.
    reg_bin_low = np.floor(bin_ind[ind_reg]).astype(int)
    reg_bin_upp = reg_bin_low + 1

    # Takes care of the case where in_eng[-1] = out_eng[-1]
    reg_bin_low[reg_bin_low == new_eng.size-2] = new_eng.size - 3
    reg_bin_upp[reg_bin_upp == new_eng.size-1] = new_eng.size - 2

    reg_N_low = (reg_bin_upp - bin_ind[ind_reg]) * N_arr_reg
    reg_N_upp = (bin_ind[ind_reg] - reg_bin_low) * N_arr_reg

    reg_dNdE_low = ((reg_bin_upp - bin_ind[ind_reg]) * N_arr_reg
                   /new_E_dlogE[reg_bin_low+1])
    reg_dNdE_upp = ((bin_ind[ind_reg] - reg_bin_low) * N_arr_reg
                   /new_E_dlogE[reg_bin_upp+1])

    # Low bins.
    low_bin_low = np.floor(bin_ind[ind_low]).astype(int)

    N_above_underflow = np.sum((bin_ind[ind_low] - low_bin_low)
        * N_arr_low)
    eng_above_underflow = N_above_underflow * new_eng[1]

    N_underflow = np.sum(N_arr_low) - N_above_underflow
    eng_underflow = np.sum(toteng_arr_low) - eng_above_underflow
    low_dNdE = N_above_underflow/new_E_dlogE[1]

    new_dNdE = np.zeros(new_eng.size)
    new_dNdE[1] += low_dNdE
    # reg_dNdE_low = -1 refers to new_eng[0]
    np.add.at(new_dNdE, reg_bin_low+1, reg_dNdE_low)
    np.add.at(new_dNdE, reg_bin_upp+1, reg_dNdE_upp)
    # new_dNdE[reg_bin_low+1] += reg_dNdE_low
    # new_dNdE[reg_bin_upp+1] += reg_dNdE_upp

    # Generate the new Spectrum.

    out_spec = Spectrum(new_eng[1:], new_dNdE[1:])
    if spec_type == 'N':
        out_spec.switch_spec_type()
    elif spec_type != 'dNdE':
        raise TypeError('invalid spec_type.')
    out_spec.underflow['N'] += N_underflow
    out_spec.underflow['eng'] += eng_underflow

    return out_spec


def discretize(eng, func_dNdE, *args):
    r"""Discretizes a continuous function. 

    The function is integrated between the bin boundaries specified by `eng` to obtain the discretized spectrum, so that the final spectrum conserves number and energy between the bin **boundaries**.

    Parameters
    ----------
    eng : ndarray
        Both the bin boundaries to integrate between and the new abscissa after discretization (bin centers).
    func_dNdE : function
        A single variable function that takes in energy as an input, and then returns a dN/dE spectrum value.
    *args : optional
        Additional arguments and keyword arguments to be passed to `func_dNdE`.

    Returns
    -------
    Spectrum
        The discretized spectrum. rs is set to -1, and must be set manually.

    Notes
    ------
    Given a spectrum :math:`dN/dE`\ , represented by the function ``func_dNdE``, this function calculates the following quantities at the energy values :math:`E_i` specified in ``eng``:

    .. math::
        
        N[i] = \\int_{E_i}^{E_{i+1}} \\frac{dN}{dE} \\, dE
    
    .. math::

        \\epsilon[i] = \\frac{1}{N[i]} \\int_{E_i}^{E_{i+1}} E \\frac{dN}{dE} \\, dE

    We can now treat :math:`N[i]` and :math:`\\epsilon[i]` as a list of bins with energies :math:`\\epsilon[i]` and number of particles :math:`N[i]`. This is now rebinned into the abscissa specified by ``eng`` using :func:`.rebin_N_arr`, which conserves the total number.

    See Also
    ---------
    :func:`.rebin_N_arr`

    """
    def func_EdNdE(eng, *args):
        return func_dNdE(eng, *args)*eng

    # Generate a list of particle number N and mean energy eng_mean, so that N*eng_mean = total energy in each bin. eng_mean != eng.
    N = np.zeros(eng.size)
    eng_mean = np.zeros(eng.size)

    for low, upp, i in zip(
        eng[:-1], eng[1:], np.arange(eng.size-1)
    ):
    # Perform an integral over the spectrum for each bin.
        N[i] = integrate.quad(func_dNdE, low, upp, args= args)[0]
    # Get the total energy stored in each bin.
        if N[i] > 0:
            eng_mean[i] = integrate.quad(
                func_EdNdE, low, upp, args=args
            )[0]/N[i]
        else:
            eng_mean[i] = 0


    return rebin_N_arr(N, eng_mean, eng)


def get_normalized_spec(spec, dE_dVdt, rs):
    """
    Normalizes the spectrum to per baryon per dlnz, given dE/(dV dt). 

    Parameters
    ----------
    spec : Spectrum
        Input spectrum to be normalized. 
    dE_dVdt : float
        The injection dE/(dV dt) in eV cm^-3 s^-1. 
    rs : float
        The redshift (1+z). 

    Returns
    -------
    Spectrum
        The normalized spectrum (per baryon per dlnz). 

    """

    dE_dNBdlnz = dE_dVdt/(phys.nB*rs**3)/phys.hubble(rs)

    return spec/spec.toteng()*dE_dNBdlnz

def engloss_rebin_fast(in_eng, eng, grid_vals, final_eng):
    """
    Fast energy loss rebin.

    Parameters
    ----------
    in_eng : ndarray
        Injection energies (first dimension of `grid_vals`)
    eng : ndarray
        Energy loss abscissa (second dimension of `grid_vals`)
    grid_vals : 2D ndarray
        Number of particles with the given energy in `eng`.  
    final_eng : ndarray
        The final energy abscissa to bin into.

    Returns
    -------
    Spectra
        The final rebinned spectra.
    """

    # 2D array, (i,j) = in_eng[i] - eng[j], in ascending order.
    # sec_spec_eng = np.fliplr(np.float128(in_eng[:,None]) - np.float128(eng))
    sec_spec_eng = np.fliplr(in_eng[:,None] - eng)

    # Flipped as well.
    N_arr = np.fliplr(grid_vals)

    # final_eng = np.float128(final_eng)

    # Get the bin indices that the current abscissa (sec_spec_eng)
    # corresponds to in the new abscissa (final_eng). Bin indices are 
    # with respect to bin centers. 

    # Add an additional bin at the lower end of out_eng so that
    # underflow can be treated easily. 
    first_bin_eng = np.exp(
        np.log(final_eng[0])
        - (np.log(final_eng[1]) - np.log(final_eng[0]))
    )

    new_eng = np.insert(final_eng, 0, first_bin_eng)
    
    # Find the relative bin indices for self.eng. The first bin in 
    # new_eng has bin index -1. Underflow has index -2, overflow
    # corresponds to new_eng.size
    
    # bin_ind_interp = interp1d(
    #     new_eng, np.arange(new_eng.size)-1,
    #     bounds_error = False, fill_value = (-2, new_eng.size),
    #     assume_sorted=True
    # )
    
    # new_eng = np.float64(new_eng)
    # sec_spec_eng = np.float64(sec_spec_eng)

    bin_ind_interp = InterpolatedUnivariateSpline(
        new_eng, np.arange(new_eng.size)-1, k=1
    )
    
    bin_ind = bin_ind_interp(sec_spec_eng)

    # Only for InterpolatedUnivariateSpline
    bin_ind[bin_ind < -1] = -2
    bin_ind[bin_ind > new_eng.size-2] = new_eng.size

    # Locate where bin_ind is below 0, above self.length-1 
    # or in between. 

    ind_low  = bin_ind < 0
    ind_reg  = (bin_ind >= 0) & (bin_ind <= new_eng.size - 1)

    # Regular bins first. 

    # reg_bin_low is the array of the lower bins to be allocated the
    # particles in N_arr_reg, similarly reg_bin_upp. This should also
    # take care of the case where bin_ind is an integer. 
    
    reg_bin_low = -2*np.ones_like(bin_ind, dtype=int)
    reg_bin_upp = -2*np.ones_like(bin_ind, dtype=int)

    reg_bin_low[ind_reg] = np.floor(bin_ind[ind_reg]).astype(int)
    reg_bin_upp[ind_reg] = reg_bin_low[ind_reg] + 1
       
    # Takes care of the case where eng[-1] = new_eng[-1], which falls
    # under regular indices. Remember the extra bin on the left. 
    reg_bin_low[reg_bin_low == new_eng.size-2] = new_eng.size - 3
    reg_bin_upp[reg_bin_upp == new_eng.size-1] = new_eng.size - 2

    reg_data_low = np.zeros_like(bin_ind)
    reg_data_upp = np.zeros_like(bin_ind)
    
    # Split the particles up into the lower bin and upper bin. 
    reg_data_low[ind_reg] = (
        (reg_bin_upp[ind_reg] - bin_ind[ind_reg]) * N_arr[ind_reg]
    )
    reg_data_upp[ind_reg] = (
        (bin_ind[ind_reg] - reg_bin_low[ind_reg]) * N_arr[ind_reg]
    )

    in_eng_mask = np.outer(
        np.arange(in_eng.size, dtype=int), np.ones_like(eng, dtype=int)
    )
    
    low_bin_low = -2*np.ones_like(bin_ind, dtype=int)
    
    # Handle low bins. 
    low_bin_low[ind_low] = np.floor(bin_ind[ind_low]).astype(int)
    
    N_above_underflow = np.zeros_like(bin_ind)
    
    N_above_underflow[ind_low] = (bin_ind[ind_low] - low_bin_low[ind_low]) * N_arr[ind_low]

    # Add up.
    new_data = np.zeros((in_eng.size, new_eng.size))
    new_data[:,1] += np.sum(N_above_underflow, axis=1)

    ## Replace add.at with agg.aggregate

    # np.add.at(new_data, (in_eng_mask[ind_reg], reg_bin_low[ind_reg]+1), reg_data_low[ind_reg])
    # np.add.at(new_data, (in_eng_mask[ind_reg], reg_bin_upp[ind_reg]+1), reg_data_upp[ind_reg])

    low_data = agg.aggregate(
        np.array(
            [in_eng_mask[ind_reg], reg_bin_low[ind_reg]+1]
        ),
        reg_data_low[ind_reg],
        size = new_data.shape, func='sum', fill_value = 0
    )

    upp_data = agg.aggregate(
        np.array(
            [in_eng_mask[ind_reg], reg_bin_upp[ind_reg]+1]
        ),
        reg_data_upp[ind_reg],
        size = new_data.shape, func='sum', fill_value = 0
    )

    new_data += (low_data + upp_data)

    return new_data[:, 1:]

class EnglossRebinData:
    """ Structure for energy loss rebinning data. 

    Parameters
    ----------
    in_eng : ndarray
        The injected energy. 
    engloss_arr : ndarray
        Energy loss abscissa (second dimension of `grid_vals`) 
    final_eng : ndarray
        The final energy abscissa to bin into.

    Attributes
    ----------
    in_eng : ndarray
        The injected energy. 
    engloss_arr : ndarray
        Energy loss abscissa (second dimension of `grid_vals`) 
    final_eng : ndarray
        The final energy abscissa to bin into.
    new_eng : ndarray
        final_eng, but with additional first bin for underflow.
    bin_ind : 2D ndarray
        The (fractional) bin indices of final_eng that the data will be assigned to.
    ind_low : slice
        Slice which should have some part assigned to underflow. 
    ind_reg : slice
        Slice which should have some part rebinned in the grid.
    reg_bin_low : ndarray
        Index of lower energy bins of the grid that particles should be assigned to.
    reg_bin_upp : ndarray
        Index of higher energy bins of the grid that particles should be assigned to.
    low_bin_low : ndarray
        Marks grid points that have some component assigned to underflow.
    in_eng_mask : ndarray
        in_eng index of every point on the grid.

    Notes
    -----
    This class is used to store data for energy loss rebinning
    that only depends on the abscissae specified. 

    """

    def __init__(
        self, in_eng, engloss_arr, final_eng
    ):

        self.in_eng  = in_eng
        self.engloss_arr = engloss_arr
        self.final_eng = final_eng

        # 2D array, (i,j) = in_eng[i] - engloss_arr[j], in ascending order.
        sec_spec_eng = np.fliplr(in_eng[:,None] - engloss_arr)

        # Get the bin indices that the current abscissa (sec_spec_eng)
        # corresponds to in the new abscissa (final_eng). Bin indices are 
        # with respect to bin centers. 

        # Add an additional bin at the lower end of out_eng so that
        # underflow can be treated easily. 
        first_bin_eng = np.exp(
            np.log(final_eng[0])
            - (np.log(final_eng[1]) - np.log(final_eng[0]))
        )

        self.new_eng = np.insert(final_eng, 0, first_bin_eng)

        # Find the relative bin indices for sec_spec_eng. The first bin in 
        # new_eng has bin index -1. Underflow has index -2, overflow
        # corresponds to new_eng.size.

        bin_ind_interp = InterpolatedUnivariateSpline(
            self.new_eng, np.arange(self.new_eng.size)-1, k=1
        )
        
        self.bin_ind = bin_ind_interp(sec_spec_eng)
        # self.bin_ind is a 2D array.
        self.bin_ind[self.bin_ind < -1] = -2
        self.bin_ind[self.bin_ind > self.new_eng.size-2] = self.new_eng.size

        # Locate where self.bin_ind is below 0, above self.length-1 
        # or in between. 

        self.ind_low  = self.bin_ind < 0
        self.ind_reg  = (
            (self.bin_ind >= 0) & (self.bin_ind <= self.new_eng.size - 1)
        )

        # reg_bin_low is the array of the lower energy bins that particles
        # should be assigned to, similarly for reg_bin_upp. 
        # This should also take care of the case where 
        # bin_ind is an integer. 
        
        self.reg_bin_low = -2*np.ones_like(self.bin_ind, dtype=int)
        self.reg_bin_upp = -2*np.ones_like(self.bin_ind, dtype=int)

        self.reg_bin_low[self.ind_reg] = np.floor(
            self.bin_ind[self.ind_reg]
        ).astype(int)
        self.reg_bin_upp[self.ind_reg] = self.reg_bin_low[self.ind_reg] + 1
           
        # Takes care of the case where eng[-1] = new_eng[-1], which falls
        # under regular indices. Remember the extra bin on the left. 
        self.reg_bin_low[self.reg_bin_low == self.new_eng.size-2] = (
            self.new_eng.size - 3
        )
        self.reg_bin_upp[self.reg_bin_upp == self.new_eng.size-1] = (
            self.new_eng.size - 2
        )

        # low_bin_low is the array of the bins that need to be assigned
        # to underflow.
        self.low_bin_low = -2*np.ones_like(self.bin_ind, dtype=int)
        self.low_bin_low[self.ind_low] = np.floor(
            self.bin_ind[self.ind_low]
        ).astype(int)

        # in_eng_mask labels the bin number for in_eng. 
        self.in_eng_mask = np.outer(
            np.arange(self.in_eng.size, dtype=int), 
            np.ones_like(self.engloss_arr, dtype=int)
        )

    def rebin(self, grid_vals):

        # Flip grid_vals, since sec_spec_eng is flipped. 
        N_arr = np.fliplr(grid_vals)

        # Initialize arrays that will store the number of particles
        # to assign to lower/upper bins.

        reg_data_low = np.zeros_like(self.bin_ind)
        reg_data_upp = np.zeros_like(self.bin_ind)

        # Compute the assignments.
        reg_data_low[self.ind_reg] = (
            (self.reg_bin_upp[self.ind_reg] - self.bin_ind[self.ind_reg]) 
            * N_arr[self.ind_reg]
        )
        reg_data_upp[self.ind_reg] = (
            (self.bin_ind[self.ind_reg] - self.reg_bin_low[self.ind_reg]) 
            * N_arr[self.ind_reg]
        )

        # Handle case where some gets assigned to underflow, some to the 
        # first bin.
        N_above_underflow = np.zeros_like(self.bin_ind)

        N_above_underflow[self.ind_low] = (
            (self.bin_ind[self.ind_low] - self.low_bin_low[self.ind_low]) 
            * N_arr[self.ind_low]
        )

        new_data = np.zeros((self.in_eng.size, self.new_eng.size))
        new_data[:,1] += np.sum(N_above_underflow, axis=1)

        # Get number of particles assigned into lower energy bin.
        low_data = agg.aggregate(
            np.array([
                self.in_eng_mask[self.ind_reg], 
                self.reg_bin_low[self.ind_reg]+1
            ]),
            reg_data_low[self.ind_reg],
            size = new_data.shape, func='sum', fill_value = 0
        )

        upp_data = agg.aggregate(
            np.array([
                self.in_eng_mask[self.ind_reg], 
                self.reg_bin_upp[self.ind_reg]+1
            ]),
            reg_data_upp[self.ind_reg],
            size = new_data.shape, func='sum', fill_value = 0
        )

        new_data += (low_data + upp_data)

        return new_data[:, 1:]


    

