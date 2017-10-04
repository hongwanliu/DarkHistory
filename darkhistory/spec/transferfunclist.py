"""``transferfunclist`` contains functions and classes for \
processing lists of transfer functions."""

import numpy as np

import darkhistory.utilities as utils

class TransferFuncList:
    """List of transfer functions.
    
    Parameters
    ----------
    tflist : list of TransferFunction
        A list of transfer functions at various injection energies. 
    
    Attributes
    ----------
    in_eng : ndarray
        The injection energies of the transfer functions. 
    rs : ndarray
        The redshift abscissa for all transfer functions.
    dlnz : float
        The d log(1+z) step for all transfer functions.

    """
    def __init__(self, tflist):
        self.tflist = tflist
        self.in_eng = np.array([tf.in_eng for tf in tflist])
        if not all(np.diff(self.in_eng) > 0):
            raise TypeError("injection energies must be \
            ordered in increasing energy")
        if not utils.arrays_equal([tf.rs for tf in tflist]):
            raise TypeError("transfer function redshifts must be \
                the same.")
        if len(set(tf.dlnz for tf in tflist)) > 1:
            raise TypeError("transfer functions must have the \
                same dlnz.")

        self.dlnz = tflist[0].dlnz

    def __iter__(self):
        return iter(self.tflist)

    def __getitem__(self,key):
        if np.issubdtype(type(key), int) or isinstance(key, slice):
            return self.tflist[key]
        else:
            raise TypeError("index must be int.")

    def __setitem__(self,key,value):
        if isinstance(key, int):
            if not isinstance(value, (list, tuple)):
                if np.issubclass_(type(value), TransferFunction):
                    self.spec_arr[key] = value
                else:
                    raise TypeError("can only add TransferFunction.")
            else:
                raise TypeError("can only add one TransferFunction \
                    per index.")
        elif isinstance(key, slice):
            if len(self.spec_arr[key]) == len(value):
                for i,spec in zip(key,value): 
                    if np.issubclass_(type(spec), TransferFunction):
                        self.spec_arr[i] = spec
                    else: 
                        raise TypeError("can only add TransferFunction.")
            else:
                raise TypeError("can only add one TransferFunction \
                    per index.")
        else:
            raise TypeError("index must be int or slice.")





