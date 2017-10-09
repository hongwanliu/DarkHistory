"""Functions and classes for processing lists of transfer functions."""

import numpy as np
from numpy.linalg import matrix_power
from tqdm import tqdm_notebook as tqdm

from darkhistory.utilities import arrays_equal
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec.spectra import Spectra
import darkhistory.spec.transferfunction as tf

class TransferFuncList:
    """List of transfer functions. 

    Parameters
    ----------
    tflist : list of TransFuncAtRedshift or TransFuncAtEnergy

    Attributes
    ----------
    tftype : {'rs', 'eng'}
        Type of transfer functions in the list: 'rs' for TransFuncAtRedshift, 'eng' for TransFuncAtEnergy
    rs : ndarray
        Redshift abscissa of the transfer functions. 
    in_eng : ndarray
        Injection energy abscissa of the transfer functions.
    dlnz : float
        The d ln(1+z) step for the transfer functions.
    """

    def __init__(self, tflist):

        self.tflist = tflist

        if (not np.all([isinstance(tfunc, tf.TransFuncAtRedshift) 
                for tfunc in tflist]) and
            not np.all([isinstance(tfunc, tf.TransFuncAtEnergy)
                for tfunc in tflist])
        ):

            raise TypeError('transfer functions must be of the same type.')

        if not arrays_equal([tfunc.eng for tfunc in self.tflist]):
            raise TypeError('all transfer functions must have the same \
                energy abscissa.')

        if len(set([tfunc.dlnz for tfunc in self.tflist])) > 1:
            raise TypeError('all transfer functions must have the same \
                dlnz.')

        if isinstance(tflist[0], tf.TransFuncAtRedshift):
            self.tftype = 'rs'
            self.rs = np.array([tfunc.rs for tfunc in self.tflist])
            self.in_eng = tflist[0].in_eng
            self.dlnz = tflist[0].dlnz
        elif isinstance(tflist[0], tf.TransFuncAtEnergy):
            self.tftype = 'eng'
            self.rs = tflist[0].rs
            self.in_eng = np.array([tfunc.in_eng for tfunc in self.tflist])
            self.dlnz = tflist[0].dlnz
        else:
            raise TypeError('can only be list of valid transfer functions.')

    def __iter__(self):
        return iter(self.tflist)

    def __getitem__(self, key):
        return self.tflist[key]

    def __setitem__(self, key, value):
        self.tflist[key] = value

    def at_val(self, axis, new_val):
        """Returns the transfer functions at the new abscissa.

        Parameters
        ----------
        axis : {'rs', 'eng'}
            The axis along which to perform the interpolation.
        new_val : ndarray
            The new redshift or injection energy abscissa.
        """

        # i enables the use of tqdm. 

        if self.tftype == 'rs':
            new_tflist = [tf.at_eng(new_val)
                for i,tf in zip(
                    tqdm(np.arange(len(self.tflist))), self.tflist
                )
            ]

            self.tflist = new_tflist
            self.in_eng = new_val

        elif self.tftype == 'eng':
            new_tflist = [tf.at_rs(new_val)
                for i,tf in zip(
                    tqdm(np.arange(len(self.tflist))), self.tflist
                )
            ]

            self.tflist = new_tflist
            self.rs = new_val

        else: 
            raise TypeError('TransferFuncList.tftype is neither rs nor eng')

    def transpose(self):
        """ Transposes the list of transfer functions. 

        This takes a TransferFuncList made of TransFuncAtEnergy into a list a TransferFuncList made of TransFuncAtRedshift and vice-versa. 
        """
        if self.tftype == 'eng':

            new_tflist = [tf.TransFuncAtRedshift(
                    [tfunc.spec_arr[i] for tfunc in self.tflist],
                    self.in_eng, self.dlnz
                ) for i,rs in zip(
                    np.arange(self.rs.size), self.rs
                )
            ]

            self.tflist = new_tflist
            self.tftype = 'rs'

        elif self.tftype == 'rs':

            new_tflist = [tf.TransFuncAtEnergy(
                    [tfunc.spec_arr[i] for tfunc in self.tflist],
                    self.in_eng[i], self.dlnz
                ) for i,in_eng in zip(
                    np.arange(self.in_eng.size), self.in_eng
                )
            ]

            self.tflist = new_tflist
            self.tftype = 'eng'

        else:

            raise TypeError('TransferFuncList.tftype is neither rs nor eng')

    def extend_dlnz(self, dlnz_factor):
        """Obtains the new transfer function with larger dlnz. 

        This is obtained by multiplying the transfer function by itself dlnz_factor times. 

        Parameters
        ----------
        dlnz_factor : int
            The factor to increase dlnz by. 

        """
        transposed = False

        if self.tftype != rs:
            self.transpose()
            transposed = True

        for i,tf in zip(np.arange(self.tflist.size),self.tflist):
            self.tflist[i] = matrix_power(tf,dlnz_factor)

        if transposed:
            self.transpose()












