"""Functions and classes for processing lists of transfer functions."""

import numpy as np
from tqdm import tqdm_notebook as tqdm

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
    """

    def __init__(self, tflist):
        self.tflist = tflist
        print(type(tflist[0]))

        if (not np.all([isinstance(tfunc, tf.TransFuncAtRedshift) 
                for tfunc in tflist]) and
            not np.all([isinstance(tfunc, tf.TransFuncAtEnergy)
                for tfunc in tflist])
        ):

            raise TypeError('transfer functions must be of the same type.')

        if isinstance(tflist[0], tf.TransFuncAtRedshift):
            self.tftype = 'rs'
            self.rs = np.array([tfunc.rs for tfunc in self.tflist])
            self.in_eng = tflist[0].in_eng
        elif isinstance(tflist[0], tf.TransFuncAtEnergy):
            self.tftype = 'eng'
            self.rs = tflist[0].rs
            self.in_eng = np.array([tfunc.in_eng for tfunc in self.tflist])
        else:
            raise TypeError('can only be list of valid transfer functions.')

    def __iter__(self):
        return iter(self.tflist)

    def __getitem__(self, key):
        return self.tflist[key]

    def __setitem__(self, key, value):
        self.tflist[key] = value

    def at_val(self, new_val):
        """Returns the transfer functions at the new abscissa.

        Parameters
        ----------
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

        else:
            new_tflist = [tf.at_rs(new_val)
                for i,tf in zip(
                    tqdm(np.arange(len(self.tflist))), self.tflist
                )
            ]

            self.tflist = new_tflist
            self.rs = new_val