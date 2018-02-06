"""Electron cooling through ICS."""

import numpy as np 
import pickle

import darkhistory.physics as phys

from darkhistory.electrons.ics.ics_spectrum import ics_spec 
from darkhistory.electrons.ics.ics_engloss_spectrum import engloss_spectrum 

def get_ICS_photon_spectrum(
    nonrel_tf_filename, rel_tf_filename, engloss_tf_filename
):
    """Returns the photon spectrum obtained by electron cooling through ICS. 

    Parameters
    ----------
    nonrel_tf_filename : string
        The file name for the nonrelativistic ICS photon spectrum produced by a single primary electron scattering.
    rel_tf_filename : string
        The file name for the relativistic ICS photon spectrum produced by a single primary electron scattering. 
    engloss_tf_filename : string
        The file name for energy loss ICS spectrum produced by a single primary electron scattering.  

    Returns
    -------

    """



    raw_nonrel_ICS_tf = pickle.load(open("/Users/hongwan/Dropbox (MIT)/Photon Deposition/ICS_nonrel.raw","rb"))
    raw_rel_ICS_tf = pickle.load(open("/Users/hongwan/Dropbox (MIT)/Photon Deposition/ICS_rel.raw","rb"))
    raw_engloss_tf = pickle.load(open("/Users/hongwan/Dropbox (MIT)/Photon Deposition/ICS_englossspec.raw","rb"))