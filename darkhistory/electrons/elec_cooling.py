"""Functions for electron cooling."""

import numpy as np
from darkhistory.spec.transferfunction import TransFuncAtRedshift

def cool_elec(
    elec_spec, sec_phot_tf, sec_elec_tf, 
    is_engloss_spec=True
):
    """ Generic function for electron cooling. 

    Parameters
    ----------
    elec_spec : Spectrum
        The electron spectrum to cool. 
    sec_phot_tf : TransFuncAtRedshift
        The transfer function to produce secondary photons, units dN/(dE dt), normalized to 1 primary electron and 1 second. The abscissa is identical to elec_spec.
    sec_elec_tf : TransFuncAtRedshift
        The transfer function to produce secondary electrons. This is either the secondary electron spectrum itself (dN/(dE dt)) or the energy loss spectrum (dN /(d Delta dt)), where Delta is the energy loss. Normalized to 1 primary electron and 1 second.
    is_engloss_spec : bool, optional
        If true, takes sec_elec_tf to be an energy loss spectrum. 

    Returns
    -------
    Spectrum
        The final secondary photon spectrum, units dN/(dE dt). 
    """

    if is_engloss_spec:
        engloss_tf = sec_elec_tf
        for spec, in_eng in zip(
            engloss_tf.spec_arr, engloss_tf.in_eng
        ):
            new_eng = in_eng - spec.eng
            spec.shift_eng(new_eng, reverse=True)




        sec_elec_tf = TransFuncAtRedshift


    
