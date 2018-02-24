import sys
sys.path.append("../..")

import numpy as np
import physics as phys
import spec.spectools as spectools

def compute_dep_inj_ionization_ratio(photon_spectrum, n, tot_inj, method='old'):
    """ Given a spectrum of deposited photons, resolve its energy into continuum photons, HI excitation, and HI, HeI, HeII ionization in that order.  The
        spectrum must provide the energy density of photons per unit time within each bin, not just the total energy within each bin.
        Q: can photons heat the IGM?  Should this method keep track of the fact that x_e, xHII, etc. are changing?

    Parameters
    ----------
    photon_spectrum : Spectrum object
        spectrum of photons
    n : list of floats
        density of (HI, HeI, HeII).
    tot_inj : float
        total energy injected by DM
    method : {'old','ion','new'}
        'old': All photons >= 13.6eV ionize hydrogen, within [10.2, 13.6)eV excite hydrogen, < 10.2eV are labelled continuum.
        'ion': Same as 'old', but now photons >= 13.6 can ionize HeI and HeII also.
        'new': Same as 'ion', but now [10.2, 13.6)eV photons treated more carefully

    Returns
    -------
    tuple of floats
        Ratio of deposited energy to a given channel over energy deposited by DM.
        The order of the channels is HI excitation and HI, HeI, HeII ionization
    """
    continuum, excite_HI, f_HI, f_HeI, f_HeII = 0,0,0,0,0

    continuum = photon_spectrum.toteng(bound_type='eng', bound_arr=np.array([photon_spectrum.eng[0],phys.lya_eng]))[0]/tot_inj
    ion_index = np.searchsorted(photon_spectrum.eng,phys.rydberg)
    ion_bin = spectools.get_bin_bound(photon_spectrum.eng)[ion_index]

    if(method != 'new'):
        excite_HI = photon_spectrum.toteng(bound_type='eng', bound_arr=np.array([phys.lya_eng,phys.rydberg]))[0]/tot_inj
    if(method == 'old'):
        f_HI = photon_spectrum.toteng(bound_type='eng', bound_arr=np.array([phys.rydberg,photon_spectrum.eng[-1]]))[0]/tot_inj
    elif(method == 'ion'):
        # probability of being absorbed within time step dt in channel a = \sigma(E)_a n_a c*dt
        # First convert from probability of being absorbed in channel 'a' to conditional probability given that these are deposited photons
        ionHI, ionHeI, ionHeII = [phys.photo_ion_xsec(photon_spectrum.eng[ion_index:],channel)*n[i] for i,channel in enumerate(['H0','He0','He1'])]
        totList = ionHI + ionHeI + ionHeII

        f_HI, f_HeI, f_HeII = [sum(photon_spectrum.eng[ion_index:]*photon_spectrum.N[ion_index:]*llist/totList)/tot_inj for llist in [ionHI, ionHeI, ionHeII]]

        #There's an extra piece of energy between 13.6 and the energy at ion_index. We expect 100% ionization within this region, so
        extra_HI, extra_HeI, extra_HeII = [phys.photo_ion_xsec(np.array([phys.rydberg+ion_bin])/2,channel)*n[i] for i, channel in enumerate(['H0','He0','He1'])]
        tot = extra_HI + extra_HeI + extra_HeII
        df_HI, df_HeI, df_HeII = [(photon_spectrum.toteng(bound_type='eng',bound_arr=np.array([phys.rydberg,ion_bin]))*extra/tot)[0]/tot_inj for extra in [extra_HI, extra_HeI, extra_HeII]]
        f_HI = f_HI + df_HI
        f_HeI = f_HeI + df_HeI
        f_HeII = f_HeII + df_HeII
    return continuum, excite_HI, f_HI, f_HeI, f_HeII
