"""ICS functions for low energy electrons."""

import numpy as np
from scipy import integrate
from tqdm import tqdm_notebook as tqdm

import darkhistory.physics as phys
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec.transferfunction import TransFuncAtRedshift


def icsspec_loweng(eleceng_arr, photeng_arr, rs):
    """Returns the ICS scattered photon spectrum at low electron energies. 

    ICS off the CMB is assumed. 

    Parameters
    ----------
    eleceng_arr : ndarray
        A list of electron *total* energies. 
    photeng_arr : ndarray
        A list of scattered photon energies. 
    rs : float
        The redshift to evaluate the ICS rate at. 

    Returns
    -------
    TransFuncAtRedshift
        A transfer function at fixed redshift, indexed by in_eng = electron kinetic energy, eng = scattered photon energy of (dN/dE dt), where E is the energy of the scattered photon, normalized to one electron.
    """
    gamma_arr = eleceng_arr/phys.me
    beta_arr = np.sqrt(1 - 1/gamma_arr**2)

    def integrand_div_by_CMB(CMBeng, eleceng, photeng):

        gamma = eleceng/phys.me
        beta = np.sqrt(1 - 1/gamma**2)

        def prefac(CMBeng):
            
            return (phys.c*(3/16)*phys.thomson_xsec
                /(gamma**3 * beta**2 * CMBeng**2)
            )

        def integrand_part(CMBeng, photeng):

            photenghigh = (
                CMBeng*(1+beta**2)/(beta**2)*np.sqrt((1+beta)/(1-beta))
                + (2/beta)*np.sqrt((1-beta)/(1+beta))*photeng
                - (((1-beta)**2)/(beta**2)*np.sqrt((1+beta)/(1-beta))
                        *(photeng**2/CMBeng)
                )
                + 2/(gamma*beta**2)*photeng*np.log(
                     (1-beta)/(1+beta)*photeng/CMBeng
                )
            )

            photenglow = (
                - CMBeng*(1+beta**2)/(beta**2)*np.sqrt((1-beta)/(1+beta))
                + (2/beta)*np.sqrt((1+beta)/(1-beta))*photeng
                + (1+beta)/(gamma*beta**2)*(photeng**2/CMBeng)
                - 2/(gamma*beta**2)*photeng*np.log(
                    (1+beta)/(1-beta)*photeng/CMBeng
                )
            )

            outval = np.where(photeng > CMBeng, photenghigh, photenglow)
            # zero out parts where photeng exceeds theoretical max value
            outval = np.where(photeng < gamma**2*(1+beta)**2*CMBeng,
                            outval, np.zeros(outval.size)
            )

            return outval

        return prefac(CMBeng)*integrand_part(CMBeng, photeng)

    def integrand(CMBeng, eleceng, photeng):

        return (integrand_div_by_CMB(CMBeng, eleceng, photeng)
            * phys.CMB_spec(CMBeng, phys.TCMB(rs))
        )

    lowlim = np.array([(1-beta)/(1+beta)*photeng_arr for beta in beta_arr])
    upplim = np.array([(1+beta)/(1-beta)*photeng_arr for beta in beta_arr])

    # Zero out where the CMB spectrum is already set to zero.
    upplim = np.where(upplim < 100*phys.TCMB(rs),
                        upplim,
                        100*phys.TCMB(rs)*np.ones(upplim.shape)
    )

    spec_arr_raw = np.array([
        [
            integrate.quad(integrand, lowlim[i,j], upplim[i,j],
                args = (eleceng, photeng), epsabs = 0, epsrel = 1e-3
            )[0] if eleceng > photeng else 0 for j,photeng in zip(
                np.arange(photeng_arr.size), photeng_arr
            )
        ] for i,eleceng in zip(
            tqdm(np.arange(eleceng_arr.size)), eleceng_arr
        )
    ])

    spec_arr = [
        Spectrum(photeng_arr, np.array(spec), rs) for spec in spec_arr_raw
    ]

    # dlnz set to 1 second, which is the normalization for dN/dE dt. 
    return TransFuncAtRedshift(
        spec_arr, eleceng_arr, 1/rs*(phys.dtdz(rs)**-1)
    )






