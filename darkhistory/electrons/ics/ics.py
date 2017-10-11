"""ICS functions."""

import numpy as np
from scipy import integrate
from tqdm import tqdm_notebook as tqdm

import darkhistory.physics as phys
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec.transferfunction import TransFuncAtRedshift


def icsspec(eleckineng_arr, photeng_arr, rs):
    """Returns the ICS scattered photon spectrum at all energies.

    ICS off the CMB is assumed. 

    Parameters
    ----------
    eleckineng_arr : ndarray
        A list of electron kinetic energies. 
    photeng_arr : ndarray
        A list of scattered photon energies. 
    rs : float
        The redshift to evaluate the ICS rate at. 

    Returns
    -------
    TransFuncAtRedshift
        A transfer function at fixed redshift, indexed by in_eng = electron kinetic energy, eng = scattered photon energy of (dN/dE dt), where E is the energy of the scattered photon, normalized to one electron.
    """
    
    gamma_where_rel = 10

    gamma_arr = 1 + eleckineng_arr/phys.me
    beta_arr = np.sqrt(1 - 1/gamma_arr**2)

    def integrand_div_by_CMB(CMBeng, eleckineng, photeng):

        gamma = 1 + eleckineng/phys.me
        beta = np.sqrt(1 - 1/gamma**2)

        relativistic = False
        if gamma > gamma_where_rel: 
            relativistic = True

        def prefac(CMBeng):

            if relativistic:
                return (3/4)*phys.thomson_xsec*phys.c/(gamma**2*CMBeng)
            else:
                return (phys.c*(3/16)*phys.thomson_xsec
                    /(gamma**3 * beta**2 * CMBeng**2)
                )

        def integrand_part(CMBeng, photeng):

            if relativistic:
                Gamma_eps = 4*CMBeng*gamma/(phys.me)
                E1 = photeng/phys.me
                q = E1/(Gamma_eps*(1 - E1))

                outval = (2*q*np.log(q) + (1+2*q)*(1-q)
                            + (1-q)/2*((Gamma_eps*q)**2)/(1+Gamma_eps*q)
                )

                # zero out parts where q exceeds theoretical max value
                outval = np.where(
                    q >= 1 or q <= 1/(4*gamma**2), 
                    np.zeros(outval.size), outval
                )

            else:
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

    def integrand(CMBeng, eleckineng, photeng):

        return (integrand_div_by_CMB(CMBeng, eleckineng, photeng)
            * phys.CMB_spec(CMBeng, phys.TCMB(rs))
        )

    lowlim_nonrel = np.array(
        [(1-beta)/(1+beta)*photeng_arr for beta in beta_arr]
    )
    upplim_nonrel = np.array(
        [(1+beta)/(1-beta)*photeng_arr for beta in beta_arr]
    )

    lowlim_rel = (photeng_arr/(4*gamma_arr)*phys.me
                    /(gamma_arr*phys.me - photeng_arr)
    )
    upplim_rel = photeng_arr

    lowlim = np.where(gamma_arr < gamma_where_rel, lowlim_nonrel, lowlim_rel)
    upplim = np.where(gamma_arr < gamma_where_rel, upplim_nonrel, upplim_rel)

    # Zero out where the CMB spectrum is already set to zero.
    upplim = np.where(upplim < 100*phys.TCMB(rs),
                        upplim,
                        100*phys.TCMB(rs)*np.ones(upplim.shape)
    )

    spec_arr_raw = np.array([
        [
            integrate.quad(integrand, lowlim[i,j], upplim[i,j],
                args = (eleceng, photeng), epsabs = 0, epsrel = 1e-3
            )[0] for j,photeng in zip(
                np.arange(photeng_arr.size), photeng_arr
            )
        ] for i,eleceng in zip(
            tqdm(np.arange(eleckineng_arr.size)), eleckineng_arr
        )
    ])

    spec_arr = [
        Spectrum(photeng_arr, np.array(spec), rs) for spec in spec_arr_raw
    ]

    # dlnz set to 1 second, which is the normalization for dN/dE dt. 
    return TransFuncAtRedshift(
        spec_arr, eleckineng_arr, 1/rs*(phys.dtdz(rs)**-1)
    )





