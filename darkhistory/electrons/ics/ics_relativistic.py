"""ICS functions for relativistic electrons."""

import numpy as np
from scipy import integrate
from tqdm import tqdm_notebook as tqdm

import darkhistory.physics as phys
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec.transferfunction import TransFuncAtRedshift


def icsspec_relativistic(eleceng_arr, photeng_arr, rs):
    """Returns the ICS scattered photon spectrum at relativistic electron energies. 

    ICS off the CMB is assumed. 

    Parameters
    ----------
    eleceng_arr : ndarray
        A list of electron *total* energies. Can be less than electron mass.
    photeng_arr : ndarray
        A list of scattered photon energies. 
    rs : float
        The redshift to evaluate the ICS rate at. 

    Returns
    -------
    TransFuncAtRedshift
        A transfer function at fixed redshift, indexed by in_eng = electron kinetic energy, eng = scattered photon energy of (dN/dE dt), where E is the energy of the scattered photon, normalized to one electron.

    Note
    ----
    Explain why the total electron energy can be less than the total electron mass.
    """
    gamma_arr = eleceng_arr/phys.me

    Gamma_eps_over_CMBeng_arr = 4*gamma_arr/phys.me
    E1_arr = np.array(
        [
            [
                photeng/eleceng for photeng in photeng_arr
            ] for eleceng in eleceng_arr
        ]
    )
    q_arr_times_CMBeng = np.array(
        [
            [
                E1_arr[i,j]/(Gamma_eps_over_CMBeng_arr[i]*(1 - E1_arr[i,j]))
                for j in np.arange(photeng_arr.size)
            ] for i in np.arange(eleceng_arr.size) 
        ]
    )


    # beta_arr = np.sqrt(1 - 1/gamma_arr**2)
    def integrand(CMBeng, indelec, indphot):

        prefac = (3/4)*phys.thomson_xsec*phys.c/(gamma_arr[indelec]**2*CMBeng)

        Gamma_eps = CMBeng*Gamma_eps_over_CMBeng_arr[indelec]
        E1 = E1_arr[indelec, indphot]
        q = q_arr_times_CMBeng[indelec, indphot]/CMBeng

        outval = (2*q*np.log(q) + (1+2*q)*(1-q)
                    + (1-q)/2*((Gamma_eps*q)**2)/(1+Gamma_eps*q)
        )

        CMB_spec_prefac = 8*np.pi*(CMBeng**2)/((phys.ele_compton*phys.me)**3)
        if CMBeng < 100*0.235e-3*rs:
            CMB_spec_local = CMB_spec_prefac/(
                                np.exp(CMBeng/(0.235e-3*rs)) - 1
                            )
        else:
            CMB_spec_local = 0

        return prefac*outval*CMB_spec_local

    # def integrand_div_by_CMB(CMBeng, eleceng, photeng):

    #     gamma = eleceng/phys.me
    #     # beta = np.sqrt(1 - 1/gamma**2)

    #     def prefac(CMBeng):
            
    #         return((3/4)*phys.thomson_xsec*phys.c/(gamma**2*CMBeng))

    #     def integrand_part(CMBeng, photeng):

    #         Gamma_eps = 4*CMBeng*gamma/(phys.me)
    #         E1 = photeng/eleceng
    #         q = E1/(Gamma_eps*(1 - E1))

    #         outval = (2*q*np.log(q) + (1+2*q)*(1-q)
    #                     + (1-q)/2*((Gamma_eps*q)**2)/(1+Gamma_eps*q)
    #         )

    #         # zero out parts where q exceeds theoretical max value
    #         # outval = np.where(
    #         #     q >= 1 or q <= 1/(4*gamma**2), np.zeros(outval.size), outval
    #         # )

    #         return outval

    #     return prefac(CMBeng)*integrand_part(CMBeng, photeng)

    # def integrand(CMBeng, eleceng, photeng):

    #     return (integrand_div_by_CMB(CMBeng, eleceng, photeng)
    #         * CMB_spec(CMBeng, phys.TCMB(rs))
    #     )

    # Not entirely clear to me that the limits here are right. 

    lowlim = np.array(
                [photeng_arr/gamma*phys.me
                    /(gamma*phys.me - photeng_arr)
                    for gamma in gamma_arr
                ]
    )

    upplim = np.array([photeng_arr for gamma in gamma_arr])

    # Zero out where the CMB spectrum is already set to zero.
    upplim = np.where(upplim < 100*phys.TCMB(rs),
                        upplim,
                        100*phys.TCMB(rs)*np.ones(upplim.shape)
    )

    # lowlim = np.array([(1-beta)/(1+beta)*photeng_arr for beta in beta_arr])
    # upplim = np.array([(1+beta)/(1-beta)*photeng_arr for beta in beta_arr])

    # # Zero out where the CMB spectrum is already set to zero.
    # upplim = np.where(upplim < 100*phys.TCMB(rs),
    #                     upplim,
    #                     100*phys.TCMB(rs)*np.ones(upplim.shape)
    # )

    spec_arr_raw = np.array([
        [
            integrate.quad(integrand, lowlim[i,j], upplim[i,j],
                args = (i, j), epsabs = 0, epsrel = 1e-3
            )[0] if i > j else 0 for j in np.arange(photeng_arr.size)
        ] for i in tqdm(np.arange(eleceng_arr.size))
    ])

    spec_arr = [
        Spectrum(photeng_arr, np.array(spec), rs) for spec in spec_arr_raw
    ]

    # dlnz set to 1 second, which is the normalization for dN/dE dt. 
    return TransFuncAtRedshift(
        spec_arr, eleceng_arr, 1/rs*(phys.dtdz(rs)**-1)
    )






