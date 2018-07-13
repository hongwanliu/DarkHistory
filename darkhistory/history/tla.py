"""Three-level atom model.

"""

import numpy as np
import darkhistory.physics as phys
import darkhistory.history.reionization as reion
from scipy.integrate import odeint

def compton_cooling_rate(xHII, xHeII, xHeIII, T_m, rs):
    """Returns the Compton cooling rate.

    Parameters
    ----------
    xHII : float
        n_HII/n_H.
    xHeII : float
        n_HeII/n_H.
    xHeIII : float
        n_HeIII/n_H.
    T_m : float
        The matter temperature.
    rs : float
        The redshift in 1+z.

    Returns
    -------
    float
        The Compton cooling rate in eV/s.

    Note
    ----
    This is the energy loss rate, *not* the temperature loss rate.

    """
    xe = xHII + xHeII + 2*xHeIII

    return (
        4 * phys.thomson_xsec * 4 * phys.stefboltz / phys.me
        * xe * phys.nH*rs**3 * (phys.TCMB(rs) - T_m)
        * phys.TCMB(rs)**4
    )

def get_history(
    init_cond, f_H_ion_in, f_H_exc_in, f_heating_in,
    dm_injection_rate, rs_vec, reion_switch=True
):
    """Returns the ionization and thermal history of the IGM.

    Parameters
    ----------
    init_cond : array
        Array containing [initial temperature, initial xHII, initial xHeII, initial xHeIII].
    f_H_ion_in : function or float
        f(rs, x_HI, x_HeI, x_HeII) for hydrogen ionization. Treated as constant if float.
    f_H_exc_in : function
        f(rs, x_HI, x_HeI, x_HeII) for hydrogen Lyman-alpha excitation. Treated as constant if float.
    f_heating_in : function
        f(rs, x_HI, x_HeI, x_HeII) for heating. Treated as constant if float.
    dm_injection_rate : function
        Injection rate of DM as a function of redshift.
    rs_vec : ndarray
        Abscissa for the solution.
    reion_switch : bool
        Reionization model included if true.

    Returns
    -------
    list of ndarray
        [temperature solution (in eV), xHII solution, xHeII, xHeIII].

    Note
    ----
    The actual differential equation that we solve is expressed in terms of y = arctanh(f*(x - f)), where f = 0.5 for x = xHII, and f = nHe/nH * 0.5 for x = xHeII or xHeIII, where nHe/nH is approximately 0.083.

    """

    # Defines the f(z) functions, which return a constant, 
    # if the input fz's are floats. 

    def f_H_ion(rs, xHI, xHeI, xHeII):
        if isinstance(f_H_ion_in, float):
            return f_H_ion_in
        elif callable(f_H_ion_in):
            return f_H_ion_in(rs, xHI, xHeI, xHeII)
        else:
            raise TypeError('f_H_ion_in must be float or an appropriate function.')

    def f_H_exc(rs, xHI, xHeI, xHeII):
        if isinstance(f_H_exc_in, float):
            return f_H_exc_in
        elif callable(f_H_exc_in):
            return f_H_exc_in(rs, xHI, xHeI, xHeII)
        else:
            raise TypeError('f_H_exc_in must be float or an appropriate function.')

    def f_heating(rs, xHI, xHeI, xHeII):
        if isinstance(f_heating_in, float):
            return f_heating_in
        elif callable(f_heating_in):
            return f_heating_in(rs, xHI, xHeI, xHeII)
        else:
            raise TypeError('f_heating_in must be float or an appropriate function.')

    chi = phys.nHe/phys.nH

    photoion_rate_HI   = reion.photoion_rate('HI')
    photoion_rate_HeI  = reion.photoion_rate('HeI')
    photoion_rate_HeII = reion.photoion_rate('HeII')

    photoheat_rate_HI   = reion.photoheat_rate('HI')
    photoheat_rate_HeI  = reion.photoheat_rate('HeI')
    photoheat_rate_HeII = reion.photoheat_rate('HeII')

    def tla_before_reion(var, rs):
        # Returns an array of values for [dT/dz, dyHII/dz,
        # dyHeII/dz, dyHeIII/dz].
        # var is the [temperature, xHII, xHeII, xHeIII] inputs.

        def xHII(yHII):
            return 0.5 + 0.5*np.tanh(yHII)
        def xHeII(yHeII):
            return chi/2 + chi/2*np.tanh(yHeII)
        def xHeIII(yHeIII):
            return chi/2 + chi/2*np.tanh(yHeIII)

        def dT_dz(yHII, yHeII, yHeIII, T_m, rs):

            xe = xHII(yHII) + xHeII(yHeII) + 2*xHeIII(yHeIII)
            xHI = 1 - xHII(yHII)
            xHeI = chi - xHeII(yHeII) - xHeIII(yHeIII)

            # This rate is temperature loss per redshift.
            adiabatic_cooling_rate = 2 * T_m/rs

            # This rate is *energy* loss per redshift, divided by
            # 3/2 * phys.nH * rs**3 * (1 + chi + xe).
            entropy_cooling_rate = - 3/2 * phys.nH * rs**3 * T_m * (
                dyHII_dz(yHII, yHeII, yHeIII, T_m, rs)
                    * 0.5/np.cosh(yHII)**2
                + dyHeII_dz(yHII, yHeII, yHeIII, T_m, rs)
                    * (chi/2)/np.cosh(yHeII)**2
                + dyHeIII_dz(yHII, yHeII, yHeIII, T_m, rs)
                    * (chi/2)/np.cosh(yHeIII)**2
            )


            return adiabatic_cooling_rate + (
                entropy_cooling_rate
                - phys.dtdz(rs)*(
                    compton_cooling_rate(
                        xHII(yHII), xHeII(yHeII), xHeIII(yHeIII), T_m, rs
                    )
                    + f_heating(rs, xHI, xHeI, xHeII(yHeII)) * dm_injection_rate(rs)
                )
            )/ (3/2 * phys.nH*rs**3 * (1 + chi + xe))


        def dyHII_dz(yHII, yHeII, yHeIII, T_m, rs):

            xe = xHII(yHII) + xHeII(yHeII) + 2*xHeIII(yHeIII)
            ne = xe * phys.nH*rs**3
            xHI = 1 - xHII(yHII)
            xHeI = chi - xHeII(yHeII) - xHeIII(yHeIII)

            return 2 * np.cosh(yHII)**2 * -phys.dtdz(rs) * (
                # Recombination processes
                - phys.peebles_C(xHII(yHII), rs) * (
                    phys.alpha_recomb(T_m) * xHII(yHII)*xe * phys.nH * rs**3
                    - phys.beta_ion(phys.TCMB(rs)) * xHI
                        * np.exp(-phys.lya_eng/T_m)
                )
                # DM injection. Note that C = 1 at late times.
                + f_H_ion(rs, xHI, xHeI, xHeII(yHeII)) * dm_injection_rate(rs)
                    / (phys.rydberg * phys.nH * rs**3)
                + (1 - phys.peebles_C(xHII(yHII), rs)) * (
                    f_H_exc(rs, xHI, xHeI, xHeII(yHeII)) * dm_injection_rate(rs)
                    / (phys.lya_eng * phys.nH * rs**3)
                )
            )

        def dyHeII_dz(yHII, yHeII, yHeIII, T_m, rs):
            xe = xHII(yHII) + xHeII(yHeII) + 2*xHeIII(yHeIII)
            ne = xe * phys.nH*rs**3
            xHeI = chi - xHeII(yHeII) - xHeIII(yHeIII)

            return 0

        def dyHeIII_dz(yHII, yHeII, yHeIII, T_m, rs):
            xe = xHII(yHII) + xHeII(yHeII) + 2*xHeIII(yHeIII)
            ne = xe * phys.nH*rs**3

            return 0

        T_m, yHII, yHeII, yHeIII = var[0], var[1], var[2], var[3]

        return [
            dT_dz(yHII, yHeII, yHeIII, T_m, rs),
            dyHII_dz(yHII, yHeII, yHeIII, T_m, rs),
            dyHeII_dz(yHII, yHeII, yHeIII, T_m, rs),
            dyHeIII_dz(yHII, yHeII, yHeIII, T_m, rs)
        ]

    def tla_reion(var, rs):
        # Returns an array of values for [dT/dz, dyHII/dz,
        # dyHeII/dz, dyHeIII/dz].
        # var is the [temperature, xHII, xHeII, xHeIII] inputs.

        def xHII(yHII):
            return 0.5 + 0.5*np.tanh(yHII)
        def xHeII(yHeII):
            return chi/2 + chi/2*np.tanh(yHeII)
        def xHeIII(yHeIII):
            return chi/2 + chi/2*np.tanh(yHeIII)

        def dT_dz(yHII, yHeII, yHeIII, T_m, rs):

            xe = xHII(yHII) + xHeII(yHeII) + 2*xHeIII(yHeIII)
            xHI = 1 - xHII(yHII)
            xHeI = chi - xHeII(yHeII) - xHeIII(yHeIII)

            # This rate is temperature loss per redshift.
            adiabatic_cooling_rate = 2 * T_m/rs

            # This rate is *energy* loss per redshift, divided by
            # 3/2 * phys.nH * rs**3 * (1 + chi + xe).
            entropy_cooling_rate = -T_m * (
                dyHII_dz(yHII, yHeII, yHeIII, T_m, rs)
                    * 0.5/np.cosh(yHII)**2
                + dyHeII_dz(yHII, yHeII, yHeIII, T_m, rs)
                    * (chi/2)/np.cosh(yHeII)**2
                + dyHeIII_dz(yHII, yHeII, yHeIII, T_m, rs)
                    * (chi/2)/np.cosh(yHeIII)**2
            )/(1 + chi + xe)

            # The reionization rates and the Compton rate
            # are expressed in *energy loss* *per second*.

            photoheat_total_rate = phys.nH * rs**3 * (
                xHI * photoheat_rate_HI(rs)
                + xHeI * photoheat_rate_HeI(rs)
                + xHeII(yHeII) * photoheat_rate_HeII(rs)
            )


            return (
                adiabatic_cooling_rate
                + entropy_cooling_rate
                + (
                    - phys.dtdz(rs)*(
                        compton_cooling_rate(
                            xHII(yHII), xHeII(yHeII), xHeIII(yHeIII), T_m, rs
                        )
                        + f_heating(rs, xHI, xHeI, xHeII(yHeII)) * dm_injection_rate(rs)
                    )
                    - phys.dtdz(rs) * (
                        + photoheat_total_rate
                        + reion.recomb_cooling_rate(
                            xHII(yHII), xHeII(yHeII), xHeIII(yHeIII), T_m, rs
                        )
                        + reion.coll_ion_cooling_rate(
                            xHII(yHII), xHeII(yHeII), xHeIII(yHeIII), T_m, rs
                        )
                        + reion.coll_exc_cooling_rate(
                            xHII(yHII), xHeII(yHeII), xHeIII(yHeIII), T_m, rs
                        )
                        + reion.brem_cooling_rate(
                            xHII(yHII), xHeII(yHeII), xHeIII(yHeIII), T_m, rs
                        )
                    )
                ) / (3/2 * phys.nH*rs**3 * (1 + chi + xe))
            )

        def dyHII_dz(yHII, yHeII, yHeIII, T_m, rs):

            xe = xHII(yHII) + xHeII(yHeII) + 2*xHeIII(yHeIII)
            ne = xe * phys.nH*rs**3
            xHI = 1 - xHII(yHII)
            xHeI = chi - xHeII(yHeII) - xHeIII(yHeIII)

            return 2 * np.cosh(yHII)**2 * -phys.dtdz(rs) * (
                # DM injection. Note that C = 1 at late times.
                + f_H_ion(rs, xHI, xHeI, xHeII(yHeII)) * dm_injection_rate(rs)
                    / (phys.rydberg * phys.nH * rs**3)
                + (1 - phys.peebles_C(xHII(yHII), rs)) * (
                    f_H_exc(rs, xHI, xHeI, xHeII(yHeII)) * dm_injection_rate(rs)
                    / (phys.lya_eng * phys.nH * rs**3)
                )
                # Reionization rates.
                + (
                    # Photoionization.
                    xHI * photoion_rate_HI(rs)
                    # Collisional ionization.
                    + xHI * ne * reion.coll_ion_rate('HI', T_m)
                    # Recombination.
                    - xHII(yHII) * ne * reion.alphaA_recomb('HII', T_m)
                )
            )

        def dyHeII_dz(yHII, yHeII, yHeIII, T_m, rs):
            xe = xHII(yHII) + xHeII(yHeII) + 2*xHeIII(yHeIII)
            ne = xe * phys.nH*rs**3
            xHeI = chi - xHeII(yHeII) - xHeIII(yHeIII)

            return 2/chi * np.cosh(yHeII)**2 * -phys.dtdz(rs) * (
                # Photoionization of HeI into HeII.
                xHeI * photoion_rate_HeI(rs)
                # Collisional ionization of HeI to HeII.
                + xHeI * ne * reion.coll_ion_rate('HeI', T_m)
                # Recombination of HeIII to HeII.
                + xHeIII(yHeIII) * ne * reion.alphaA_recomb('HeIII', T_m)
                # Photoionization of HeII to HeIII.
                - xHeII(yHeII) * photoion_rate_HeII(rs)
                # Collisional ionization of HeII to HeIII.
                - xHeII(yHeII) * ne * reion.coll_ion_rate('HeII', T_m)
                # Recombination of HeII into HeI.
                - xHeII(yHeII) * ne * reion.alphaA_recomb('HeII', T_m)
            )

        def dyHeIII_dz(yHII, yHeII, yHeIII, T_m, rs):
            xe = xHII(yHII) + xHeII(yHeII) + 2*xHeIII(yHeIII)
            ne = xe * phys.nH*rs**3

            return 2/chi * np.cosh(yHeIII)**2 * -phys.dtdz(rs) * (
                # Photoionization of HeII into HeIII.
                xHeII(yHeII) * photoion_rate_HeII(rs)
                # Collisional ionization of HeII into HeIII.
                + xHeII(yHeII) * ne * reion.coll_ion_rate('HeII', T_m)
                # Recombination of HeIII into HeII.
                - xHeIII(yHeIII) * ne * reion.alphaA_recomb('HeIII', T_m)
            )

        T_m, yHII, yHeII, yHeIII = var[0], var[1], var[2], var[3]

        return [
            dT_dz(yHII, yHeII, yHeIII, T_m, rs),
            dyHII_dz(yHII, yHeII, yHeIII, T_m, rs),
            dyHeII_dz(yHII, yHeII, yHeIII, T_m, rs),
            dyHeIII_dz(yHII, yHeII, yHeIII, T_m, rs)
        ]


    if init_cond[1] == 1:
        init_cond[1] = 1 - 1e-12
    if init_cond[2] == 0:
        init_cond[2] = 1e-12
    if init_cond[3] == 0:
        init_cond[3] = 1e-12

    init_cond[1] = np.arctanh(2*(init_cond[1] - 0.5))
    init_cond[2] = np.arctanh(2/chi * (init_cond[2] - chi/2))
    init_cond[3] = np.arctanh(2/chi *(init_cond[3] - chi/2))

    rs_before_reion = rs_vec[rs_vec > 16.1]
    rs_reion = rs_vec[rs_vec <= 16.1]

    if reion_switch:

        if rs_reion.size == 0:
            soln = odeint(
                tla_before_reion, init_cond, rs_before_reion, mxstep = 1000
            )

        elif rs_before_reion.size <= 1:
            if rs_before_reion.size == 1:
                # Covers the case where there is only 1 
                # redshift before reionization.
                rs_reion = np.insert(rs_reion, 0, rs_before_reion[0])
            soln = odeint(
                tla_reion, init_cond, rs_reion, mxstep = 1000
            )

        else:
            rs_reion = np.insert(rs_reion, 0, rs_before_reion[0])
            
            soln_before_reion = odeint(
                tla_before_reion, init_cond, rs_before_reion, mxstep = 500
            )

            init_cond_reion = [
                soln_before_reion[-1,0],
                soln_before_reion[-1,1],
                np.arctanh(2/(chi)*(1e-12 - chi/2)),
                np.arctanh(2/(chi)*(1e-12 - chi/2))
            ]

            soln_reion = odeint(
                tla_reion, init_cond_reion, rs_reion, mxstep = 1000,
            )

            soln = np.vstack((soln_before_reion[:-1,:], soln_reion))

    soln[:,1] = 0.5 + 0.5*np.tanh(soln[:,1])
    soln[:,2] = (
        chi/2 + chi/2*np.tanh(soln[:,2])
    )
    soln[:,3] = (
        chi/2 + chi/2*np.tanh(soln[:,3])
    )

    return soln
