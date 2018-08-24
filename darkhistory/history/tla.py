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
    dm_injection_rate_in, rs_vec, reion_switch=True, reion_rs=None,
    photoion_rate_func=None, photoheat_rate_func=None,
    mxstep = 1000
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
    dm_injection_rate_in : function or float
        Injection rate of DM as a function of redshift. Treated as constant if float.
    rs_vec : ndarray
        Abscissa for the solution.
    reion_switch : bool
        Reionization model included if true.
    reion_rs : float, optional
        Redshift 1+z at which reionization effects turn on.
    photoion_rate_func : tuple of functions, optional
        Functions take redshift 1+z as input, return the photoionization rate in s^-1 of HI, HeI and HeII respectively. If not specified, defaults to `darkhistory.history.reionization.photoion_rate`. 
    photoheat_rate_func : tuple of functions, optional
        Functions take redshift 1+z as input, return the photoheating rate in s^-1 of HI, HeI and HeII respectively. If not specified, defaults to `darkhistory.history.reionization.photoheat_rate`.  
    mxstep : int, optional
        Maximum number of (internally defined) steps allowed for each integration point in t. See scipy.integrate.odeint

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

    def dm_injection_rate(rs):
        if isinstance(dm_injection_rate_in, float):
            return dm_injection_rate_in
        elif callable(dm_injection_rate_in):
            return dm_injection_rate_in(rs)
        else:
            raise TypeError('dm_injection_rate_in must be a float or an appropriate function.')

    chi = phys.nHe/phys.nH

    if reion_switch:

        if photoion_rate_func is None:

            photoion_rate_HI   = reion.photoion_rate('HI')
            photoion_rate_HeI  = reion.photoion_rate('HeI')
            photoion_rate_HeII = reion.photoion_rate('HeII')

        else:

            photoion_rate_HI   = photoion_rate_func[0]
            photoion_rate_HeI  = photoion_rate_func[1]
            photoion_rate_HeII = photoion_rate_func[2]

    if reion_switch:

        if photoheat_rate_func is None:

            photoheat_rate_HI   = reion.photoheat_rate('HI')
            photoheat_rate_HeI  = reion.photoheat_rate('HeI')
            photoheat_rate_HeII = reion.photoheat_rate('HeII')

        else:

            photoheat_rate_HI   = photoheat_rate_func[0]
            photoheat_rate_HeI  = photoheat_rate_func[1]
            photoheat_rate_HeII = photoheat_rate_func[2]

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
                + f_H_ion(rs, xHI, xHeI, xHeII(yHeII)) * (
                    dm_injection_rate(rs)
                    / (phys.rydberg * phys.nH * rs**3)
                )
                + (1 - phys.peebles_C(xHII(yHII), rs)) * (
                    f_H_exc(rs, xHI, xHeI, xHeII(yHeII)) 
                    * dm_injection_rate(rs)
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

    if reion_rs is None:
        reion_rs = 16.1

    rs_before_reion_vec = rs_vec[rs_vec > reion_rs]
    rs_reion_vec = rs_vec[rs_vec <= reion_rs]

    if not reion_switch:
        # No reionization model implemented.
        soln = odeint(
                tla_before_reion, init_cond, rs_vec, mxstep = mxstep
            )
    else:
        # Reionization model implemented. 
        # First, check if required in the first place. 
        if rs_reion_vec.size == 0:
            soln = odeint(
                tla_before_reion, init_cond, 
                rs_before_reion_vec, mxstep = mxstep
            )
        # Conversely, solving before reionization may be unnecessary.
        elif rs_before_reion_vec.size == 0:
            soln = odeint(
                tla_reion, init_cond, rs_reion_vec, mxstep = mxstep
            )
        # Remaining case straddles both before and after reionization.
        else:
            # First, solve without reionization up to rs = reion_rs.
            rs_before_reion_vec = np.append(rs_before_reion_vec, reion_rs)
            soln_before_reion = odeint(
                tla_before_reion, init_cond, 
                rs_before_reion_vec, mxstep = mxstep
            )
            # Next, solve with reionization starting from reion_rs.
            rs_reion_vec = np.insert(rs_reion_vec, 0, reion_rs)
            # Initial conditions taken from last step before reionization.
            init_cond_reion = [
                soln_before_reion[-1,0],
                soln_before_reion[-1,1],
                np.arctanh(2/(chi)*(1e-12 - chi/2)),
                np.arctanh(2/(chi)*(1e-12 - chi/2))
            ]
            soln_reion = odeint(
                tla_reion, init_cond_reion, rs_reion_vec, mxstep = mxstep,
            )
            # Stack the solutions. Remove the solution at 16.1.
            soln = np.vstack((soln_before_reion[:-1,:], soln_reion[1:,:]))

    soln[:,1] = 0.5 + 0.5*np.tanh(soln[:,1])
    soln[:,2] = (
        chi/2 + chi/2*np.tanh(soln[:,2])
    )
    soln[:,3] = (
        chi/2 + chi/2*np.tanh(soln[:,3])
    )

    return soln
