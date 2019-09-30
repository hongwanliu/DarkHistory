"""Three-level atom model and integrator.

"""

import numpy as np
import darkhistory.physics as phys
import darkhistory.history.reionization as reion
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.misc import derivative
from scipy.special import erf
import sys

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

    Notes
    -----
    This is the energy loss rate, *not* the temperature loss rate.

    """
    xe = xHII + xHeII + 2*xHeIII

    return (
        4 * phys.thomson_xsec * 4 * phys.stefboltz / phys.me
        * xe * phys.nH*rs**3 * (phys.TCMB(rs) - T_m)
        * phys.TCMB(rs)**4
    )

def DM_IGM_cooling_rate(mDM, T_matter, T_DM, V_pec, xHII, rs, xsec, fDM, n, particle_type, mcharge_switch=True, eps=0):
    """Cooling rate for baryons from scattering with (possibly millicharged) DM.

    Parameters
    ----------
    mDM : float
        The dark matter mass in eV.
    T_matter : float
        The IGM temperature in eV.
    T_DM : float
        The dark matter temperature in eV.
    V_pec : float
        The dark matter peculiar velocity w.r.t. baryons, dimensionless.
    xHII : float
        The ionization fraction ne/nH.
    rs : float
        The redshift (1+z).
    xsec : float
        The dark matter-baryon interaction cross section in cm^2, excluding the velocity dependence
        The full cross section is xsec*velocity**n.
    fDM : float
        Ratio of millicharged dark matter to the full abundance.
    n : int
        The velocity dependence of the cross section.
    particle_type : {'matter', 'DM'}
        The particle type for which the cooling rate is to be specified. See 1708.08923 for details.
    mcharge_switch : bool, optional
        if True, compute the cooling rate for the millicharge model
    eps : float, optional
        The fraction of the charge of the millicharged DM.

    Returns
    -------
        The cooling rate in eV/s.

    """
    #See 1509.00029
    if mcharge_switch:
        if eps == 0:
            return 0

    mu_p = mDM*1.22*phys.mp/(mDM + 1.22*phys.mp)
    if xsec == None:
        xsec_0_p = None
    else:
        xsec_0_p = xsec/mu_p**2

    #u_th and r defined just after eqn. 13.  
    u_th_p = np.sqrt(T_matter/phys.mp + T_DM/mDM)
    r_p = V_pec/u_th_p
    #Eqn. 14
    F_p = erf(r_p/np.sqrt(2)) - np.sqrt(2/np.pi)*r_p*np.exp(-r_p**2/2)
    #print(u_th_p, " ", r_p, " ", F_p)
     
    #Put (13) into (16) into (18) || (19), then put this into rate_x to see that they indeed match
    drag_cooling_term_p  = np.divide(mDM*F_p, r_p, out=np.zeros_like(F_p), where=r_p!=0)
    drag_cooling_term_DM_p = np.divide(phys.mp*F_p, r_p, out=np.zeros_like(F_p), where=r_p!=0)
    #print(drag_cooling_term_p, " ", drag_cooling_term_DM_p)

    if mcharge_switch:
        #Everything's doubled, one for protons, one for electrons
        u_th_e = np.sqrt(T_matter/phys.me + T_DM/mDM)
        r_e = V_pec/u_th_e
        F_e = erf(r_e/np.sqrt(2)) - np.sqrt(2/np.pi)*r_e*np.exp(-r_e**2/2)

        drag_cooling_term_e  = np.divide(mDM*F_e, r_e, out=np.zeros_like(F_e), where=r_e!=0)
        drag_cooling_term_DM_e = np.divide(phys.me*F_e, r_e, out=np.zeros_like(F_e), where=r_e!=0)

        #Eqn 2, Munoz and Loeb
        xi = np.log(9*T_matter**3/(4*phys.hbar**3*phys.c**3*np.pi*eps**2*phys.alpha**3*xHII*phys.nH*rs**3))

        mu_p = mDM*phys.mp/(mDM + phys.mp)
        mu_e = mDM*phys.me/(mDM + phys.me)

        xsec_0_p = 2*np.pi*phys.alpha**2*eps**2*xi/mu_p**2*phys.hbar**2*phys.c**2
        xsec_0_e = 2*np.pi*phys.alpha**2*eps**2*xi/mu_e**2*phys.hbar**2*phys.c**2

        if particle_type == 'matter':
            rate_p = 2/(3*(1 + phys.nHe/phys.nH + xHII))*(fDM*phys.rho_DM*rs**3*phys.mp*xHII)/(mDM + phys.mp)**2*(
                    (xsec_0_p/u_th_p)*(
                        np.sqrt(2/np.pi)*np.exp(-r_p**2/2)*(T_DM - T_matter)/u_th_p**2 + drag_cooling_term_p
                    )
                )*phys.c

            rate_e = 2/(3*(1 + phys.nHe/phys.nH + xHII))*(fDM*phys.rho_DM*rs**3*phys.me*xHII)/(mDM + phys.me)**2*(
                    (xsec_0_e/u_th_e)*(
                        np.sqrt(2/np.pi)*np.exp(-r_e**2/2)*(T_DM - T_matter)/u_th_e**2 + drag_cooling_term_e
                    )
                )*phys.c

        elif particle_type == 'DM':
            rate_p = 2/3*(mDM*phys.mp*xHII*phys.nH*rs**3)/(mDM + phys.mp)**2*(
                (xsec_0_p/u_th_p)*(
                    np.sqrt(2/np.pi)*np.exp(-r_p**2/2)*(T_matter - T_DM)/u_th_p**2 + drag_cooling_term_DM_p
                )
            )*phys.c

            rate_e = 2/3*(mDM*phys.me*xHII*phys.nH*rs**3)/(mDM + phys.me)**2*(
                (xsec_0_e/u_th_e)*(
                    np.sqrt(2/np.pi)*np.exp(-r_e**2/2)*(T_matter - T_DM)/u_th_e**2 + drag_cooling_term_DM_e
                )
            )*phys.c
        else:
            raise TypeError('Invalid particle_type.')

    else:
        rate_e = 0
        if particle_type == 'matter':
            rate_p = 2/3*(
                fDM * phys.rho_DM*rs**3/1.08 * phys.mp/(mDM + phys.mp)**2 * xsec_p_0/u_th_p * (
                    np.sqrt(2/np.pi)*np.exp(-r_p**2/2)/u_th_p**2 * (T_DM - T_matter)
                    + drag_cooling_term_p
                )
            )*phys.c

        elif particle_type == 'DM':
            rate_p = 2/3*(
                phys.nH*rs**3 * (mDM*phys.mp/(mDM + phys.mp)**2) * xsec_p_0/u_th_p * (
                    np.sqrt(2/np.pi)*np.exp(-r_p**2/2)/u_th_p**2 * (T_matter - T_DM)
                    + drag_cooling_term_DM_p
                )
            )*phys.c

        else:
            raise TypeError('Invalid particle_type.')

    return rate_p + rate_e

def get_history(
    rs_vec, init_cond=None, baseline_f=False,
    inj_particle=None,
    f_H_ion=None, f_H_exc=None, f_heating=None,
    DM_process=None, mDM=None, sigmav=None, lifetime=None,
    struct_boost=None, injection_rate=None, 
    reion_switch=False, reion_rs=None,
    photoion_rate_func=None, photoheat_rate_func=None,
    xe_reion_func=None, helium_TLA=False, f_He_ion=None, 
    mxstep = 1000, rtol=1e-4,
    dm_baryon_switch=False, xsec=None, fDM=None, n=None, mcharge_switch=False, eps=0, z_td=None
):
    """Returns the ionization and thermal history of the IGM.

    Parameters
    ----------
    rs_vec : ndarray
        Abscissa for the solution.
    init_cond : array, optional
        Array containing [initial matter temperature, initla DM temperature, initial V_pec, initial xHII, initial xHeII, initial xHeIII]. Defaults to standard values if None.
    baseline_f : bool
        If True, uses the baseline f values with no backreaction returned by :func:`.f_std`. Default is False. 
    inj_particle : {'elec', 'phot'}, optional
        Specifies which set of f to use: electron/positron or photon. 
    f_H_ion : function or float, optional
        f(rs, x_HI, x_HeI, x_HeII) for hydrogen ionization. Treated as constant if float.
    f_H_exc : function or float, optional
        f(rs, x_HI, x_HeI, x_HeII) for hydrogen Lyman-alpha excitation. Treated as constant if float.
    f_heating : function or float, optional
        f(rs, x_HI, x_HeI, x_HeII) for heating. Treated as constant if float.
    DM_process : {'swave', 'decay'}, optional
        Dark matter process to use. Default is None.
    sigmav : float, optional
        Thermally averaged cross section for ``DM_process == 'swave'``. Default is None.
    lifetime : float, optional
        Decay lifetime for ``DM_process == 'decay'``. Default is None.
    struct_boost : function, optional
        Energy injection boost factor due to structure formation. Default is None.
    injection_rate : function or float, optional
        Injection rate of DM as a function of redshift. Treated as constant if float. Default is None. 
    reion_switch : bool
        Reionization model included if True.
    reion_rs : float, optional
        Redshift 1+z at which reionization effects turn on.
    photoion_rate_func : tuple of functions, optional
        Functions take redshift 1+z as input, return the photoionization rate in s^-1 of HI, HeI and HeII respectively. If not specified, defaults to `darkhistory.history.reionization.photoion_rate`. 
    photoheat_rate_func : tuple of functions, optional
        Functions take redshift 1+z as input, return the photoheating rate in s^-1 of HI, HeI and HeII respectively. If not specified, defaults to `darkhistory.history.reionization.photoheat_rate`. 
    xe_reion_func : function, optional
        Specifies a fixed ionization history after reion_rs.  
    helium_TLA : bool, optional
        Specifies whether to track helium before reionization. 
    f_He_ion : function or float, optional
        f(rs, x_HI, x_HeI, x_HeII) for helium ionization. Treated as constant if float. If None, treated as zero.
    mxstep : int, optional
        The maximum number of steps allowed for each integration point. See *scipy.integrate.odeint* for more information.
    rtol : float, optional
        The relative error of the solution. See *scipy.integrate.odeint* for more information.
    dm_baryon_switch : bool, optional
        DM baryon scattering included if True
    xsec : float, optional
        DM baryon scattering cross-section coefficient.  The cross-section is xsec*v^n/mDM^2.
    fDM : float, optional
        fraction of DM that can scatter with baryons
    n : int, optional
        power of velocity, v, with which the DM baryon scattering cross-section scales
    mcharge_switch : bool, optional
        millicharge switch. If True, assume the scattering dark matter is millicharged.
    eps : float, optional
        fraction of electric charge contained in a millicharged DM particle.
    z_td : float, optional
        redshift of thermal decoupling.  For all redshifts after z_dec, turn off Coulomb energy exchange.

    Returns
    -------
    list of ndarray
        [temperature solution (in eV), xHII solution, xHeII, xHeIII].

    Notes
    -----
    The actual differential equation that we solve is expressed in terms of y = arctanh(f*(x - f)), where f = 0.5 for x = xHII, and f = nHe/nH * 0.5 for x = xHeII or xHeIII, where nHe/nH is approximately 0.083.

    """

    # Defines the f(z) functions, which return a constant, 
    # if the input fz's are floats. 

    if (baseline_f) and (mDM == None):
        raise ValueError('Specify mDM to use baseline_f.')

    if baseline_f and (
        f_H_ion is not None or f_H_exc is not None
        or f_heating is not None
    ):
        raise ValueError('Use either baseline_f or specify f manually.')

    if baseline_f and (DM_process == 'swave'):
        struct_switch = True
    else:
        struct_switch = False

    #if DM_process=='pwave' and dm_baryon_switch:
    #    raise ValueError("No baseline f's generated for pwave models yet.")

    if dm_baryon_switch and reion_switch:
        raise ValueError("Have not implemented DM/baryon scattering in the presence of reionization yet.")

    if mcharge_switch:
        dm_baryon_switch = True

    def _f_H_ion(rs, xHI, xHeI, xHeII):
        if baseline_f: 
            return phys.f_std(
                mDM, rs, inj_particle=inj_particle, inj_type=DM_process, struct=struct_switch,
                channel='H ion'
            )
        if f_H_ion is None:
            return 0.
        elif callable(f_H_ion):
            return f_H_ion(rs, xHI, xHeI, xHeII)
        else:
            return f_H_ion

    def _f_H_exc(rs, xHI, xHeI, xHeII):
        if baseline_f: 
            return phys.f_std(
                mDM, rs, inj_particle=inj_particle, inj_type=DM_process, struct=struct_switch,
                channel='exc'
            )
        if f_H_exc is None:
            return 0.
        elif callable(f_H_exc):
            return f_H_exc(rs, xHI, xHeI, xHeII)
        else:
            return f_H_exc

    def _f_heating(rs, xHI, xHeI, xHeII):
        if baseline_f: 
            return phys.f_std(
                mDM, rs, inj_particle=inj_particle, inj_type=DM_process, struct=struct_switch,
                channel='heat'
            )
        if f_heating is None:
            return 0.
        elif callable(f_heating):
            return f_heating(rs, xHI, xHeI, xHeII)
        else:
            return f_heating
        
    def _f_He_ion(rs, xHI, xHeI, xHeII):
        if f_He_ion is None:
            return 0.
        elif callable(f_He_ion):
            return f_He_ion(rs, xHI, xHeI, xHeII)
        else:
            return f_He_ion

    if (DM_process == 'swave' or DM_process == 'pwave')  and ((sigmav is None and eps is None) or mDM is None):
        raise ValueError('sigmav, mDM must be specified for '+DM_process+'.')
    if DM_process == 'decay' and (lifetime is None or mDM is None):
        raise ValueError('lifetime, mDM must be specified for decay.')
    #if DM_process is not None and injection_rate is not None:
    #    raise ValueError(
    #        'cannot specify both DM_process and injection_rate.'
    #    )

    #Eqn 2, millicharged model
    if mcharge_switch:
        sigmav = (np.pi*phys.alpha**2*eps**2)/mDM**2 *(
            np.sqrt(1 - phys.me**2/mDM**2) * (1 + phys.me**2/(2*mDM**2))
        )*phys.hbar**2*phys.c**3 / 6

    chi = phys.chi

    # Initial Condition
    if init_cond is None:
        rs_start = rs_vec[0]
        sigma_1D_over_c = 1e-11*(1/100)**0.5 * rs_start
        if helium_TLA:
            _init_cond = [
                phys.Tm_std(rs_start), 
                mDM * sigma_1D_over_c**2 / 3, #T_DM
                0, #V_pec !!!
                phys.xHII_std(rs_start), 
                phys.xHeII_std(rs_start), 
                1e-12
            ]

        else:
            _init_cond = [
                phys.Tm_std(rs_start), 
                mDM * sigma_1D_over_c**2 / 3, #T_DM
                0, #V_pec !!!
                phys.xHII_std(rs_start), 
                1e-12, 
                1e-12
            ]
    else:
        _init_cond = np.array(init_cond)

        if init_cond[-3]   == 1:
            _init_cond[-3] = 1 - 1e-12
        if init_cond[-2]   == 0:
            _init_cond[-2] = 1e-12
        elif init_cond[-2] == chi:
            _init_cond[-2] = (1. - 1e-12) * chi
        if init_cond[-1]   == 0:
            _init_cond[-1] = 1e-12

    _init_cond[0] = np.log(_init_cond[0])
    _init_cond[1] = np.log(_init_cond[1])
    _init_cond[-3] = np.arctanh(2*(_init_cond[-3] - 0.5))
    _init_cond[-2] = np.arctanh(2/chi * (_init_cond[-2] - chi/2))
    _init_cond[-1] = np.arctanh(2/chi *(_init_cond[-1] - chi/2))

    # struct_boost should be defined to just return 1 if undefined.
    if struct_boost is None:
        def struct_boost(rs): 
            return 1.

    def _injection_rate(rs):
        if callable(injection_rate):
            return injection_rate(rs)
        else:
            if (DM_process == 'swave') or (DM_process == 'pwave'):
                return (
                    fDM**2 * phys.inj_rate(DM_process, rs, mDM=mDM, sigmav=sigmav) 
                    * struct_boost(rs)
                )
            elif DM_process == 'decay':
                return phys.inj_rate('decay', rs, mDM=mDM, lifetime=lifetime)
            elif injection_rate is None:
                return 0

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

    # Define conversion functions between x and y. 
    def xHII(yHII):
            return 0.5 + 0.5*np.tanh(yHII)
    def xHeII(yHeII):
        return chi/2 + chi/2*np.tanh(yHeII)
    def xHeIII(yHeIII):
        return chi/2 + chi/2*np.tanh(yHeIII)

    def tla_before_reion(rs, var):
        # Returns an array of values for [dT/dz, dyHII/dz,
        # dyHeII/dz, dyHeIII/dz].
        # var is the [log_T_m, log_T_DM, V_pec, xHII, xHeII, xHeIII] inputs.

        nH = phys.nH*rs**3
        inj_rate = _injection_rate(rs)
        fac_td = 1
        if z_td is not None:
            if 1+z_td >= rs:
                fac_td=0


        def dlogT_dz(yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs):

            T_m  = np.exp(log_T_m)
            T_DM = np.exp(log_T_DM)

            xe   = xHII(yHII) + xHeII(yHeII) + 2*xHeIII(yHeIII)
            xHI  = 1 - xHII(yHII)
            xHeI = chi - xHeII(yHeII) - xHeIII(yHeIII)

            # This rate is temperature loss per redshift.
            adiabatic_cooling_rate = 2 * T_m/rs
            if dm_baryon_switch:
                baryon_dm_cooling_rate = (phys.dtdz(rs) * DM_IGM_cooling_rate(
                    mDM, T_m, T_DM, V_pec, xHII(yHII), rs,
                    xsec, fDM, n, particle_type='matter', mcharge_switch=mcharge_switch, eps=eps
                    )
                )
            else:
                baryon_dm_cooling_rate = 0


            dT = (1 / T_m * adiabatic_cooling_rate + 1/T_m * baryon_dm_cooling_rate + 1/T_m * (
                    phys.dtdz(rs)*(
                        fac_td*compton_cooling_rate(
                            xHII(yHII), xHeII(yHeII), xHeIII(yHeIII), T_m, rs
                        )
                        + _f_heating(rs, xHI, xHeI, xHeII(yHeII)) * inj_rate
                    )
                )/ (3/2 * nH * (1 + chi + xe))
            )
            #print(1/T_m * phys.dtdz(rs)*(compton_cooling_rate(xHII(yHII), xHeII(yHeII), xHeIII(yHeIII), T_m, rs)/(3/2 * nH * (1 + chi + xe))))
            return dT

        def dlogTDM_dz(yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs):

            if dm_baryon_switch:
                T_m  = np.exp(log_T_m)
                T_DM = np.exp(log_T_DM)

                xe   = xHII(yHII) + xHeII(yHeII) + 2*xHeIII(yHeIII)
                xHI  = 1 - xHII(yHII)
                xHeI = chi - xHeII(yHeII) - xHeIII(yHeIII)

                # This rate is temperature loss per redshift.
                adiabatic_cooling_rate    = 2 * T_DM/rs
                dm_baryon_cooling_rate = (phys.dtdz(rs) * DM_IGM_cooling_rate(
                    mDM, T_m, T_DM, V_pec, xHII(yHII), rs, xsec, fDM, n, particle_type='DM', mcharge_switch=mcharge_switch, eps=eps
                    )
                )

                return 1 / T_DM * (adiabatic_cooling_rate + dm_baryon_cooling_rate)
            else:
                return 0

        def dV_pec_dz(yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs):
            
            T_m = np.exp(log_T_m)
            T_DM = np.exp(log_T_DM)
            if V_pec == 0:
                return 0
            elif dm_baryon_switch or mcharge_switch:
                if mcharge_switch:
                    if eps == 0:
                        return 0

                    xi = np.log(9*T_m**3/(4*phys.hbar**3*phys.c**3*np.pi*eps**2*phys.alpha**3*xHII(yHII)*phys.nH*rs**3))

                    mu_p = mDM*phys.mp/(mDM + phys.mp)
                    mu_e = mDM*phys.me/(mDM + phys.me)

                    xsec_0_p = 2*np.pi*phys.alpha**2*eps**2*xi/mu_p**2*phys.hbar**2*phys.c**2
                    xsec_0_e = 2*np.pi*phys.alpha**2*eps**2*xi/mu_e**2*phys.hbar**2*phys.c**2
                    
                    u_th_e = np.sqrt(T_m/phys.me + T_DM/mDM)
                    r_e = V_pec/u_th_e
                    F_e = erf(r_e/np.sqrt(2)) - np.sqrt(2/np.pi)*r_e*np.exp(-r_e**2/2)

                    D_e = (1 + fDM*phys.rho_DM/phys.rho_baryon)*(
                          xsec_0_e*phys.me*xHII(yHII)*phys.nH*rs**3/(mDM + phys.me)*F_e/V_pec**2
                    )*phys.c
                else:
                    xsec_0_p = xsec
                    D_e = 0
                    
                T_m  = np.exp(log_T_m)
                T_DM = np.exp(log_T_DM)

                u_th_p = np.sqrt(T_m/phys.mp + T_DM/mDM)
                r_p = V_pec/u_th_p
                F_p = erf(r_p/np.sqrt(2)) - np.sqrt(2/np.pi)*r_p*np.exp(-r_p**2/2)

                D_p = (1 + fDM*phys.rho_DM/phys.rho_baryon)*(
                      xsec_0_p*phys.mp*phys.nH*rs**3/(mDM + phys.mp)*F_p/V_pec**2
                )*phys.c

                return -phys.dtdz(rs)*(
                    - phys.hubble(rs)*V_pec - D_p - D_e
                )
            else:
                return 0

            

        def dyHII_dz(yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs):

            T_m = np.exp(log_T_m)
            T_DM = np.exp(log_T_DM)

            if 1 - xHII(yHII) < 1e-6 and rs < 100:
                # At this point, leave at 1 - 1e-6
                return 0
            # if yHII > 14. or yHII < -14.:
            #     # Stops the solver from wandering too far.
            #     return 0    
            if xHeII(yHeII) > 0.99*chi and rs > 1500:
                # This is prior to helium recombination.
                # Assume H completely ionized.
                return 0

            if helium_TLA and xHII(yHII) > 0.999 and rs > 1500:
                # Use the Saha value. 
                return 2 * np.cosh(yHII)**2 * phys.d_xe_Saha_dz(rs, 'HI')

            if not helium_TLA and xHII(yHII) > 0.99 and rs > 1500:
                # Use the Saha value. 
                return 2 * np.cosh(yHII)**2 * phys.d_xe_Saha_dz(rs, 'HI')


            xe = xHII(yHII) + xHeII(yHeII) + 2*xHeIII(yHeIII)
            ne = xe * nH
            xHI = 1 - xHII(yHII)
            xHeI = chi - xHeII(yHeII) - xHeIII(yHeIII)

            return 2 * np.cosh(yHII)**2 * phys.dtdz(rs) * (
                # Recombination processes. 
                # Boltzmann factor is T_r, agrees with HyREC paper.
                - phys.peebles_C(xHII(yHII), rs) * (
                    phys.alpha_recomb(T_m, 'HI') * xHII(yHII) * xe * nH
                    - 4*phys.beta_ion(phys.TCMB(rs), 'HI') * xHI
                        * np.exp(-phys.lya_eng/phys.TCMB(rs))
                )
                # DM injection. Note that C = 1 at late times.
                + _f_H_ion(rs, xHI, xHeI, xHeII(yHeII)) * inj_rate
                    / (phys.rydberg * nH)
                + (1 - phys.peebles_C(xHII(yHII), rs)) * (
                    _f_H_exc(rs, xHI, xHeI, xHeII(yHeII)) * inj_rate
                    / (phys.lya_eng * nH)
                )
            )

        def dyHeII_dz(yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs):

            T_m = np.exp(log_T_m)

            if not helium_TLA: 

                return 0

            if chi - xHeII(yHeII) < 1e-6 and rs < 100:
                # At this point, leave at 1 - 1e-6
                return 0
            
            # Stop the solver from reaching these extremes. 
            if yHeII > 14 or yHeII < -14:
                return 0

            # # Use the Saha values at high ionization. 
            # if xHeII(yHeII) > 0.995*chi: 

            #     # print(phys.d_xe_Saha_dz(rs, 'HeI'))

            #     return (
            #         2/chi * np.cosh(yHeII)**2 * phys.d_xe_Saha_dz(rs, 'HeI')
            #     )

            xe = xHII(yHII) + xHeII(yHeII) + 2*xHeIII(yHeIII)
            ne = xe * nH
            xHI = 1 - xHII(yHII)
            xHeI = chi - xHeII(yHeII) - xHeIII(yHeIII)

            term_recomb_singlet = (
                xHeII(yHeII) * xe * nH * phys.alpha_recomb(T_m, 'HeI_21s')
            )
            term_ion_singlet = (
                phys.beta_ion(phys.TCMB(rs), 'HeI_21s')*(chi - xHeII(yHeII))
                * np.exp(-phys.He_exc_eng['21s']/phys.TCMB(rs))
            )

            term_recomb_triplet = (
                xHeII(yHeII) * xe * nH * phys.alpha_recomb(T_m, 'HeI_23s')
            )
            term_ion_triplet = (
                3*phys.beta_ion(phys.TCMB(rs), 'HeI_23s') 
                * (chi - xHeII(yHeII)) 
                * np.exp(-phys.He_exc_eng['23s']/phys.TCMB(rs))
            )

            return 2/chi * np.cosh(yHeII)**2 * phys.dtdz(rs) * (
                -phys.C_He(xHII(yHII), xHeII(yHeII), rs, 'singlet') * (
                    term_recomb_singlet - term_ion_singlet
                )
                -phys.C_He(xHII(yHII), xHeII(yHeII), rs, 'triplet') * (
                    term_recomb_triplet - term_ion_triplet
                )
                + _f_He_ion(rs, xHI, xHeI, xHeII(yHeII)) * inj_rate
                    / (phys.He_ion_eng * nH)
            )

        def dyHeIII_dz(yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs):

            if chi - xHeIII(yHeIII) < 1e-6 and rs < 100:
                # At this point, leave at 1 - 1e-6
                return 0

            xe = xHII(yHII) + xHeII(yHeII) + 2*xHeIII(yHeIII)
            ne = xe * nH

            return 0

        log_T_m, log_T_DM, V_pec, yHII, yHeII, yHeIII = var[0], var[1], var[2], var[3], var[4], var[5]
        #print(rs, np.exp(log_T_m), np.exp(log_T_DM), V_pec, xHII(yHII))

        #print ([rs, 
        #    dlogT_dz(yHII, yHeII, yHeIII, log_T_m, rs),
        #    dyHII_dz(yHII, yHeII, yHeIII, log_T_m, rs),
        #    dyHeII_dz(yHII, yHeII, yHeIII, log_T_m, rs),
        #    dyHeIII_dz(yHII, yHeII, yHeIII, log_T_m, rs)
        #])
        # print(rs, phys.peebles_C(xHII(yHII), rs))


        #print(rs, log_T_m, xHII(yHII), xHeII(yHeII), xHeIII(yHeIII))

        return [
            dlogT_dz(  yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs),
            dlogTDM_dz(yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs),
            dV_pec_dz( yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs),
            dyHII_dz(  yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs),
            dyHeII_dz( yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs),
            dyHeIII_dz(yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs)
        ]

    def tla_reion(rs, var):
        # TLA with photoionization/photoheating reionization model.
        # Returns an array of values for [dT/dz, dyHII/dz,
        # dyHeII/dz, dyHeIII/dz].
        # var is the [T_m, T_DM, V_pec, xHII, xHeII, xHeIII] inputs.

        sys.exit()
        inj_rate = _injection_rate(rs)
        nH = phys.nH*rs**3

        #def dlogT_dz(yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs):
        def dlogT_dz(yHII, yHeII, yHeIII, log_T_m, V_pec, rs):

            T_m = np.exp(log_T_m)
            #T_DM = np.exp(log_T_DM)

            xe = xHII(yHII) + xHeII(yHeII) + 2*xHeIII(yHeIII)
            xHI = 1 - xHII(yHII)
            xHeI = chi - xHeII(yHeII) - xHeIII(yHeIII)

            # This rate is temperature loss per redshift.
            adiabatic_cooling_rate = 2 * T_m/rs

            # The reionization rates and the Compton rate
            # are expressed in *energy loss* *per second*.

            photoheat_total_rate = nH * (
                xHI * photoheat_rate_HI(rs)
                + xHeI * photoheat_rate_HeI(rs)
                + xHeII(yHeII) * photoheat_rate_HeII(rs)
            )

            compton_rate = phys.dtdz(rs)*(
                compton_cooling_rate(
                    xHII(yHII), xHeII(yHeII), xHeIII(yHeIII), T_m, rs
                )
            ) / (3/2 * nH * (1 + chi + xe))

            dm_heating_rate = phys.dtdz(rs)*(
                _f_heating(rs, xHI, xHeI, xHeII(yHeII)) * inj_rate
            ) / (3/2 * nH * (1 + chi + xe))
            
            dm_baryon_cooling_rate = fDM*switch*phys.dtdez(rs)*(
                            DM_IGM_cooling_rate(
                            mDM, T_m, T_DM, V_pec, xe(y), rs,
                            xsec, fDM, n, particle_type='DM'
                        )
                    )

            reion_rate = phys.dtdz(rs) * (
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
            ) / (3/2 * nH * (1 + chi + xe))

            return 1 / T_m * (
                adiabatic_cooling_rate + compton_rate 
                + dm_heating_rate + dm_baryon_cooling_rate + reion_rate
            )

        #def dV_pec_dz(yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs):
        def dV_pec_dz(yHII, yHeII, yHeIII, log_T_m, V_pec, rs):
            
            T_m    = np.exp(log_T_m)
            T_DM   = np.exp(log_T_DM)
            u_th_p = np.sqrt(T_m/phys.mp + T_DM/mDM)
            r_p    = V_pec/u_th_p
            F_p    = erf(r_p/np.sqrt(2)) - np.sqrt(2/np.pi)*r_p*np.exp(-r_p**2/2)

            if V_pec == 0:
                return 0
            else:

                D = (1 + fDM*phys.rho_DM/phys.rho_baryon)*(
                      xsec*phys.mp*phys.nH*rs**3/(mDM + phys.mp)*F_p/V_pec**2
                )*phys.c

                return -phys.dtdz(rs)*(
                    - phys.hubble(rs)*V_pec - D
                )


        #def dyHII_dz(yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs):
        def dyHII_dz(yHII, yHeII, yHeIII, log_T_m, V_pec, rs):

            T_m = np.exp(log_T_m)

            if 1 - xHII(yHII) < 1e-6 and rs < 100:
                # At this point, leave at 1 - 1e-6
                return 0


            xe = xHII(yHII) + xHeII(yHeII) + 2*xHeIII(yHeIII)
            ne = xe * nH
            xHI = 1 - xHII(yHII)
            xHeI = chi - xHeII(yHeII) - xHeIII(yHeIII)

            return 2 * np.cosh(yHII)**2 * phys.dtdz(rs) * (
                # DM injection. Note that C = 1 at late times.
                + _f_H_ion(rs, xHI, xHeI, xHeII(yHeII)) * (
                    inj_rate / (phys.rydberg * nH)
                )
                + (1 - phys.peebles_C(xHII(yHII), rs)) * (
                    _f_H_exc(rs, xHI, xHeI, xHeII(yHeII)) 
                    * inj_rate / (phys.lya_eng * nH)
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

        #def dyHeII_dz(yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs):
        def dyHeII_dz(yHII, yHeII, yHeIII, log_T_m, V_pec, rs):

            T_m = np.exp(log_T_m)

            if chi - xHeII(yHeII) < 1e-6 and rs < 100:
                # At this point, leave at 1 - 1e-6
                return 0

            xe = xHII(yHII) + xHeII(yHeII) + 2*xHeIII(yHeIII)
            ne = xe * nH
            xHI = 1 - xHII(yHII)
            xHeI = chi - xHeII(yHeII) - xHeIII(yHeIII)

            return 2/chi * np.cosh(yHeII)**2 * phys.dtdz(rs) * (
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
                # DM contribution
                + _f_He_ion(rs, xHI, xHeI, xHeII(yHeII)) * inj_rate
                    / (phys.He_ion_eng * nH)
            )

        def dyHeIII_dz(yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs):

            T_m = np.exp(log_T_m)

            if chi - xHeIII(yHeIII) < 1e-6 and rs < 100:
                # At this point, leave at 1 - 1e-6
                return 0

            xe = xHII(yHII) + xHeII(yHeII) + 2*xHeIII(yHeIII)
            ne = xe * nH

            return 2/chi * np.cosh(yHeIII)**2 * phys.dtdz(rs) * (
                # Photoionization of HeII into HeIII.
                xHeII(yHeII) * photoion_rate_HeII(rs)
                # Collisional ionization of HeII into HeIII.
                + xHeII(yHeII) * ne * reion.coll_ion_rate('HeII', T_m)
                # Recombination of HeIII into HeII.
                - xHeIII(yHeIII) * ne * reion.alphaA_recomb('HeIII', T_m)
            )

        log_T_m, log_T_DM, V_pec, yHII, yHeII, yHeIII = var[0], var[1], var[2], var[3], var[4], var[5]

        # print(rs, T_m, xHII(yHII), xHeII(yHeII), xHeIII(yHeIII))
        
        return [
            dlogT_dz(   yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs),
            dlogTDM_dz( yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs),
            dV_pec_dz(  yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs),
            dyHII_dz(   yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs),
            dyHeII_dz(  yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs),
            dyHeIII_dz( yHII, yHeII, yHeIII, log_T_m, log_T_DM, V_pec, rs)
        ]

    def tla_reion_fixed_xe(rs, var):
        # TLA with fixed ionization history. 
        # Returns an array of values for [dT/dz, dyHII/dz].]. 
        # var is the [T_m, T_DM, V_pec] input.

        sys.exit()
        def dxe_dz(rs):

            return derivative(xe_reion_func, rs)

        def dlogT_dz(log_T_m, log_T_DM, V_pec, rs):

            T_m = np.exp(log_T_m)
            T_DM = np.exp(log_T_DM)

            xe    = xe_reion_func(rs)
            xHII  = xe * (1. / (1. + chi))
            xHeII = xe * (chi / (1. + chi)) 
            xHI   = 1. - xHII
            xHeI  = chi - xHeII


            # This is the temperature loss per redshift. 
            adiabatic_cooling_rate = 2 * T_m/rs

            return 1 / T_m * (
                adiabatic_cooling_rate
                + (
                    phys.dtdz(rs)*(
                        compton_cooling_rate(xHII, xHeII, 0, T_m, rs)
                        + _f_heating(rs, xHI, xHeI, 0) * _injection_rate(rs)
                    )
                )/ (3/2 * nH * (1 + chi + xe)) 
                - switch*(
                        phys.dtdz(rs) * DM_IGM_cooling_rate(
                        mDM, T_m, T_DM, V_pec, xHII(yHII), rs,
                        xsec, fDM, n, particle_type='matter'
                        )
                    )
            )

        log_T_m, log_T_DM, V_pec = var[0], var[1], var[2]

        return dlogT_dz(log_T_m, log_T_DM, V_pec, rs)


    if reion_rs is None: 
        if photoion_rate_func is None and xe_reion_func is None:
            # Default Puchwein model value.
            reion_rs = 16.1
        else:
            raise TypeError('must specify reion_rs if not using default.')


    rs_before_reion_vec = rs_vec[rs_vec > reion_rs]
    rs_reion_vec = rs_vec[rs_vec <= reion_rs]

    if not reion_switch:
        # No reionization model implemented.
        soln = odeint(
                tla_before_reion, _init_cond, rs_vec, 
                mxstep = mxstep, tfirst=True, rtol=rtol
            )
        # print(init_cond)
        # print(rs_vec)
        # soln = solve_ivp(
        #     tla_before_reion, [rs_vec[0], rs_vec[-1]],
        #     init_cond, method='Radau'
        # )
        # print(soln)
    elif xe_reion_func is not None:
        # Fixed xe reionization model implemented. 
        # First, solve without reionization.

        # tfirst=True means that tla_before_reion accepts rs as 
        # first argument.
        soln_no_reion = odeint(
            tla_before_reion, _init_cond, rs_vec, 
            mxstep = mxstep, tfirst=True, rtol=rtol
        )
        # soln_no_reion = solve_ivp(
        #     tla_before_reion, (rs_vec[0], rs_vec[-1]),
        #     init_cond, method='BDF', t_eval=rs_vec
        # )
        # Check if reionization model is required in the first place.
        if rs_reion_vec.size == 0:
            soln = soln_no_reion
            # Convert to xe
            soln[:,3] = 0.5 + 0.5*np.tanh(soln[:,3])
            soln[:,4] = chi/2 + chi/2*np.tanh(soln[:,4])
            soln[:,5] = chi/2 + chi/2*np.tanh(soln[:,5])
        else:
            xHII_no_reion   = 0.5 + 0.5*np.tanh(soln_no_reion[:,3])
            xHeII_no_reion  = chi/2 + chi/2*np.tanh(soln_no_reion[:,4])
            xHeIII_no_reion = chi/2 + chi/2*np.tanh(soln_no_reion[:,5])
            
            xe_no_reion = xHII_no_reion + xHeII_no_reion + xHeIII_no_reion

            xe_reion    = xe_reion_func(rs_vec)
            # Find where to solve the TLA. Must lie below reion_rs and 
            # have xe_reion > xe_no_reion.

            # Earliest redshift index where xe_reion > xe_no_reion. 
            # min because redshift is in decreasing order.
            where_xe = np.min(np.argwhere(xe_reion > xe_no_reion))
            # Redshift index where rs_vec < reion_rs. 
            where_rs = np.min(np.argwhere(rs_vec < reion_rs))
            # Start at the later redshift, i.e. the larger index. 
            where_start = np.max([where_xe, where_rs])
            # Define the boolean mask.
            where_new_soln = (np.arange(rs_vec.size) >= where_start)


            # Find the respective redshift arrays. 
            rs_above_std_xe_vec = rs_vec[where_new_soln]
            rs_below_std_xe_vec = rs_vec[~where_new_soln]
            # Append the last redshift before reionization model. 
            rs_above_std_xe_vec = np.insert(
                rs_above_std_xe_vec, 0, rs_below_std_xe_vec[-1]
            )

            # Define the solution array. Get the entries from soln_no_reion.
            soln = np.zeros_like(soln_no_reion)
            # Copy Tm, xHII, xHeII only before reionization.
            soln[~where_new_soln, :3] = soln_no_reion[~where_new_soln, :3]
            # Copy xHeIII entirely with no reionization for now.
            soln[:, 3] = soln_no_reion[:, 3]
            # Convert to xe.
            soln[~where_new_soln, 1] = 0.5 + 0.5*np.tanh(
                soln[~where_new_soln, 1]
            )
            soln[~where_new_soln, 2] = chi/2 + chi/2*np.tanh(
                soln[~where_new_soln, 2]
            )
            soln[:, 3] = chi/2 + chi/2*np.tanh(soln[:, 3])


            # Solve for all subsequent redshifts. 
            if rs_above_std_xe_vec.size > 0:
                init_cond_fixed_xe = soln[~where_new_soln, 0][-1]
                soln_with_reion = odeint(
                    tla_reion_fixed_xe, init_cond_fixed_xe, 
                    rs_above_std_xe_vec, mxstep=mxstep, rtol=rtol, 
                    tfirst=True
                )
                # Remove the initial step, save to soln.
                soln[where_new_soln, 0] = np.squeeze(soln_with_reion[1:])
                # Put in the solutions for xHII and xHeII. 
                soln[where_new_soln, 1] = xe_reion_func(
                    rs_vec[where_new_soln]
                ) * (1. / (1. + phys.chi))
                soln[where_new_soln, 2] = xe_reion_func(
                    rs_vec[where_new_soln]
                ) * (phys.chi / (1. + phys.chi))

        # Convert from log_T_m to T_m
        soln[:,0] = np.exp(soln[:,0])
        soln[:,1] = np.exp(soln[:,1])

        return soln

    else:
        # Reionization model implemented. 
        # First, check if required in the first place. 
        if rs_reion_vec.size == 0:
            soln = odeint(
                tla_before_reion, _init_cond, 
                rs_before_reion_vec, mxstep = mxstep, rtol=rtol, tfirst=True
            )
            # soln = solve_ivp(
            #     tla_before_reion, 
            #     (rs_before_reion_vec[0], rs_before_reion_vec[-1]),
            #     init_cond, method='BDF', t_eval=rs_before_reion_vec
            # )
        # Conversely, solving before reionization may be unnecessary.
        elif rs_before_reion_vec.size == 0:
            soln = odeint(
                tla_reion, _init_cond, rs_reion_vec, 
                mxstep = mxstep, rtol=rtol, tfirst=True
            )
            # soln = solve_ivp(
            #     tla_reion, (rs_reion_vec[0], rs_reion_vec[-1]),
            #     init_cond, method='BDF', t_eval=rs_reion_vec
            # )
        # Remaining case straddles both before and after reionization.
        else:
            # First, solve without reionization up to rs = reion_rs.
            rs_before_reion_vec = np.append(rs_before_reion_vec, reion_rs)
            soln_before_reion = odeint(
                tla_before_reion, _init_cond, 
                rs_before_reion_vec, mxstep = mxstep, tfirst=True, rtol=rtol
            )
            # soln_before_reion = solve_ivp(
            #     tla_before_reion, 
            #     (rs_before_reion_vec[0], rs_before_reion_vec[-1]),
            #     init_cond, method='BDF', t_eval=rs_before_reion_vec
            # )
            # Next, solve with reionization starting from reion_rs.
            rs_reion_vec = np.insert(rs_reion_vec, 0, reion_rs)
            # Initial conditions taken from last step before reionization.
            init_cond_reion = [
                soln_before_reion[-1,0],
                soln_before_reion[-1,1],
                soln_before_reion[-1,2],
                soln_before_reion[-1,3]
            ]
            soln_reion = odeint(
                tla_reion, init_cond_reion, 
                rs_reion_vec, mxstep = mxstep, tfirst=True, rtol=rtol
            )
            # soln_reion = solve_ivp(
            #     tla_reion, (rs_reion_vec[0], rs_reion_vec[-1]),
            #     init_cond, method='BDF', t_eval=rs_reion_vec
            # )
            # Stack the solutions. Remove the solution at 16.1.
            soln = np.vstack((soln_before_reion[:-1,:], soln_reion[1:,:]))

    soln[:,0] = np.exp(soln[:,0])
    soln[:,1] = np.exp(soln[:,1])
    soln[:,-3] = 0.5 + 0.5*np.tanh(soln[:,-3])
    soln[:,-2] = (
        chi/2 + chi/2*np.tanh(soln[:,-2])
    )
    soln[:,-1] = (
        chi/2 + chi/2*np.tanh(soln[:,-1])
    )

    return soln
