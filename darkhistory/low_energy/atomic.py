"""H excitation state transitions, ionization, and recombination functions."""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

from darkhistory import physics as phys
from darkhistory.spec.spectrum import Spectrum
import darkhistory.spec.spectools as spectools
import scipy.special
import scipy.sparse as sp

import darkhistory.low_energy.bound_free as bf

NBINS = 100
Nkappa = 10 * NBINS + 1

hplanck = phys.hbar*2*np.pi
mu_e = phys.me/(1+phys.me/phys.mp)
lgamma = scipy.special.loggamma


#####################
# 2s->1s transition #
#####################

# Define coefficient of integral
A0 = 9 * phys.alpha**6 * (phys.rydberg/hplanck) / 2**10  # ~4.3663 s^-1

# Newton-Cotes 11-point integration weights
NC11_weights = 5. / 299376. * np.array([
    16067., 106300., -48525., 272400., -260550.,
    427368.,
    -260550., 272400., -48525., 106300., 16067.
])


def f_BB(E, Tr):
    """
    Blackbody phase space density
    """
    return np.exp(-E/Tr) / (1 - np.exp(-E/Tr))


def phi(y):
    """
    Proportional to the probability density of emitting one photon at
    dimensionless frequency y. Normalizes to 2 since two photons are emitted.
    When integrated and multiplied by A0/2, evaluates to the 2s->1s decay rate.

    Notes
    -----
    See Eq. 4 of astro-ph/0508144
    """

    if isinstance(y, float):
        y = np.array([y])

    w = y*(1-y)  # y = eng / phys.lya_eng
    C, alp, bet, gam = 46.26, 0.88, 1.53, 0.8
    res = C * (w * (1 - 4**gam * w**gam) + alp * w**(bet + gam) * 4**gam)

    res[(y <= 0) | (y >= 1)] = 0  # deal with energies above E_lya

    return res


def A_2s1s(f_gamma, direc='dn', use_quad=False, n_bins=100):
    """
    Function that takes photon spectrum and energies as input
    and returns 2s to 1s transition rate

    Parameters
    ----------
    f_gamma : function
        f_gamma(E) = photon phase space density evaluated at energy E
    direc : string
        if 'dn', return A_{2s->1s}, else return A_{1s->2s}
    use_quad : bool
        if True, use scipy.integrate.quad,
        else use Newton-Cotes 11-point integration, which is faster
    n_bins : int
        subdivide the integration interval this many times,
        perform NC11 integration on each subdivision

    Notes
    -----
    See Eq. 2 of astro-ph/0508144
    set quad=False if detailed balance is important (within machine precision)
    """

    fac = 0
    if direc == 'dn':
        fac = 1

    if not use_quad:

        # can't start the integration right at E=0, so start at E=eps
        eps = 1e-12

        engs = np.linspace(eps, phys.lya_eng-eps, n_bins*(11-1) + 1)
        dy = (engs[1]-engs[0]) / phys.lya_eng

        integrand = (
            phi(engs / phys.lya_eng) *
            (fac + f_gamma(engs)) *
            (fac + f_gamma(phys.lya_eng - engs))
        )

        integral = 0
        for i in range(n_bins):
            integral += dy * np.dot(
                NC11_weights, integrand[i*(11-1): (i+1)*11 - i]
            )

        return A0/2 * integral

    else:

        def integrand(eng):
            return (
                phi(eng / phys.lya_eng) *
                (fac + f_gamma(eng)) *
                (fac + f_gamma(phys.lya_eng - eng))
            )

        return (A0 / 2) * quad(integrand, 0, phys.lya_eng)[0] / phys.lya_eng


def N_2s1s(E, f_gamma, x_2s, x_1s):
    """
    distortion per baryon per second from two-photon emission,
    including stimulated emission effects
    """
    res = np.zeros_like(E)

    if f_gamma is None:
        def f_gamma(E):
            return 0

    legal_2s1s_bins = (phys.lya_eng >= E)
    engs = E[legal_2s1s_bins]

    # Interpolate integrand, since we need to know behavior between 0 and Lya
    phase_facs = (
        x_2s * (1 + f_gamma(engs)) * (1 + f_gamma(phys.lya_eng - engs)) -
        x_1s * f_gamma(engs) * f_gamma(phys.lya_eng - engs)
    )

    bin_bounds = spectools.get_bin_bound(engs)
    dy = (bin_bounds[1:]-bin_bounds[:-1]) / phys.lya_eng
    norm = phys.nH/phys.nB
    res[legal_2s1s_bins] = norm * A0 * phase_facs * phi(engs / phys.lya_eng)*dy

    return res


###############################
# The other bound-bound rates #
###############################


def Hey_A(n, l):
    """
    A_{nl} coefficient defined in Hey (2006), Eq. (9)
    """
    return np.sqrt(n**2-l**2)/(n*l)


def Hey_R_initial(n, n_p):
    """
    Absolute value of R_{n', n'- 1}^{n n'} as in Hey (2006), Eq. (B.4).
    Assumes n > n_p.
    """

    # Evaluates to
    # (-1)**(n-n_p-1) * 2**(2*n_p+2) * np.sqrt(
    # fact(n+n_p)/fact(n-n_p-1)/fact(2*n_p-1)) * (
    # n_p/n)**(n_p+2) * (1-n_p/n)**(n-n_p-2)/(1+n_p/n)**(n+n_p+2)
    return np.exp(
        (2.0*n_p + 2.0) * np.log(2.0) +
        0.5 * (lgamma(n+n_p+1) - lgamma(n-n_p) - lgamma(2.0*n_p))
        + (n_p + 2.0) * np.log(n_p/n)
        + (n-n_p - 2.0) * np.log(1.0 - n_p/n)
        - (n + n_p + 2.0) * np.log(1.0 + n_p/n)
    )


def populate_radial(nmax):
    """
    Populates a matrix with the radial matrix elements.

    Returns
    -------
    dict
        R['up'][n][n'][l] = R([n,l],[n',l+1])
        R['dn'][n][n'][l] = R([n,l],[n',l-1])

    Notes
    -----
    See Hey (2006), Eq. (B.4), (52), and (53)
    """
    R_up = np.zeros((nmax+1, nmax+1, nmax+1))
    R_dn = np.zeros((nmax+1, nmax+1, nmax+1))

    for n in np.arange(2, nmax+1, 1):
        for n_p in np.arange(1, n):
            # Initial conditions: Eq. (B.4)
            R_dn[n][n_p][n_p] = Hey_R_initial(n, n_p)
            R_up[n][n_p][n_p-1] = R_up[n][n_p][n_p] = 0
            for l in np.arange(n_p-1, 0, -1):

                R_dn[n][n_p][l] = (
                    (2*l+1) * Hey_A(n, l+1) * R_dn[n][n_p][l+1]
                    + Hey_A(n_p, l+1) * R_up[n][n_p][l]
                    ) / (2.0 * l * Hey_A(n_p, l))  # Hey Eq.(52)

                R_up[n][n_p][l-1] = (
                    (2*l+1) * Hey_A(n_p, l + 1) * R_up[n][n_p][l]
                    + Hey_A(n, l+1) * R_dn[n][n_p][l+1]
                    ) / (2.0 * l * Hey_A(n, l))  # Hey Eq.(53)

    return {'up': R_up, 'dn': R_dn}


def populate_bound_bound(nmax, Tr, R, Delta_f=None, simple_2s1s=False):
    """
    Populates two matrices with the bound-bound rates for emission and
    absorption in a generic radiation field.

    Parameters
    ----------
    nmax : int
        Maximum energy level of hydrogen atom to populate
    Tr : float
        Blackbody radiation temperature
    R : dict of 3-index arrays
        precomputed radial matrix elements (see populate_radial)
    Delta_f : function
        Deviation from blackbody phase space density as a function of energy,
        e.g. f_BB(E) = 1/(e^E/T - 1), so f(E) = f_BB(E) + Delta_f(E)
    simple_2s1s : bool, optional
        if *True*, fixes the decay rate to :math:`8.22` s:math:`^{-1}`. Default is *False*.

    Returns
    -------
    dict
        BB['up'][n][n'][l] = A([n,l]-> [n',l+1]) * (1 + f(E_{nn'}))  if n>n'
            = (2l+3)/(2l+1) exp(-E_{nn'}/Tr) * BB['dn'][n'][n][l+1]  if n<n'
        BB['dn'][n][n'][l] = A([n,l]-> [n',l-1]) * (1 + f(E_{nn'}))  if n>n'
            = (2l-1)/(2l+1) exp(-E_{nn'}/Tr) * BB['up'][n'][n][l-1]  if n<n'

    Notes
    -----
    The sobolev optical depth suppression factor has not been included yet
    """
    # 'up' and 'dn' here refers to the change in l. 
    keys = ['up', 'dn']

    BB = {key: np.zeros((nmax+1, nmax+1, nmax)) for key in keys}
    BB_2s1s = {key : 0 for key in keys}
    if Delta_f is None:
        def Delta_f(E):
            return 0

    def f_gamma(E):
        return f_BB(E, Tr) + Delta_f(E)

    # Vector of normalized energy levels. eng_levels[0] = np.nan, eng_levels[n] = 1 / n**2
    eng_norm_levels = np.divide(
        1., np.arange(nmax+1.)**2, out=np.ones(nmax+1)*np.nan, 
        where=np.arange(nmax+1) != 0
    )
    # n x np matrix of energy level differences, Ennp_mat[n, np] = (1 / np**2 - 1 / n**2) * phys.rydberg
    Ennp_mat = (eng_norm_levels[None,:] - eng_norm_levels[:,None]) * phys.rydberg
    
    # Masks for emission and absorption, and non-positive entries of Ennp_mat.
    # Nonzero for n_p > n, emission. 
    emission_mask = np.ones_like(Ennp_mat)
    emission_mask[Ennp_mat < 0] *= 0.
    # Nonzero for n > n_p, absorption. 
    absorption_mask = np.ones_like(Ennp_mat)
    absorption_mask[Ennp_mat > 0] *= 0.
    # Nonzero for n_p > n and n, n_p > 0. 
    non_pos_mask = np.zeros_like(Ennp_mat) 
    non_pos_mask[(Ennp_mat <= 0) | (np.isnan(Ennp_mat))] = 1. 

    # n x np matrix, occupation number of photons corresponding to positive energy level differences. 
    # Make a mask which sets the energy to a finite value if the energy <= 0 or is nan, 
    # so that f_gamma won't complain. We'll mask these values after anyway. 
    
    Ennp_mat[non_pos_mask > 0] = 1.
    # f_gamma_mat is nonzero for n_p > n, and n, n_p > 0. 
    f_gamma_mat = f_gamma(Ennp_mat) 
    f_gamma_mat[non_pos_mask > 0] = 0. 

    # n x np matrix. 
    prefac = 2*np.pi/3 * phys.rydberg / hplanck * (
        phys.alpha * (eng_norm_levels[None,:] - eng_norm_levels[:,None])
    )**3

    # zero out negative values. 
    prefac[non_pos_mask > 0] = 0. 

    n_ary = np.arange(nmax+1)

    # Emission

    # A_up = prefac * (l+1) / (2*l+1) * R['up'][n][n_p][l]**2
    # BB['up'][n][n_p][l] = A_up * (1+fEnnp)
    BB_emission_up = np.einsum(
        'ij,k,ijk,ij->ijk', prefac, (n_ary + 1.) / (2.*n_ary + 1.), R['up']**2, 1. + f_gamma_mat
    )

    # A_dn = prefac * l / (2*l+1) * R['dn'][n][n_p][l]**2
    # BB['dn'][n][n_p][l] = A_dn * (1+fEnnp)
    BB_emission_dn = np.einsum(
        'ij,k,ijk,ij->ijk', prefac, n_ary / (2.*n_ary + 1.), R['dn']**2, 1. + f_gamma_mat
    )


    # Absorption
    BB_absorption_up = np.zeros_like(BB_emission_up)
    BB_absorption_dn = np.zeros_like(BB_emission_dn) 

    # BB['up'][n_p][n][l] =   ((2*l+3)/(2*l+1) * BB['dn'][n][n_p][l+1]/(1+fEnnp) * fEnnp)
    BB_absorption_up[:,:,:-1] += np.transpose(BB_emission_dn, axes=(1, 0, 2))[:,:,1:]
    BB_absorption_up = np.einsum(
        'ijk,k,ji->ijk', BB_absorption_up, 
        (2. * n_ary + 3.) / (2. * n_ary + 1.), f_gamma_mat / (1. + f_gamma_mat)
    )

    # BB['dn'][n_p][n][l+1] = ((2*l+1)/(2*l+3) * BB['up'][n][n_p][l] / (1+fEnnp) * fEnnp)
    BB_absorption_dn[:,:,1:] += np.transpose(BB_emission_up, axes=(1, 0, 2))[:,:,:-1]
    BB_absorption_dn = np.einsum(
        'ijk,k,ji->ijk', BB_absorption_dn, 
        (2. * (n_ary - 1.) + 1.) / (2. * (n_ary - 1.) + 3), f_gamma_mat / (1. + f_gamma_mat)
    )
    # l = 0 is bogus, so zero it out. 
    BB_absorption_dn[:,:,0] = 0. 

    BB['up'] = BB_emission_up + BB_absorption_up 
    BB['dn'] = BB_emission_dn + BB_absorption_dn 



    # !!! parallelize these loops
    # for n in np.arange(2, nmax+1):
    #     n2 = n**2
    #     for n_p in np.arange(1, n):
    #         n_p2 = n_p**2
    #         Ennp = (1/n_p2 - 1/n2) * phys.rydberg
    #         fEnnp = f_gamma(Ennp)

    #         prefac = 2*np.pi/3 * phys.rydberg / hplanck * (
    #             phys.alpha * (1/n_p2 - 1/n2))**3

    #         ### Spont + stim emission ###
    #         l = np.arange(n_p+1)
    #         A_up = prefac * (l+1) / (2*l+1) * R['up'][n][n_p][l]**2
    #         A_dn = prefac * l / (2*l+1) * R['dn'][n][n_p][l]**2
    #         BB['up'][n][n_p][l] = A_up * (1+fEnnp)
    #         BB['dn'][n][n_p][l] = A_dn * (1+fEnnp)

    #         BB['up'][n][n_p][n_p] = BB['up'][n][n_p][n_p-1] = 0.0   # No l'>=n'
    #         BB['dn'][n][n_p][0] = 0.0                               # No l' < 0
    #         # if n == nmax: 
    #         #     print(BB['up'][n_p][n][l])
    #         #     print(BB_emission_up[n_p][n][l])
    #         #     if n_p == n-1: 
    #         #         raise ValueError('exit!')

    #         ### Absorption ###
    #         # When adding distortion, detailed balance takes thought.
    #         # To do it, take away the 1+fEnnp from a couple of lines
    #         # above, then replace it with fEnnp (that's all detailed
    #         # balance was doing).
    #         l = np.arange(n_p)  # absorption: use detailed balance
    #         BB['up'][n_p][n][l] = (
    #             (2*l+3)/(2*l+1) *
    #             BB['dn'][n][n_p][l+1]/(1+fEnnp) * fEnnp)
    #         BB['dn'][n_p][n][l+1] = (
    #             (2*l+1)/(2*l+3) *
    #             BB['up'][n][n_p][l] / (1+fEnnp) * fEnnp)


    if not simple_2s1s: 

        for key in ['up', 'dn']:    
            BB_2s1s[key] = A_2s1s(f_gamma, key)

    else: 
        BB_2s1s['dn'] = phys.width_2s1s_H 
        BB_2s1s['up'] = phys.width_2s1s_H * np.exp(-phys.lya_eng / Tr) 


    return BB, BB_2s1s


def tau_np_1s(n, rs, xHI=None):
    """
    Sobolev optical depth of np-1s line photons

    Notes
    -----
    see astro-ph/9912182 Eq. 40
    """
    l = 1
    nu = (1 - 1/n**2) * phys.rydberg/hplanck
    lam = phys.c/nu
    if xHI is None:
        xHI = phys.x_std(rs, 'HI')

    nHI = xHI * phys.nH*rs**3
    pre = lam**3 * nHI / (8*np.pi*phys.hubble(rs))

    A_prefac = 2*np.pi/3 * phys.rydberg / hplanck * (
        phys.alpha * (1 - 1/n**2))**3

    R = Hey_R_initial(n, 1)  # R['dn'][n][1][l]
    A_dn = A_prefac * l/(2*l+1) * R**2
    g = (2*l+1)/(2*l-1)
    return pre * A_dn * g


def p_np_1s(n, rs, xHI=None):
    """
    Escape probability of np-1s line photon

    Notes
    -----
    See astro-ph/9912182 Eq. 41

    Notice that p ~ 1/tau so

    R*p = A*(1+f)/tau ~ 1/(pre*g)
        = 8 pi H / (3 n_1s lam^3)

    where pre and g are defined above in tau_np_1s
    """
    tau = tau_np_1s(n, rs, xHI=xHI)
    return (1-np.exp(-tau))/tau


# BOUND_FREE FUNCTIONS


def populate_gnlk(nmax, n, kappa):
    """
    Populates two matrices with the coefficients g(n, l; kappa, l+1)
    and g(n, l; kappa, l-1).

    Parameters
    ----------
    !!! incomplete

    Returns
    -------
    Two matrices
        g_up[l] = g(n, l; kappa, l+1)
        g_dn[l] = g(n, l; kappa, l-1)

    Notes
    -----
    Reference: Burgess A.,1965, MmRAS..69....1B, Eqs. (28)-(34).
    """
    gnk_up = np.zeros((nmax, len(kappa)))
    gnk_dn = np.zeros((nmax, len(kappa)))

    k2 = kappa**2
    n2 = n**2

    log_product = 0.

    for s in range(1, n+1):
        log_product += np.log(1.0 + s**2 * k2)

    log_init = (0.5 * (np.log(np.pi/2) - lgamma(2.0 * n)) + np.log(4.0)
                + n * np.log(4.0 * n) + 0.5 * log_product
                - 0.5 * np.log(1.0 - np.exp(-2.0 * np.pi / kappa))
                - 2.0 * np.arctan(n * kappa) / kappa
                - (n + 2.0) * np.log(1.0 + n2 * k2))

    gnk_up[n-1] = np.exp(log_init)
    gnk_dn[n-1] = 0.5 * np.sqrt((1.0 + n2 * k2) / (
        1.0 + (n - 1.0) * (n - 1.0) * k2)) / n * gnk_up[n-1]

    if n > 1:
        gnk_up[n-2] = 0.5 * np.sqrt((2*n - 1) * (1 + n2 * k2)) * gnk_up[n-1]
        gnk_dn[n-2] = 0.5 * (4 + (n - 1) * (1 + n2 * k2)) * np.sqrt(
            (2 * n - 1) / (1 + (n - 2) * (n - 2) * k2)) / n * gnk_dn[n-1]

        for l in range(n-1, 1, -1):
            l2 = l**2
            gnk_up[l-2] = 0.5 * (
                ((4 * (n2 - l2) + l * (2*l - 1) * (1 + n2 * k2)) * gnk_up[l-1]
                 - 2*n * np.sqrt((n2 - l2) * (1 + (l+1)**2 * k2)) * gnk_up[l])
                / np.sqrt((n2 - (l - 1.0) * (l - 1.0)) * (1.0 + l2 * k2)) / n)

        for l in range(n-2, 0, -1):
            l2 = l**2
            gnk_dn[l-1] = 0.5 * (
                ((4 * (n2 - l2) + l * (2*l + 1) * (1 + n2 * k2)) * gnk_dn[l]
                 - 2*n * np.sqrt((n2 - (l+1)**2) * (1 + l2*k2)) * gnk_dn[l+1])
                / np.sqrt((n2 - l2) * (1.0 + (l-1)**2 * k2)) / n)

    return np.transpose(gnk_up), np.transpose(gnk_dn)


def populate_k2_and_g(nmax, Tm):
    """
    k2[n][ik] because boundaries depend on n
    """
    k2_tab = np.zeros((nmax+1, 10 * (NBINS-1) + 11))
    g = {key: np.zeros((nmax+1, Nkappa, nmax)) for key in ['up', 'dn']}
    k2max = 7e2*Tm/phys.rydberg

    for n in range(1, nmax+1):
        k2min = 1e-25/n**2
        bigBins = np.logspace(np.log10(k2min), np.log10(k2max), NBINS + 1)
        iBig = np.arange(NBINS)
        temp = np.linspace(bigBins[iBig], bigBins[iBig+1], 11)
        for i in range(11):
            k2_tab[n][10 * iBig + i] = temp[i]
        ik = np.arange(10 * NBINS + 1)
        g['up'][n, ik], g['dn'][n, ik] = populate_gnlk(
            nmax, n, np.sqrt(k2_tab[n, ik]))
    return k2_tab, g


def Newton_Cotes_11pt(x, f):
    """
    11 point Newton-Cotes integration.
    Parameters
    ----------
    an 11-point array x, an 11-point array f(x).

    Returns
    -------
    \int f(x) dx over the interval provided.
    """
    h = (x[10] - x[0])/10  # step size

    return (5 * h * (16067 * (f[0] + f[10]) + 106300 * (f[1] + f[9])
                     - 48525 * (f[2] + f[8]) + 272400 * (f[3] + f[7])
                     - 260550 * (f[4] + f[6]) + 427368 * f[5]) / 299376)


def populate_beta(Tr, nmax, Delta_f=None, Thetas=None):
    """ Populating the photoionization rates beta(n, l, Tr)
        From prefac we see that the units are s^-1

        Parameters
        ----------

        Tr : float
            radiation temperature in eV
        nmax : int
            maximum principle quantum number (energy level)
    """
    beta = np.zeros((nmax+1, nmax))

    if Delta_f is None:
        def Delta_f(E): return 0

    def f_gamma(Ennp):
        return np.exp(-Ennp / Tr)/(1.0 - np.exp(-Ennp / Tr)) + Delta_f(Ennp)

    for n in range(1, nmax+1):
        l = np.arange(n)
        beta[n][l] = bf.beta_n(n, Thetas, f_gamma=f_gamma)
    #    for l in range(n):
    #        beta[n][l] = bf.beta_nl(n, l, f_gamma=f_gamma)

    return beta


def populate_alpha(Tm, Tr, nmax, Delta_f=None, stimulated_emission=True, Thetas=None):
    """ Populate the recombination coefficients alpha(n, l, Tm, Tr)

        Parameters
        ----------

        Tm : float
            matter temperature [eV]
        Tr : float
            radiation temperature of background blackbody [eV]
        nmax : int
            maximum principle quantum number (energy level)
        Delta_f : function
            deviation of photon phase space density from blackbody
        stimulated_emission : bool
            if True, include stimulated emission factors, 1+f
    """
    alpha = np.zeros((int(nmax+1), nmax))

    if stimulated_emission:

        if Delta_f is None:
            def Delta_f(E): return 0

        def f_gamma(Ennp):
            return np.exp(-Ennp/Tr)/(1.0 - np.exp(-Ennp/Tr)) + Delta_f(Ennp)
    else:
        f_gamma = None

    for n in np.arange(1, nmax+1):
        l = np.arange(n)
        alpha[n][l] = bf.alpha_n(
            n, Tm, Thetas, f_gamma=f_gamma,
            stimulated_emission=stimulated_emission
        )
    # # Slower implementation
    #    for l in np.arange(n):
    #        alpha[n][l] = bf.alpha_nl(
    #            n, l, Tm, f_gamma=f_gamma,
    #            stimulated_emission=stimulated_emission
    #        )

    return alpha


def get_transition_energies(nmax):
    """
    Compute the exact energy bins for transitions between excited state of H.
    This includes an extra bin at 20 eV to represent bound-free transitions.

    Parameters
    ----------
    nmax : int
        Highest excited state to be included

    Returns
    -------
    H_engs : array
    """

    H_engs = np.zeros((nmax+1, nmax+1))
    for n1 in np.arange(1, nmax+1):
        #H_engs[0,n1] = phys.rydberg / n1**2
        for n2 in range(1, n1):
            H_engs[n1, n2] = phys.rydberg * ((1/n2)**2 - (1/n1)**2)

    H_engs = np.sort(np.unique(H_engs))
    # Add a separate energy bin to temporarily represent 2s->1s
    #H_engs = np.concatenate((H_engs, [20]))

    # Get rid of zero energy
    return H_engs[1:]



# def process_MLA_vectorized(
#     rs, dt, xHI, Tm, nmax, eng, R, Thetas,
#     Delta_f=None, cross_check=False,
#     include_BF=True, simple_2s1s=False,
#     # fexc_switch=False, deposited_exc_arr=None, elec_spec=None,
#     # distortion=None, H_states=None, rate_func_eng=None,
#     delta_b={}, stimulated_emission=True
# ):

#     if cross_check:
#         xHI = phys.x_std(rs, 'HI')
#         # Tm = phys.TCMB(rs)

#     # Number of Hydrogen states at or below n=nmax
#     num_states = int(nmax*(nmax+1)/2)

#     # Mapping from spectroscopic letters to numbers
#     # spectroscopic_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
#     l_to_spec_map = {0: 's', 1: 'p', 2: 'd', 3: 'f'}

#     def num_to_l(ll):
#         if ll < 4:
#             return l_to_spec_map[ll]

#         else:
#             return '-'

#     # Indices for the bound states
#     # e.g. (1s, 2s, 2p, 3s...) are states (0, 1, 2, 3...),
#     # so states_n[3], states_l[3] = 3,0 for '3s'
#     states_n = np.concatenate([
#         list(map(int, k*np.ones(k))) for k in range(1, nmax+1, 1)])
#     states_l = np.concatenate([np.arange(k) for k in range(1, nmax+1)])
#     # Index arrays for moving between n x n' x l x l' representation to i = (n,l), j = (n', l') representation. 
#     # Simply call A[states_n_2d, states_l_2d, states_n_p_2d, states_l_p_2d]. 
#     states_n_p_2d = np.tile(states_n, (num_states,1)) 
#     states_l_p_2d = np.tile(states_l, (num_states,1)) 
#     states_n_2d = np.transpose(states_n_p_2d)
#     states_l_2d = np.transpose(states_l_p_2d)


#     # Bound state energies
#     def E(n): return phys.rydberg/n**2

#     xe = 1-xHI
#     nH = phys.nH * rs**3
#     nB = phys.nB * rs**3

#     if Delta_f is None:
#         def Delta_f(E): return 0

#     # Radiation Temperature
#     Tr = phys.TCMB(rs)

#     # Get the transition rates
#     # !!! Think about parallelizing
#     #R = populate_radial(nmax)  # Need not be recomputed every time
#     BB, BB_2s1s = populate_bound_bound(nmax, Tr, R, Delta_f=Delta_f, simple_2s1s=simple_2s1s)
#     alpha = populate_alpha(
#         Tm, Tr, nmax, Delta_f=Delta_f, Thetas=Thetas, 
#         stimulated_emission=stimulated_emission
#     )
#     beta = populate_beta(Tr, nmax, Delta_f=Delta_f, Thetas=Thetas)

#     # Include Sobolev optical depth. 
#     BB['up'][1,:,0] *= p_np_1s(np.arange(2, nmax+1), rs, xHI=xHI)
#     BB['dn'][:,1,1] *= p_np_1s(np.arange(2, nmax+1), rs, xHI=xHI) 

#     # 4D versions of BB['up'] and BB['dn'], dimensions n x n' x l x l',  
#     # with entries in the (..., l, l+1) and (..., l, l-1) positions respectively.
#     BB_up_4d = np.zeros((nmax+1, nmax+1, nmax+1, nmax+1))
#     BB_dn_4d = np.zeros((nmax+1, nmax+1, nmax+1, nmax+1))
    
#     # BB['up'] has dimensions n x n' x l, with transitions to l' = l+1. Add BB['up'] with an extra axis, 
#     # then zero out all irrelevant entries, except l + 1. 
#     BB_up_4d += BB['up'][:,:,:,None]
#     BB_up_4d = np.tril(np.triu(BB_up_4d, k=1), k=1)
#     # Similar for BB['dn'], except l' = l-1. 
#     BB_dn_4d += BB['dn'][:,:,:,None]
#     BB_dn_4d = np.tril(np.triu(BB_dn_4d, k=-1), k=-1)
    
    
#     # Total rate out of the state, dimensions n x l. 
#     # First add all bound-bound and photoionization. 
#     tot_rate += np.sum(BB['up'] + BB['dn'], axis=1) + beta
#     # Add 1s -> 2s 
#     tot_rate[1, 0] += BB_2s1s['up']
#     # Add 2s -> 1s
#     tot_rate[2, 0] += BB_2s1s['dn']

#     # Initialize K_ij = R_ji / R_i,tot, dimensions n x n' x l x l', where i = (n,l), j = (n',l')
#     K = np.zeros((nmax+1, nmax+1, nmax+1, nmax+1))
    
#     # R_ji is the rate from n', l' --> n, l, which is the
#     # transpose of how BB is saved. 
#     K = (
#         np.transpose(BB_up_4d, axes=(1, 3, 0, 2)) + np.transpose(BB_dn_4d, axes=(1, 3, 0, 2)) 
#     ) / tot_rate[:,:,None,None]
#     # Add the special case 1s <-> 2s results. 
#     K[1, 0, 2, 0] += BB_2s1s['dn'] / tot_rate[1, 0]
#     K[2, 0, 1, 0] += BB_2s1s['up'] / tot_rate[2, 0]

#     #############################
#     #        Source Terms       #
#     #############################

#     # Excitation source term into n,l state, dimensions n x l. 
#     b_exc_2D = np.zeros_like(tot_rate)  
#     # 1s -> 2p excitations. 
#     b_exc_2D[:,1] += BB['up'][1, :, 0]
#     # 1s -> 2s excitations. 
#     b_exc_2D[2,0] += BB_2s1s['up'] 

#     # Recombination source term into n,l state, dimensions n x l. 
#     b_rec_2D = alpha

#     # DM source term into n,l state, dimensions n x l. 
#     # This is a short loop. 
#     for state in delta_b.keys()
#     spec_ind = str(n) + num_to_l(l) 
#     b_DM
    














def process_MLA(
        rs, dt, xHI, Tm, nmax, eng, R, Thetas,
        Delta_f=None, cross_check=False,
        include_BF=True, simple_2s1s=False,
        # fexc_switch=False, deposited_exc_arr=None, elec_spec=None,
        # distortion=None, H_states=None, rate_func_eng=None,
        delta_b={}, stimulated_emission=True, vectorized=False
        ):
    """
    Solve the steady state equation Mx=b, then compute the ionization rate
    beta^T x and the net distortion produced by transitions.

    Parameters
    ----------
    rs : float
        redshift
    xHI : float
        neutral fraction of hydrogen
    Tm : float
        matter temperature
    nmax : int
        Highest excited state to be included (principle quantum number)
    eng : array
        abscissa of output distortion spectrum
    Delta_f : function
        photon phase space density as a function of energy, minus f_BB
    cross_check : bool
        if True, set xHI to its standard value.
    include_BF : bool
        includes the bound-free transition photons to the output distortion
    simple_2s1s : bool, optional
        If *True*, sets the 2s -> 1s rate to be a constant :math:`8.22` s:math:`^{-1}`, and does not include distortions from 2s -> 1s.  
    fexc_switch : bool
        deposited_exc_arr elec_spec distortion H_states rate_func_eng
    vectorized : bool
        If *True*, uses the vectorized calculation. 

    Returns
    -------
    Pto1s_many, PtoCont_many : vectors of probabilities for transitioning to
        1s, continuum after many transitions
    transition_specs : dictionary of photon spectra, labeled by
        initial excited state (in N, not dNdE)
    """

    # if vectorized: 

    #     return process_MLA_vectorized(
    #         rs, dt, xHI, Tm, nmax, eng, R, Thetas,
    #         Delta_f=Delta_f, cross_check=cross_check,
    #         include_BF=include_BF, simple_2s1s=simple_2s1s,
    #         delta_b=delta_b, stimulated_emission=stimulated_emission
    #     )

    if cross_check:
        xHI = phys.x_std(rs, 'HI')
        # Tm = phys.TCMB(rs)

    # Number of Hydrogen states at or below n=nmax
    num_states = int(nmax*(nmax+1)/2)

    # Mapping from spectroscopic letters to numbers
    # spectroscopic_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
    spectroscopic_map = {0: 's', 1: 'p', 2: 'd', 3: 'f'}

    def num_to_l(ll):
        if ll < 4:
            return spectroscopic_map[ll]

        else:
            return '-'

    # Indices of the bound states
    # e.g. (1s, 2s, 2p, 3s...) are states (0, 1, 2, 3...),
    # so states_n[3], states_l[3] = 3,0 for '3s'
    states_n = np.concatenate([
        list(map(int, k*np.ones(k))) for k in range(1, nmax+1, 1)])
    states_l = np.concatenate([np.arange(k) for k in range(1, nmax+1)])

    # Bound state energies
    def E(n): return phys.rydberg/n**2

    xe = 1-xHI
    nH = phys.nH * rs**3
    nB = phys.nB * rs**3

    if Delta_f is None:
        def Delta_f(E): return 0

    # Radiation Temperature
    Tr = phys.TCMB(rs)

    # Get the transition rates
    # !!! Think about parallelizing
    #R = populate_radial(nmax)  # Need not be recomputed every time
    BB, BB_2s1s = populate_bound_bound(nmax, Tr, R, Delta_f=Delta_f, simple_2s1s=simple_2s1s)
    alpha = populate_alpha(Tm, Tr, nmax, Delta_f=Delta_f, Thetas=Thetas,
                           stimulated_emission=stimulated_emission)
    beta = populate_beta(Tr, nmax, Delta_f=Delta_f, Thetas=Thetas)

    # Include sobolev optical depth
    for n in np.arange(2, nmax+1, 1):
        BB['dn'][n][1][1] *= p_np_1s(n, rs, xHI=xHI)
        BB['up'][1][n][0] *= p_np_1s(n, rs, xHI=xHI)



    ### Build matrix K_ij = R_ji/R_i,tot and source term ###
    K = np.zeros((num_states, num_states))

    # source term
    b_exc = np.zeros(num_states)  # from CMB-photon + 1s -> nl
    b_rec = np.zeros(num_states)  # from e+p -> nl
    b_DM = np.zeros(num_states)   # from DM-product + 1s -> nl

    # Rate of transitions from excited states -> 1s
    tot_rate = np.zeros(num_states)

    for nl in np.arange(num_states):
        n, l = states_n[nl], states_l[nl]
        tot_rate[nl] = (
            np.sum(BB['dn'][n, :, l]) + np.sum(BB['up'][n, :, l]) + beta[n][l]
        )

        ###
        # Construct the matrix
        ###

        # Set 2s <-> 1s rates to their default values w/o photon backgrounds
        # BB_2s1s['dn'] = phys.width_2s1s_H
        # BB_2s1s['up'] = phys.width_2s1s_H * np.exp(-phys.lya_eng / Tr)
        if nl == 0:  # special case: 1s -> 2s
            tot_rate[nl] += BB_2s1s['up']
            if tot_rate[nl] > 0:
                K[0][1] = BB_2s1s['up'] / tot_rate[nl]

        if nl == 1:  # special case: 2s -> 1s
            tot_rate[nl] += BB_2s1s['dn']
            K[1][0] = BB_2s1s['dn'] / tot_rate[nl]

        if l != 0:
            K[nl, states_l == l-1] = BB['up'][l:, n, l-1]/tot_rate[nl]

        if l != nmax-1:
            if tot_rate[nl] > 0:
                K[nl, states_l == l+1] = BB['dn'][l+2:, n, l+1]/tot_rate[nl]

        ###
        # Construct the source terms
        ###

        # excitations from direct recombinations
        b_rec[nl] += alpha[n][l]

        if l == 1:  # excitations from 1s->np transitions
            b_exc[nl] += BB['up'][1, n, 0]


        elif nl == 1:  # excitations from 1s->2s
            b_exc[nl] += BB_2s1s['up']

        # Add DM contribution to source term
        # i.e. f_exc -> distortion and ionization
        spec_ind = str(n) + num_to_l(l)
        b_DM[nl] = delta_b.get(spec_ind, 0)  # if key not in delta_b, return 0

        if tot_rate[nl] > 0:
            b_exc[nl] /= tot_rate[nl]
            b_rec[nl] /= tot_rate[nl]
            b_DM[nl] /= tot_rate[nl]


    # sparse matrix
    mat = sp.csr_matrix(np.identity(num_states-1) - K[1:, 1:])
    # b_tot = b_exc * xHI + b_rec * xe**2 * nH + b_DM
    # x_vec = np.linalg.solve(mat, b_tot[1:])


    # components of x_vec
    dx_exc = sp.linalg.spsolve(mat, b_exc[1:])  # np.linalg.solve if dense mat
    dx_rec = sp.linalg.spsolve(mat, b_rec[1:])
    dx_DM = sp.linalg.spsolve(mat, b_DM[1:])

    # print(x_vec/(dx_exc*xHI+dx_rec*xe**2*nH+dx_DM)-1)
    x_vec = dx_exc*xHI + dx_rec*xe**2*nH + dx_DM

    # naively you'd want x_full[0] = 1 - sum(x_full) - xe, but
    # we already assumed xe = 1 - xHI above, so we'd run into
    # detailed balance issues if we didn't set x_full[0] = xHI
    x_full = np.append(xHI, x_vec)

    ###
    # Now calculate the total ionization and distortion
    ###

    E_current = 0
    ind_current = 0
    H_engs = np.zeros(num_states)

    Nphot_cascade = np.zeros(num_states)
    BF_spec = Spectrum(eng, np.zeros_like(eng), spec_type='dNdE')
    alpha_MLA, beta_MLA, beta_DM = 0, 0, 0

    def f_gamma(E):
        return f_BB(E, Tr) + Delta_f(E)

    # !!! Parallelize this loop
    for nl in np.arange(num_states):
        n, l = states_n[nl], states_l[nl]
        if nl > 0:
            # beta_MLA += x_full[nl] * beta[n][l]
            beta_MLA += beta[n][l] * dx_exc[nl-1]
            alpha_MLA += alpha[n][l] - beta[n][l] * dx_rec[nl-1]
            beta_DM += beta[n][l] * dx_DM[nl-1]

        # Add new transition energies to H_engs
        if E_current != E(n):
            if n > 1:
                ind_current += nmax-n+1

            E_current = E(n)
            H_engs[ind_current:ind_current + nmax-n] = (
                E(n)-E(np.arange(n+1, nmax+1)))

        # photons from l <-> l+1 transitions (per baryon per second)
        if l < nmax-1:
            Nphot_cascade[ind_current:ind_current + nmax-n] += nH*(
                # Downscattering adds photons
                x_full[(states_l == l+1) * (
                    states_n > n)] * BB['dn'][n+1:, n, l+1]

                # Upscattering adds them
                - x_full[nl] * BB['up'][n, n+1:, l]
            )/nB * dt
        # note: 'dn' and 'up' have nothing to do with down- or up-scattering,
        # just if the l quantum number go up or down

        # photons from l <-> l-1 transitions
        if l > 0:
            Nphot_cascade[ind_current:ind_current + nmax-n] += nH * (
                x_full[(states_l == l-1) * (
                    states_n > n)] * BB['up'][n+1:, n, l-1]
                - x_full[nl] * BB['dn'][n, n+1:, l]
            )/nB * dt

        if not stimulated_emission:
            f_gam = None
        else:
            f_gam = f_gamma

        if l == 0:  # once for each n
            BF_contribution = bf.net_spec_n(
                n, Tm, xe, x_full, nH, Thetas, Tr, f_gamma=f_gam,
                stimulated_emission=stimulated_emission
            )/nB * dt
            BF_contribution.rebin(eng)
            BF_spec.dNdE += BF_contribution.dNdE

        ## This is where f_ion -> distortion
        #BF_tmp = nH**2 * xe**2 * bf.gamma_nl(
        #    n, l, Tm, T_r=Tr, f_gamma=f_gam,
        #    stimulated_emission=stimulated_emission
        #)/nB * dt
        #BF_tmp -= nH * x_full[nl] * bf.xi_nl(
        #    n, l, T_r=Tr, f_gamma=f_gam)/nB * dt
        #BF_tmp.rebin(eng)
        #BF_spec.dNdE += BF_tmp.dNdE

    # Make a spectrum
    data = sorted(np.flipud(np.transpose([H_engs, Nphot_cascade])),
                  key=lambda pair: pair[0])

    # Consolidate duplicates (e.g. 6 <-> 9 transition
    # is same energy as 8 <-> 72)
    i = 0
    sz = num_states
    while i < sz-1:
        while (i < sz-1) and (data[i][0] == data[i+1][0]):
            data[i][1] += data[i+1][1]
            data.pop(i+1)
            sz -= 1

        i += 1

    data = np.array(data)

    transition_spec = Spectrum(data[:, 0], data[:, 1], spec_type='N', rs=rs)
    transition_spec.rebin(eng)

    # Add the bound-free photons
    if include_BF:
        transition_spec.N += BF_spec.N

    # Add the 2s-1s component
    if not simple_2s1s:
        #spec_2s1s = discretize(dist_eng, phys.dNdE_2s1s)
        #amp_2s1s = nH * phys.width_2s1s_H * (
        #    x_full[1] - x_full[0]*np.exp(-phys.lya_eng/Tr)
        #) / nB * dt
        #transition_spec.N += amp_2s1s * spec_2s1s.N
        transition_spec.N += N_2s1s(eng, f_gam, x_full[1], x_full[0]) * dt

    return [alpha_MLA, beta_MLA, beta_DM], transition_spec


def absorb_photons(distortion, H_states, dt, x1s, nmax):
    """ Allow ground state atoms to absorb distortion photons

        Identify the bins that contain resonant photons, calculate what
        fraction of them get absorbed, subtract them from distortion.N, and
        return f_{exc,i} to account for the energy that got absorbed.
    """
    rs = distortion.rs
    if rs == -1:
        raise TypeError('must initialize the redshift\'s distortion properly')

    photeng = distortion.eng

    # Photon phase space density
    prefac = phys.nB * (phys.hbar*phys.c*rs)**3 * np.pi**2
    Delta_f = interp1d(photeng, prefac * distortion.dNdE/photeng**2)

    # np -> 1s decay rate
    R_1snp = Hey_R_initial(np.arange(2, nmax+1), 1)

    A_1snp = 1/3 * phys.rydberg / phys.hbar * (
        phys.alpha * (1-1/np.arange(2, nmax+1)**2)
    )**3 * R_1snp**2  # need not be recomputed every time

    # energy bin boundaries
    bnds = spectools.get_bin_bound(photeng)

    # amount of energy that got absorbed
    dE_absorbed = {state: 0 for state in H_states}

    # Container object for photons that get absorbed
    spec_absorbed = distortion.copy() * 0

    # energy bins that contain each Lyman series line
    line_bins = np.array([
        [int(state[:-1]), sum(bnds < phys.H_exc_eng(state))-1]
        for state in H_states if state[-1] == 'p'])

    for state in H_states:
        if state[-1] == 'p':

            n = int(state[:-1])

            # all excitation lines that share the same bin
            ns = line_bins[line_bins[:, 1] == line_bins[n-2, 1], 0]

            # Energy of absorbing photons
            E_1np = (1 - 1/ns**2) * phys.rydberg

            # Bin containing this excitation line
            line_bin = sum(bnds < phys.H_exc_eng(state))-1

            # Calculate the fraction of photons that get absorbed
            # Note: the CMB component is untouched (in equilibrium emission and
            #   absorption balances out), we only modify the distortion.

            # !!! Technically, I should subtract off inverse process
            #   (stimulated emission) but there are so few x_np
            #   states that I neglect this process.
            # I !!! include the heaviside step function to only allow
            #   positive distortions to be absorbed (those correspond
            #   to actual photons getting absorbed)
            absorption_rates = A_1snp[ns-2] * x1s * np.heaviside(
                Delta_f(E_1np), 0)
            escape_frac = np.exp(-sum(absorption_rates)*dt)
            absorbed_frac = 1-escape_frac

            # If there's more than one line in this energy bin, compute the
            # fraction absorbed by each line
            line_frac = A_1snp[n-2]/sum(A_1snp[ns-2])

            # Photon energy absorbed *per baryon*
            dE_absorbed[state] = (
                absorbed_frac * line_frac
                * distortion.N[line_bin] * distortion.eng[line_bin])

            spec_absorbed.N[line_bin] += (
                distortion.N[line_bin] * absorbed_frac * line_frac)

    distortion.N -= spec_absorbed.N
    return dE_absorbed


def f_exc_to_b_numerator(deposited_exc_arr, elec_spec, distortion,
                         H_states, dt, rate_func_eng, nmax, x1s):

    rs = elec_spec.rs
    nB = phys.nB * rs**3
    nH = phys.nH * rs**3

    # Convert from energy per baryon to
    # (dimensionless) fraction of injected energy
    norm = nB / rate_func_eng(rs) / dt

    # fraction of energy deposited into excitation into the i-th state
    f_exc = {state: 0 for state in H_states}

    # Allow H(1s) to absorb line photons from distortion (E per baryon)
    dE_absorbed = absorb_photons(distortion, H_states, dt, x1s, nmax)

    for state in H_states:

        # electron contribution to f_exc for the i-th state
        f_exc[state] = np.dot(deposited_exc_arr[state], elec_spec.N) * norm

        # photon contribution
        f_exc[state] += dE_absorbed[state] * norm

    # f_exc contribution to source term in the MLA equation
    delta_b = {state: 0 for state in H_states}

    # Calculate the electron contribution to f_exc for the i-th state
    for state in H_states:
        delta_b[state] = (
            f_exc[state] * rate_func_eng(rs)/nH/phys.H_exc_eng(state)
        )

    return delta_b


def x2s_steady_state(rs, Tr, Tm, xe, x1s, tau_S, fudge=1.125):

    # Boltzmann Factor at lya energy
    B_Lya = np.exp(-phys.lya_eng/Tr)

    # Photon occupation number
    f_Lya = B_Lya/(1-B_Lya)

    # 2p-1s rate, including Sobolev optical depth
    R_Lya = 2*(2/3)**8 * phys.alpha**3 * phys.rydberg/phys.hbar * (1+f_Lya)
    R_Lya *= (1-np.exp(-tau_S))/tau_S

    # Total deexcitation rate including 2s->1s rate
    sum_rates = (3*R_Lya + phys.width_2s1s_H)/4

    # Denominator of Peebles C factor
    denom = sum_rates + phys.beta_ion(Tm, 'HI', fudge)

    # Two numerator terms for x2 steady state solution
    nH = phys.nH * rs**3
    term1 = xe**2 * nH * phys.alpha_recomb(Tm, 'HI', fudge)
    term2 = 4 * x1s * np.exp(-phys.lya_eng/Tr) * sum_rates
    # print(term1, term2)

    # Factor of 4 converts from x2 to x2s
    ans = (term1 + term2)/denom / 4.
    if type(ans) is np.ndarray:
        if len(ans) == 1:
            return ans[0]
        else:
            return ans
    else:
        return ans
