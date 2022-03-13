"""H excitation state transitions, ionization, and recombination functions."""

import numpy as np
from scipy.interpolate import interp1d

from darkhistory import physics as phys
from darkhistory.spec.spectrum import Spectrum
import darkhistory.spec.spectools as spectools
import scipy.special

import darkhistory.low_energy.bound_free as bf

NBINS = 100
Nkappa = 10 * NBINS + 1

hplanck = phys.hbar*2*np.pi
mu_e = phys.me/(1+phys.me/phys.mp)
lgamma = scipy.special.loggamma

#*************************************************#
#A_{nl} coefficient defined in Hey (2006), Eq. (9)#
#*************************************************#

def Hey_A(n, l):
    return np.sqrt(n**2-l**2)/(n*l)

#/*****************************************************************#
#Absolute value of R_{n', n'- 1}^{n n'} as in Hey (2006), Eq. (B.4)
#Assumes n > np.
#******************************************************************#


def Hey_R_initial(n, n_p):
    # return (-1)**(n-n_p-1) * 2**(2*n_p+2) * np.sqrt(
    # fact(n+n_p)/fact(n-n_p-1)/fact(2*n_p-1)) * (
    # n_p/n)**(n_p+2) * (1-n_p/n)**(n-n_p-2)/(1+n_p/n)**(n+n_p+2)
    return np.exp(
        (2.0*n_p + 2.0) * np.log(2.0) +
        0.5 * (lgamma(n+n_p+1) - lgamma(n-n_p) - lgamma(2.0*n_p))
        + (n_p + 2.0) * np.log(n_p/n)
        + (n-n_p - 2.0) * np.log(1.0 - n_p/n)
        - (n + n_p + 2.0) * np.log(1.0 + n_p/n)
    )

#/*********************************************************************
#Populates a matrix with the radial matrix elements.
#Inputs: two [nmax+1][nmax+1][nmax] matrices Rup, Rdn, nmax.
#Rup is populated s.t. R_up[n][n'][l] = R([n,l],[n',l+1])
#Rdn is populated s.t. R_dn[n][n'][l] = R([n,l],[n',l-1])
#Assumption: n > n'
#**********************************************************************/

def populate_radial(nmax):
    R_up = np.zeros((nmax+1, nmax, nmax))
    R_dn = np.zeros((nmax+1, nmax, nmax))

    for n in np.arange(2, nmax+1, 1):
        for n_p in np.arange(1, n):
            # Initial conditions: Hey (2006) Eq. (B.4)
            R_dn[n][n_p][n_p] = Hey_R_initial(n, n_p)
            R_up[n][n_p][n_p-1] = R_up[n][n_p][n_p] = 0
            for l in np.arange(n_p-1, 0, -1):
                # Hey (52-53)
                R_dn[n][n_p][l] = (
                    (2*l+1) * Hey_A(n, l+1) * R_dn[n][n_p][l+1]
                    + Hey_A(n_p, l+1) * R_up[n][n_p][l]
                    ) / (2.0 * l * Hey_A(n_p, l))

                R_up[n][n_p][l-1] = (
                    (2*l+1) * Hey_A(n_p, l + 1) * R_up[n][n_p][l]
                    + Hey_A(n, l+1) * R_dn[n][n_p][l+1]
                    ) / (2.0 * l * Hey_A(n, l))

    return {'up': R_up, 'dn': R_dn}

# /***************************************************************************************************************
# Populates two matrices with the bound-bound rates for emission and absorption
#  in a generic radiation field.
# Inputs : two [nmax+1][nmax+1][nmax] matrices BB_up, BB_dn, nmax, Tr (in eV),
#  and the two precomputed matrices of radial matrix elements R_up and R_dn.
# BB_up[n][n'][l] = A([n,l]-> [n',l+1]) * (1 + f(E_{nn'}))              if n>n'
#                 = (2l+3)/(2l+1) exp(-E_{nn'}/Tr) * BB_dn[n'][n][l+1]  if n<n'
# BB_dn[n][n'][l] = A([n,l]-> [n',l-1]) * (1 + f(E_{nn'}))              if n>n'
#                 = (2l-1)/(2l+1) exp(-E_{nn'}/Tr) * BB_up[n'][n][l-1]  if n<n'
#  *****************************************************************************************************************/


def populate_bound_bound(nmax, Tr, R, ZEROTEMP=1e-10, Delta_f=None):
    BB = {key: np.zeros((nmax+1, nmax+1, nmax)) for key in ['up', 'dn']}

    for n in np.arange(2, nmax+1, 1):
        n2 = n**2
        for n_p in np.arange(1, n, 1):
            n_p2 = n_p**2
            Ennp = (1/n_p2 - 1/n2) * phys.rydberg
            if (Tr < ZEROTEMP):    # if Tr = 0
                fEnnp = 0.0
            else:
                if Delta_f is not None:
                    fEnnp = np.exp(-Ennp/Tr)/(
                        1-np.exp(-Ennp/Tr)) + Delta_f(Ennp)
                else:
                    fEnnp = np.exp(-Ennp/Tr)/(1-np.exp(-Ennp/Tr))
                    # fEnnp = 1/(np.exp(Ennp/Tr)-1)

            prefac = 2*np.pi/3 * phys.rydberg / hplanck * (
                phys.alpha * (1/n_p2 - 1/n2))**3

            for l in np.arange(0, n_p+1, 1):  # Spont + stim emission
                A_up = prefac * (l+1) / (2*l+1) * R['up'][n][n_p][l]**2
                A_dn = prefac * l / (2*l+1) * R['dn'][n][n_p][l]**2
                BB['up'][n][n_p][l] = A_up * (1+fEnnp)
                BB['dn'][n][n_p][l] = A_dn * (1+fEnnp)

            BB['up'][n][n_p][n_p] = BB['up'][n][n_p][n_p-1] = 0.0   # No l'>=n'
            BB['dn'][n][n_p][0] = 0.0                          # No l' < 0
            for l in np.arange(0, n_p, 1):  # absorption: use detailed balance
                if (Tr < ZEROTEMP):  # if Tr = 0
                    BB['up'][n_p][n][l] = 0.0   
                    BB['dn'][n_p][n][l+1] = 0.0
                else:
                    # BB['up'][n_p][n][l]   = (2*l+3)/(2*l+1) * np.exp(
                    # -Ennp/Tr) * BB['dn'][n][n_p][l+1]
                    # BB['dn'][n_p][n][l+1] = (2*l+1)/(2*l+3) * np.exp(
                    # -Ennp/Tr) * BB['up'][n][n_p][l]

                    # When adding a distortion, detailed balance is inconvenient.
                    # Instead, take away the 1+fEnnp from a couple of lines above,
                    # then replace it with fEnnp (that's all detailed balance was doing).
                    BB['up'][n_p][n][l] = (2*l+3)/(2*l+1) * BB['dn'][n][n_p][l+1]/(1+fEnnp) * fEnnp
                    BB['dn'][n_p][n][l+1] = (2*l+1)/(2*l+3) * BB['up'][n][n_p][l]  /(1+fEnnp) * fEnnp

    #Include forbidden 2s->1s transition
    BB['dn'][2][1][0] = phys.width_2s1s_H
    #!!! What about the inverse process?

    return BB


# astro-ph/9912182 Eq. 40
def tau_np_1s(n, rs, xHI=None):
    l = 1
    nu = (1 - 1/n**2) * phys.rydberg/hplanck
    lam = phys.c/nu
    if xHI is None:
        # xHI = 1-phys.xHII_std(rs)
        xHI = phys.xHI_std(rs)
    nHI = xHI * phys.nH*rs**3
    pre = lam**3 * nHI / (8*np.pi*phys.hubble(rs))

    A_prefac = 2*np.pi/3 * phys.rydberg / hplanck * (
        phys.alpha * (1 - 1/n**2))**3

    R = Hey_R_initial(n, 1)  # R['dn'][n][1][l]
    A_dn = A_prefac * l/(2*l+1) * R**2
    g = (2*l+1)/(2*l-1)
    return pre * A_dn * g

# Eq. 41
def p_np_1s(n, rs, xHI=None):
    tau = tau_np_1s(n, rs, xHI=xHI)
    return (1-np.exp(-tau))/tau

# Notice that p ~ 1/tau so
# R*p = A*(1+f)/tau ~ 1/(pre*g)
#     = 8 pi H / (3 n_1s lam^3)

#~~~ BOUND_FREE FUNCTIONS ~~~

# /*********************************************************************************************************************
# Populates two matrices with the coefficients g(n, l; kappa, l+1) and g(n, l; kappa, l-1).
# Input: two [nmax] matrices g_up and g_dn, n, kappa.
# The matrices are populated s.t. g_up[l] = g(n, l; kappa, l+1)
#                                 g_dn[l] = g(n, l; kappa, l-1).
# Reference: Burgess A.,1965, MmRAS..69....1B, Eqs. (28)-(34).
# **********************************************************************************************************************/

def populate_gnlk(nmax, n, kappa):
    gnk_up = np.zeros((nmax, len(kappa)))
    gnk_dn = np.zeros((nmax, len(kappa)))
    
    k2 = kappa**2
    n2 = n**2
        
    log_product = 0.
    for s in range(1, n+1):
        log_product += np.log(1.0 + s**2 * k2)
    log_init = (0.5 * (np.log(np.pi/2) - lgamma(2.0 * n)) + np.log(4.0) + n * np.log(4.0 * n) + 0.5 * log_product
             - 0.5 * np.log(1.0 - np.exp(-2.0 * np.pi / kappa)) - 2.0 * np.arctan(n * kappa) / kappa 
             - (n + 2.0) * np.log(1.0 + n2 * k2))
    gnk_up[n-1] = np.exp(log_init)
    gnk_dn[n-1] = 0.5 * np.sqrt((1.0 + n2 * k2) / (1.0 + (n - 1.0) * (n - 1.0) * k2)) / n * gnk_up[n-1]
 
    if n > 1:
        gnk_up[n-2] = 0.5 * np.sqrt((2.0 * n - 1.0) * (1.0 + n2 * k2)) * gnk_up[n-1]    
        gnk_dn[n-2] = 0.5 * (4.0 + (n - 1.0) * (1.0 + n2 * k2)) * np.sqrt(
            (2.0 * n - 1.0) / (1.0 + (n - 2.0) * (n - 2.0) * k2)) / n * gnk_dn[n-1]  
        for l in range(n-1,1,-1): 
            l2 = l**2
            gnk_up[l-2] = 0.5 * (((4.0 * (n2 - l2) + l * (2.0 * l - 1.0) * (1.0 + n2 * k2)) * gnk_up[l-1]
                              - 2.0 * n * np.sqrt((n2 - l2) * (1.0 + (l + 1.0) * (l + 1.0) * k2)) * gnk_up[l])
                                / np.sqrt((n2 - (l - 1.0) * (l - 1.0)) * (1.0 + l2 * k2)) / n)
        
        for l in range(n-2,0,-1):  
            l2 = l**2
            gnk_dn[l-1] = 0.5 * (((4.0 * (n2 - l2) + l * (2.0 * l + 1.0) * (1.0 + n2 * k2)) * gnk_dn[l]
                              - 2.0 * n * np.sqrt((n2 - (l + 1.0) * (l + 1.0)) * (1.0 + l2 * k2)) * gnk_dn[l+1])
                              / np.sqrt((n2 - l2) * (1.0 + (l - 1.0) * (l - 1.0) * k2)) / n)
    return np.transpose(gnk_up), np.transpose(gnk_dn)

# /********************************************************************************************
# k2[n][ik] because boundaries depend on n
#  ********************************************************************************************/

def populate_k2_and_g(nmax, Tm):
    k2_tab = np.zeros((nmax+1,10 * (NBINS-1) + 11))
    g = {key: np.zeros((nmax+1,Nkappa,nmax)) for key in ['up', 'dn']}
    k2max = 7e2*Tm/phys.rydberg        

    for n in range(1,nmax+1):
        k2min = 1e-25/n**2
        bigBins = np.logspace(np.log10(k2min), np.log10(k2max), NBINS + 1)
        iBig = np.arange(NBINS)
        temp = np.linspace(bigBins[iBig], bigBins[iBig+1], 11)
        for i in range(11):
            k2_tab[n][10 * iBig + i] = temp[i]
        ik = np.arange(10 * NBINS + 1)
        g['up'][n,ik], g['dn'][n,ik] = populate_gnlk(nmax, n, np.sqrt(k2_tab[n,ik]))  
    return k2_tab, g


# /******************************************************************************************* 
# 11 point Newton-Cotes integration.
# Inputs: an 11-point array x, an 11-point array f(x).
# Output: \int f(x) dx over the interval provided.
# ********************************************************************************************/

def Newton_Cotes_11pt(x, f):
    h = (x[10] - x[0])/10.0 #/* step size */

    return (5.0 * h * (16067.0 * (f[0] + f[10]) + 106300.0 * (f[1] + f[9]) 
                      - 48525.0 * (f[2] + f[8]) + 272400.0 * (f[3] + f[7])
                      - 260550.0 * (f[4] + f[6]) + 427368.0 * f[5]) / 299376.0)

# /********************************************************************************************* 
# Populating the photoionization rates beta(n, l, Tr)
# Input: beta[nmax+1][nmax], Tr in eV, nmax
# and the precomputed tables n2k2, g_up[nmax+1][Nkappa][nmax], g_dn[nmax+1][Nkappa][nmax], 
# where Nkappa = 10 * NBINS + 1
# *********************************************************************************************/

def populate_beta(Tr, nmax, Delta_f=None):
    """ From prefac we see that the units are s^-1 """
    beta = np.zeros((nmax+1,nmax))

    if Delta_f == None:
        Delta_f = lambda l : 0

    def f_gamma(Ennp):
        return np.exp(-Ennp / Tr)/(1.0 - np.exp(-Ennp / Tr)) + Delta_f(Ennp)

    for n in range(1, nmax+1):
        for l in range(n):
            beta[n][l] = bf.beta_nl(n, l, f_gamma=f_gamma)

    return beta

# /********************************************************************************************* 
# Populating the recombination coefficients alpha(n, l, Tm, Tr) 
# Input: alpha[nmax+1][nmax], Tm, Tr in eV, nmax
# and the precomputed tables n2k2, g_up[nmax+1][Nkappa][nmax], g_dn[nmax+1][Nkappa][nmax], 
# where Nkappa = 10 * NBINS + 1
# *********************************************************************************************/


# /****************************************************************************************/

def populate_alpha(Tm, Tr, nmax, Delta_f=None, stimulated_emission=True):
    alpha = np.zeros((int(nmax+1),nmax))

    if stimulated_emission:

        if Delta_f == None:
            Delta_f = lambda l : 0

        def f_gamma(Ennp):
            return np.exp(-Ennp / Tr)/(1.0 - np.exp(-Ennp / Tr)) + Delta_f(Ennp)
    else:
        f_gamma = None

    for n in np.arange(1,nmax+1):
        for l in np.arange(n):
            alpha[n][l] = bf.alpha_nl(
                n, l, Tm, f_gamma=f_gamma, stimulated_emission=stimulated_emission
            )

    return alpha


def get_transition_energies(nmax):
    """
    Compute the exact energy bins for transitions between excited state of hydrogen.
    This includes an extra bin at 20 eV to represent bound-free transitions.

    Parameters
    ----------
    nmax : int
        Highest excited state to be included

    Returns
    -------
    H_engs : array
    """

    H_engs = np.zeros((nmax+1,nmax+1))
    for n1 in np.arange(1,nmax+1):
        #H_engs[0,n1] = phys.rydberg / n1**2
        for n2 in range(1,n1):
            H_engs[n1,n2] = phys.rydberg * ((1/n2)**2 - (1/n1)**2)
        
    H_engs = np.sort(np.unique(H_engs))
    # Add a separate energy bin to temporarily represent 2s->1s
    H_engs = np.concatenate((H_engs, [20]))
    return H_engs[1:]


def get_distortion_and_ionization(
        rs, dt, xHI, Tm, nmax, spec_2s1s,
        Delta_f=None, cross_check=False, include_2s1s=True, include_BF=True,
        fexc_switch=False, deposited_exc_arr=None, elec_spec=None,
        distortion=None, H_states=None, rate_func_eng=None, A_1snp=None
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
    spec_2s1s : Spectrum
        2s -> 1s emission spectrum of photons, normalized so spec_2s1s.totN()=2
    Delta_f : function
        photon phase space density as a function of energy, minus f_BB
    cross_check : bool
        if True, set xHI to its standard value
    include_2s1s : bool
        includes the 2s -> 1s photons to the output distortion
    include_BF : bool
        includes the bound-free transition photons to the output distortion
    fexc_switch : bool
    deposited_exc_arr elec_spec distortion H_states rate_func_eng A_1snp

    Returns
    -------
    Pto1s_many, PtoCont_many : vectors of probabilities for transitioning to
        1s, continuum after many transitions
    transition_specs : dictionary of photon spectra, labeled by
        initial excited state (in N, not dNdE)
    """

    if cross_check:
        xHI = phys.xHI_std(rs)
        # Tm = phys.TCMB(rs)

    # Number of Hydrogen states at or below n=nmax
    num_states = int(nmax*(nmax+1)/2)

    # Mapping from spectroscopic letters to numbers
    # spectroscopic_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
    spectroscopic_map = {0: 's', 1: 'p', 2: 'd', 3: 'f'}

    def num_to_l(ll):
        if ll < 4:
            return spectroscopic_map[l]

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
    R = populate_radial(nmax)  # Need not be recomputed every time
    BB = populate_bound_bound(nmax, Tr, R, Delta_f=Delta_f)
    alpha = populate_alpha(Tm, Tr, nmax, Delta_f=Delta_f,
                           stimulated_emission=True)
    beta = populate_beta(Tr, nmax, Delta_f=Delta_f)

    # Include sobolev optical depth
    for n in np.arange(2, nmax+1, 1):
        BB['dn'][n][1][1] *= p_np_1s(n, rs, xHI=xHI)
        BB['up'][1][n][0] *= p_np_1s(n, rs, xHI=xHI)

    ### Build matrix K_ij = R_ji/R_i,tot and source term ###
    K = np.zeros((num_states, num_states))

    b = np.zeros(num_states)
    db = np.zeros(num_states)

    # excitations from energy injection -- both photoexcitation and
    # electron collisions
    if fexc_switch:
        delta_b = f_exc_to_b_numerator(deposited_exc_arr,
                                       elec_spec, distortion,
                                       H_states, dt, rate_func_eng,
                                       nmax, xHI)

    for nl in np.arange(num_states):
        n, l = states_n[nl], states_l[nl]
        tot_rate = (
            np.sum(BB['dn'][n, :, l]) + np.sum(BB['up'][n, :, l]) + beta[n][l]
        )

        # Construct the matrix
        if l != 0:
            K[nl, states_l == l-1] = BB['up'][l:, n, l-1]/tot_rate

        if l != nmax-1:
            K[nl, states_l == l+1] = BB['dn'][l+2:, n, l+1]/tot_rate

        # Special 2s->1s transition
        if nl == 0:
            K[0][1] = BB['dn'][2][1][0] / tot_rate

        if nl == 1:
            # Detalied Balance
            K[1][0] = BB['dn'][2][1][0]*np.exp((E(2)-E(1))/Tr) / tot_rate


        ## Construct the source term ##

        # excitations from direct recombinations
        b[nl] += xe**2 * nH * alpha[n][l]

        # excitations from 1s->nl transitions
        if l == 1:
            b[nl] += xHI*BB['up'][1, n, 0]

        elif nl == 1:
            # 1s to 2s transition from detailed balance
            b[nl] += xHI*BB['dn'][2][1][0]*np.exp(-phys.lya_eng/Tr)

        # Add DM contribution to source term
        if fexc_switch:
            spec_ind = str(n) + num_to_l(l)
            if spec_ind in delta_b.keys():
                db[nl] = delta_b[spec_ind]

        db[nl] /= tot_rate
        b[nl] /= tot_rate

    mat = np.identity(num_states-1) - K[1:, 1:]
    x_vec = np.linalg.solve(mat, b[1:] + db[1:])
    # !!! I should be able to set xHI = 1 - sum(x_full) - xe,
    # but instead I'm stuck with 1-xHII
    x_full = np.append(xHI, x_vec)
    x_full0 = np.append(xHI, np.linalg.solve(mat, b[1:]))

    ###
    # Now calculate the total ionization and distortion
    ###

    E_current = 0
    ind_current = 0
    H_engs = np.zeros(num_states)
    eng = spec_2s1s.eng

    Nphot_cascade = np.zeros(num_states)
    BF_spec = Spectrum(eng, np.zeros_like(eng), spec_type='dNdE')
    beta_MLA, alpha_MLA = 0, 0

    def f_gamma(Ennp):
        return np.exp(-Ennp / Tr)/(1.0 - np.exp(-Ennp / Tr)) + Delta_f(Ennp)

    for nl in np.arange(num_states):
        n, l = states_n[nl], states_l[nl]
        if nl > 0:
            beta_MLA += x_full[nl] * beta[n][l]
            alpha_MLA += alpha[n][l]

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

        BF_tmp = nH**2 * xe**2 * bf.gamma_nl(
            n, l, Tm, T_r=Tr, f_gamma=f_gamma, stimulated_emission=True
        )/nB * dt
        BF_tmp -= nH * x_full[nl] * bf.xi_nl(
            n, l, T_r=Tr, f_gamma=f_gamma)/nB * dt
        BF_tmp.rebin(eng)
        BF_spec.dNdE += BF_tmp.dNdE

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
    BF_spec.rebin(eng)
    if include_BF:
        transition_spec.N += BF_spec.N

    # Add the 2s-1s component
    amp_2s1s = nH * BB['dn'][2, 1, 0] * (
        x_full[1] - x_full[0]*np.exp(-phys.lya_eng/Tr)
    ) / nB * dt
    if include_2s1s:
        transition_spec.N += amp_2s1s * spec_2s1s.N

    return alpha_MLA, beta_MLA, transition_spec


def absorb_photons(distortion, H_states, A_1snp, dt, x1s):
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

            # Technically, I should subtract off inverse process
            #   (stimulated emission) but there are so few x_np
            #   states that I neglect this process.
            # I include the heaviside step function to only allow
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

    # np -> 1s decay rate
    R_1snp = Hey_R_initial(np.arange(2, nmax+1), 1)

    A_1snp = 1/3 * phys.rydberg / phys.hbar * (
        phys.alpha * (1-1/np.arange(2, nmax+1)**2)
    )**3 * R_1snp**2  # need not be recomputed every time

    # Convert from energy per baryon to
    # (dimensionless) fraction of injected energy
    norm = nB / rate_func_eng(rs) / dt

    # fraction of energy deposited into excitation into the i-th state
    f_exc = {state: 0 for state in H_states}

    # Allow H(1s) to absorb line photons from distortion (E per baryon)
    dE_absorbed = absorb_photons(distortion, H_states, A_1snp, dt, x1s)

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
