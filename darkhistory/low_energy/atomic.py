"""Atomic excitation state transitions, ionization, and recombination functions."""

import numpy as np 
from darkhistory import physics as phys 
from darkhistory.spec.spectrum import Spectrum
from   darkhistory.spec.spectra import Spectra
from darkhistory.spec.spectools import discretize
import scipy.special
from config import load_data
from datetime import datetime

NBINS = 100
Nkappa = 10 * NBINS + 1

hplanck = phys.hbar*2*np.pi
mue      = phys.me/(1+phys.me/phys.mp)

#*************************************************#
#A_{nl} coefficient defined in Hey (2006), Eq. (9)#
#*************************************************#

def Hey_A(n, l):
    return np.sqrt(n**2-l**2)/(n*l)

#/*****************************************************************#
#Absolute value of R_{n', n'- 1}^{n n'} as in Hey (2006), Eq. (B.4)
#Assumes n > np.
#******************************************************************#

lgamma = scipy.special.loggamma
def Hey_R_initial(n, n_p):
#     return (-1)**(n-n_p-1) * 2**(2*n_p+2) * np.sqrt(
#         fact(n+n_p)/fact(n-n_p-1)/fact(2*n_p-1)) * (
#         n_p/n)**(n_p+2) * (1-n_p/n)**(n-n_p-2)/(1+n_p/n)**(n+n_p+2)
    return np.exp(
        (2.0*n_p + 2.0) * np.log(2.0) +
        0.5 * (lgamma(n+n_p+1) - lgamma(n-n_p) - lgamma(2.0*n_p))
        + (n_p + 2.0) * np.log(n_p/n)
        + (n-n_p - 2.0) * np.log(1.0 - n_p/n) - (n + n_p + 2.0) * np.log(1.0 + n_p/n)
    )

#/*********************************************************************
#Populates a matrix with the radial matrix elements.
#Inputs: two [nmax+1][nmax+1][nmax] matrices Rup, Rdn, nmax.
#Rup is populated s.t. R_up[n][n'][l] = R([n,l],[n',l+1])
#Rdn is populated s.t. R_dn[n][n'][l] = R([n,l],[n',l-1])
#Assumption: n > n'
#**********************************************************************/

def populate_radial(nmax):
    R_up = np.zeros((nmax+1,nmax,nmax))
    R_dn = np.zeros((nmax+1,nmax,nmax))

    for n in np.arange(2,nmax+1,1):
        for n_p in np.arange(1,n):
            #/* Initial conditions: Hey (2006) Eq. (B.4). */
            R_dn[n][n_p][n_p] = Hey_R_initial(n, n_p)
            R_up[n][n_p][n_p-1] = R_up[n][n_p][n_p] = 0
            for l in np.arange(n_p-1,0,-1):
                #/* Hey (52-53) */
                R_dn[n][n_p][l] = ((2*l+1) * Hey_A(n, l+1) * R_dn[n][n_p][l+1]
                    + Hey_A(n_p, l+1) * R_up[n][n_p][l]) / (2.0 * l * Hey_A(n_p, l))

                R_up[n][n_p][l-1] = ((2*l+1) * Hey_A(n_p, l + 1) * R_up[n][n_p][l]
                    + Hey_A(n, l+1) * R_dn[n][n_p][l+1]) / (2.0 * l * Hey_A(n, l))

    return {'up': R_up, 'dn': R_dn}

# /***************************************************************************************************************
# Populates two matrices with the bound-bound rates for emission and absorption,
#  in a black-body radiation field.
# Inputs : two [nmax+1][nmax+1][nmax] matrices BB_up and BB_dn, nmax, Tr (IN EV !),
#  and the two precomputed matrices of radial matrix elements R_up and R_dn.
# BB_up[n][n'][l] = A([n,l]-> [n',l+1]) * (1 + f(E_{nn'}))                if n > n'
#                 = (2l+3)/(2l+1) exp(-E_{nn'}/Tr) * BB_dn[n'][n][l+1]    if n < n'
# BB_dn[n][n'][l] = A([n,l]-> [n',l-1]) * (1 + f(E_{nn'}))                if n > n'
#                 = (2l-1)/(2l+1) exp(-E_{nn'}/Tr) * BB_up[n'][n][l-1]    if n < n'
#  *****************************************************************************************************************/

def populate_bound_bound(nmax, Tr, R, ZEROTEMP=1e-10, f_dist=None):
    BB = {key: np.zeros((nmax+1,nmax+1,nmax)) for key in ['up', 'dn']}

    for n in np.arange(2,nmax+1,1):
        n2 = n**2
        for n_p in np.arange(1,n,1):
            n_p2 = n_p**2
            Ennp = (1/n_p2 - 1/n2) * phys.rydberg;
            if (Tr < ZEROTEMP):    #/* if Tr = 0*/
                fEnnp = 0.0;
            else:
                fEnnp = np.exp(-Ennp/Tr)/(1-np.exp(-Ennp/Tr)) + f_dist(Ennp)
                #fEnnp = 1/(np.exp(Ennp/Tr)-1)

            common_factor = 2*np.pi/3 * phys.rydberg / hplanck * (
                phys.alpha * (1/n_p2 - 1/n2))**3

            for l in np.arange(0,n_p+1,1): #/* Spont + stim emission */
                A_up = common_factor * (l+1) / (2*l+1) * R['up'][n][n_p][l]**2
                A_dn = common_factor *   l   / (2*l+1) * R['dn'][n][n_p][l]**2
                BB['up'][n][n_p][l] = A_up * (1+fEnnp)
                BB['dn'][n][n_p][l] = A_dn * (1+fEnnp)

            BB['up'][n][n_p][n_p] = BB['up'][n][n_p][n_p-1] = 0.0   #/* No l' >= n' */
            BB['dn'][n][n_p][0]   = 0.0                          #/* No l' < 0   */
            for l in np.arange(0,n_p,1): #/* Absorption obtained by detailed balance */
                if (Tr < ZEROTEMP):  #/* if Tr = 0 */
                    BB['up'][n_p][n][l] = BB['dn'][n_p][n][l+1] = 0.0;
                else:
                    #BB['up'][n_p][n][l]   = (2*l+3)/(2*l+1) * np.exp(-Ennp/Tr) * BB['dn'][n][n_p][l+1]
                    #BB['dn'][n_p][n][l+1] = (2*l+1)/(2*l+3) * np.exp(-Ennp/Tr) * BB['up'][n][n_p][l]
                    
                    # When adding a distortion, detailed balance is inconvenient.
                    # Instead, take away the 1+fEnnp from a couple of lines above, 
                    # then replace it with fEnnp (that's all detailed balance was doing).
                    BB['up'][n_p][n][l]   = (2*l+3)/(2*l+1) * BB['dn'][n][n_p][l+1]/(1+fEnnp) * fEnnp
                    BB['dn'][n_p][n][l+1] = (2*l+1)/(2*l+3) * BB['up'][n][n_p][l]  /(1+fEnnp) * fEnnp

    return BB

#astro-ph/9912182 Eq. 40
def tau_np_1s(n, R, rs, xHI=None):
    l=1
    nu = (1 - 1/n**2) * phys.rydberg/hplanck
    lam = phys.c/nu
    if xHI == None:
        xHI = 1-phys.xHII_std(rs)
    nHI = xHI * phys.nH*rs**3
    pre = lam**3 * nHI / (8*np.pi*phys.hubble(rs))
    
    common_factor = 2*np.pi/3 * phys.rydberg / hplanck * (
        phys.alpha * (1 - 1/n**2))**3
    
    A_dn = common_factor * l/(2*l+1) * R['dn'][n][1][l]**2
    g = (2*l+1)/(2*l-1)
    return pre * A_dn * g 

# Eq. 41
def p_np_1s(n, R, rs, xHI=None):
    tau = tau_np_1s(n, R, rs, xHI=xHI)
    return (1-np.exp(-tau))/tau

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
            l2 = l**2;
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
        k2min = 1e-25/n/n
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

def populate_beta(Tm, Tr, nmax, k2_tab, g, f_dist=None):
    k2 = np.zeros((11, NBINS))
    int_b = np.zeros((11, NBINS))
    common_factor =  2.0/3.0 * phys.alpha**3 * phys.rydberg/hplanck
    beta = np.zeros((nmax+1,nmax))

    for n in range(1, nmax+1):
        n2 = n**2
        for l in range(n):
            iBin = np.arange(NBINS)
            for i in range(11):
                ik = 10 * iBin + i
                k2[i] = k2_tab[n][ik]
                if (Tr < 1e-10):      #/* Flag meaning TR = 0 */
                    int_b[i] = 0.0
                else:
                    Ennp = phys.rydberg / (k2[i] + 1.0 / n2)
                    fEnnp = np.exp(-Ennp / Tr)/(1.0 - np.exp(-Ennp / Tr)) + f_dist(Ennp)
                    # Burgess, Eqn 1
                    #int_b[i] = ((1.0 + k2[i]*n2)**3 / (np.exp(phys.rydberg / Tr * (k2[i] + 1.0 / n2)) - 1.0) * 
                    int_b[i] = ((1.0 + k2[i]*n2)**3 * fEnnp * ((l + 1.0) * g['up'][n,ik,l]**2 + l * g['dn'][n,ik,l]**2))
            beta[n][l] += np.sum(Newton_Cotes_11pt(k2, int_b))
            beta[n][l] *= common_factor / n2 / (2.0 * l + 1.0)
    return beta

# /********************************************************************************************* 
# Populating the recombination coefficients alpha(n, l, Tm, Tr) 
# Input: alpha[nmax+1][nmax], Tm, Tr in eV, nmax
# and the precomputed tables n2k2, g_up[nmax+1][Nkappa][nmax], g_dn[nmax+1][Nkappa][nmax], 
# where Nkappa = 10 * NBINS + 1
# *********************************************************************************************/


# /****************************************************************************************/

def populate_alpha(Tm, Tr, nmax, k2_tab, g, f_dist=None):
    k2 = np.zeros((11, NBINS))
    int_a = np.zeros((11, NBINS))   
    common_factor = (2.0/3.0 * phys.alpha**3 * phys.rydberg/hplanck 
                     * (hplanck**2 * phys.c**2/(2.0 * np.pi * mue * Tm))**1.5)
    alpha = np.zeros((nmax+1,nmax))
    
    for n in range(1,nmax+1):
        n2 = n**2
        for l in range(n):
            iBin = np.arange(NBINS)
            for i in range(11):
                ik = 10 * iBin + i
                k2[i] = k2_tab[n][ik]
                if Tr<1e-10:
                    int_a[i] = 0
                else:
                    Ennp = phys.rydberg / (k2[i] + 1.0 / n2)
                    fEnnp = np.exp(-Ennp / Tr)/(1.0 - np.exp(-Ennp / Tr)) + f_dist(Ennp)
                #    int_a[i] = ((1.0 + n2 *k2[i])**3 * np.exp(-k2[i] * phys.rydberg / Tm) *
                    int_a[i] = ((1.0 + n2 *k2[i])**3 * fEnnp *
                                 ((l + 1.0) * g['up'][n,ik,l] * g['up'][n,ik,l] 
                                  + l * g['dn'][n,ik,l] * g['dn'][n,ik,l]))
                #if Tr > 1e-10:
                #    int_a[i] /= (1. - np.exp(-phys.rydberg/Tr*(k2[i] + 1./n2)))
            alpha[n][l] += np.sum(Newton_Cotes_11pt(k2, int_a))
            alpha[n][l] *= common_factor / n2
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
    H_engs_Cont = np.zeros(nmax+1)
    for n1 in range(1,nmax+1):
        H_engs_Cont[n1] = phys.rydberg / n1**2
        H_engs[0,n1] = phys.rydberg / n1**2
        for n2 in range(1,n1):
            H_engs[n1,n2] = phys.rydberg * ((1/n2)**2 - (1/n1)**2)
        
    H_engs = np.sort(np.unique(H_engs))
    # Add a separate energy bin to temporarily represent 2s->1s
    H_engs = np.concatenate((H_engs,[20]))
    return H_engs


def get_total_transition(rs, xHI, Tm, nmax, f_dist = None):
    """
    Calculate either the probabilities to switch between between states or the 
    resulting photon spectra after many transitions.

    Parameters
    ----------
    rs : redshift
    xHI : neutral fraction of hydrogen
    Tm : matter temperature
    mode : 'prob' to return probabilities or 'spec' to return spectra
    nmax : int
        Highest excited state to be included

    Returns
    -------
    Pto1s_many, PtoCont_many : vectors of probabilities for transitioning to
        1s, continuum after many transitions
    transition_specs : dictionary of photon spectra, labeled by
        initial excited state (in N, not dNdE)
    """

    if f_dist==None:
        f_dist = lambda a : 0

    #Radiation Temperature
    Tr = phys.TCMB(rs)

    #Get the transition rates
    R = populate_radial(nmax)
    BB = populate_bound_bound(nmax, Tr, R, f_dist=f_dist)
    k2_tab, g = populate_k2_and_g(nmax, Tm)
    alpha = populate_alpha(Tm, Tr, nmax, k2_tab, g, f_dist=f_dist)
    beta = populate_beta(Tm, Tr, nmax, k2_tab, g, f_dist=f_dist)

    #Get transition energies
    H_engs = get_transition_energies(nmax)

    #Include sobolev optical depth
    for n in np.arange(2,nmax+1,1):
        BB['dn'][n][1][1] = p_np_1s(n, R, rs, xHI=xHI) * BB['dn'][n][1][1]
    #Include forbidden 2s->1s transition
    BB['dn'][2][1][0] = phys.width_2s1s_H

    # Get indices where alpha != 0 --> correspond to ground and excited states
    #nonzero_n, nonzero_l = alpha.nonzero()
    nonzero_n = np.concatenate([list(map(int,k*np.ones(k))) for k in np.arange(1,nmax+1)])
    nonzero_l = np.concatenate([np.arange(k) for k in np.arange(1,nmax+1)])

    # Make array that has the n of each excited state
    ns = np.zeros(int(nmax*(nmax+1)/2))
    counter = 0
    for i in range(1,nmax+1):
        ns[counter:counter + i] = i
        counter += i

    ### Build probability matrix for single transition, P_ij
    P_matrix = np.zeros((len(nonzero_n)+1, len(nonzero_n)+1))
    
    # Ground and continuum states are sink states
    P_matrix[0,0] = 1
    P_matrix[-1,-1] = 1
    
    for nl in range(1, len(nonzero_n)):
        # Find indices for possible states to transition to (l must change by 1)
        # Get the rates for transitioning to those states, as well as total rate
        if nonzero_l[nl] != 0:
            dn_inds = np.where(nonzero_l == nonzero_l[nl]-1)[0]
            dn_rates = BB['dn'][nonzero_n[nl], nonzero_n[dn_inds], nonzero_l[nl]]
        else:
            # If angular momentum is 0, can only transition up
            dn_rates = 0
            
        up_inds = np.where(nonzero_l == nonzero_l[nl]+1)[0]
        up_rates = BB['up'][nonzero_n[nl], nonzero_n[up_inds], nonzero_l[nl]]
        tot_rate = np.sum(up_rates) + np.sum(dn_rates) + beta[nonzero_n[nl]][nonzero_l[nl]]
        
        # Special 2s->1s transition
        if nl == 1:
            tot_rate += BB['dn'][2][1][0]
            P_matrix[1][0] = BB['dn'][2][1][0] / tot_rate
            
        # Normalize by total rate for each state to get probabilities instead of rates
        if nonzero_l[nl] != 0:
            P_matrix[nl][dn_inds] = dn_rates / tot_rate
        P_matrix[nl][up_inds] = up_rates / tot_rate
        P_matrix[nl][-1] = beta[nonzero_n[nl]][nonzero_l[nl]] / tot_rate
    
    # Vector of probabilities for single transition
    Pto1s = P_matrix[1:-1,0]
    PtoCont = P_matrix[1:-1,-1]
        
    P_sub = P_matrix[1:-1,1:-1]
    P_series = np.linalg.inv(np.identity(len(Pto1s)) - P_sub)

    ### Calculate probability of any state going to 1s or continuum after many transitions
    #if mode == 'prob':
    # Geometric series with single-transition matrix
    Pto1s_many = np.dot(P_series, Pto1s)
    PtoCont_many = np.dot(P_series, PtoCont)
    
    #return Pto1s_many, PtoCont_many
                
    ### Build matrix for photon spectra from single transitions
    #elif mode == 'spec':
    # Since this is a photon spectrum, the length of the last axis is the number of photon energy bins
    NE_single = np.zeros((len(nonzero_n)+1, len(nonzero_n)+1, len(H_engs)))
    
    # We are only putting one photon in one bin, so spectrum already correctly normalized
    # Run over initial excited states
    for nli in range(len(nonzero_n)):
        #Bound-free transitions
        eng = phys.rydberg / ns[nli]**2
        # Find correct energy bin
        ind = np.where(H_engs <= eng)[0][-1]
        # Add 1 photon for emission, -1 for absorption
        NE_single[nli,-1,ind] = -1
        NE_single[-1,nli,ind] = 1
        
        #Bound-bound transitions
        # Run over final states
        for nlf in range(len(nonzero_n)):
            # Put 2s->1s transition in special energy bin
            if (nli == 1) and (nlf == 0):
                ind = np.where(H_engs <= 20)[0][-1]
                NE_single[nli,nlf,ind] = 1
            else:
                # Energy of photon emitted/absorped
                eng = phys.rydberg * ((1/ns[nlf])**2 - (1/ns[nli])**2)
                if eng != 0.0:
                    # Find correct energy bin
                    ind = np.where(H_engs <= abs(eng))[0][-1]
                    # Add 1 photon if emission, -1 for absorption
                    NE_single[nli,nlf,ind] = np.sign(eng)
    
    ### Calculate spectrum from cascading to 1s after many transitions
    # PNE_1s = np.dot(P_matrix[:-1,:-1], NE_single[:-1,:-1])
    # PNE_diag_1s = np.transpose(np.diagonal(PNE_1s))
    PNE_diag_1s = np.zeros((len(nonzero_n), len(H_engs)))
    for i in range(len(nonzero_n)):
        PNE_diag_1s[i] = np.sum(P_matrix[:-1,:-1][i] * np.transpose(NE_single[:-1,:-1][i]), axis=-1)
    NE_1s = np.dot(P_series, PNE_diag_1s[1:])
    
    ### Calculate spectrum from going up to continuum after many transitions
    # PNE_Cont = np.dot(P_matrix[1:,1:], NE_single[1:,1:])
    # PNE_diag_Cont = np.transpose(np.diagonal(PNE_Cont))
    PNE_diag_Cont = np.zeros((len(nonzero_n), len(H_engs)))
    for i in range(len(nonzero_n)):
        PNE_diag_Cont[i] = np.sum(P_matrix[1:,1:][i] * np.transpose(NE_single[1:,1:][i]), axis=-1)
    NE_Cont = np.dot(P_series, PNE_diag_Cont[:-1])
            
    # Most photons are double counted between transitions to 1s and to continuum.
    # What the continuum spectrum does not have are the photons corresponding to 
    # the final transition to 1s, which all have energy > 10 eV.
    phot1s_inds = np.where(H_engs>10)[0]
    #photCont_inds = np.where(np.in1d(H_engs,photengs_Cont))
    NE_Cont[:,phot1s_inds] = NE_1s[:,phot1s_inds]
    
    # Use Spectrum class to rebin
    binning = load_data('binning')
    spectroscopic = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'q', 'r', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    
    # Replace 2s -> 1s transition with continuous two-photon spectrum
    spec_2s1s = discretize(binning['phot'],phys.dNdE_2s1s)
    spec_2s1s.switch_spec_type()
    amp_2s1s = np.array(NE_Cont[:, -1])
    NE_Cont[:, -1] = 0
    
    P_1s, P_ion, transition_specs = {}, {}, {}
    for i in range(1,len(nonzero_n)):
        state_string = f'{nonzero_n[i]}'+spectroscopic[nonzero_l[i]]

        P_1s[state_string]  = Pto1s_many[i-1]
        P_ion[state_string] = PtoCont_many[i-1]

        transition_specs[state_string] = Spectrum(H_engs, NE_Cont[i-1], spec_type='N')
        transition_specs[state_string].rebin(binning['phot'])
        transition_specs[state_string] += amp_2s1s[i-1]*spec_2s1s

    return P_1s, P_ion, transition_specs        

    #else:
    #    raise TypeError("mode not specified; must either be 'prob' or 'spec'.")

def process_excitations(rs, xHII, Tm, eleceng, photeng, deposited_exc_vec, H_states, nmax, f_dist):

    exc_grid = np.zeros((np.array(H_states).size, eleceng.size))
    exc_phot_spectra = Spectra(
        spec_arr = np.zeros((eleceng.size,photeng.size)), eng=photeng,
        in_eng=eleceng, spec_type='N',
        rs=np.ones_like(eleceng)*rs
    )
    #Spectra that result from ONE atom in an excited state cascading to 1s or continuum
    P_1s, P_ion, one_transition = get_total_transition(rs, 1-xHII, Tm, nmax, f_dist)

    for i, state in enumerate(H_states):

        #Use energy deposited in excitation to infer the number of excited (n>2) atoms
        exc_grid[i] += deposited_exc_vec[state]/phys.H_exc_eng(state)

        #Multiply number of excited atoms by corresponding spectra and add it in
        exc_phot_spectra += Spectra(
            spec_arr=np.outer(exc_grid[i], one_transition[state].N), eng=photeng,
            in_eng=eleceng, spec_type='N',
            rs=np.ones_like(eleceng)*rs
        )

    return P_1s, P_ion, exc_phot_spectra

def yim_distortion(nu, amp, T, dist_type):
    # Given amplitude, frequency [s^-1], temperature [K], 
    #     and distortion type ('mu' or 'y')
    # returns intensity of distortion in units of [ev / cm^2]
    x = 2 * np.pi * phys.hbar * nu / phys.kB / T
    
    # can we add i-type distortions as well?
    
    if dist_type == 'y':
        return (amp * 4. * np.pi * phys.hbar * nu**3 / phys.c**2
                * x * np.exp(x) / (np.exp(x) - 1.)**2
                *(x * (np.exp(x) + 1.) / (np.exp(x) - 1.) - 4.)
               )
    
    elif dist_type == 'mu':
        return (amp * 4. * np.pi * phys.hbar * nu**3 / phys.c**2
                * np.exp(x) / (np.exp(x) - 1.)**2
                *(x / 2.19 - 1.)
               )
