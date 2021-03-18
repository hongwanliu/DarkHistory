"""Atomic excitation state transitions, ionization, and recombination functions."""

import numpy as np 

from darkhistory import physics as phys 
import scipy.special
hplanck = phys.hbar*2*np.pi

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

def populate_bound_bound(nmax, Tr, R, ZEROTEMP=1e-10):
    BB = {key: np.zeros((nmax+1,nmax+1,nmax)) for key in ['up', 'dn']}
    for n in np.arange(2,nmax+1,1):
        n2 = n**2
        for n_p in np.arange(1,n,1):
            n_p2 = n_p**2
            Ennp = (1/n_p2 - 1/n2) * phys.rydberg;
            if (Tr < ZEROTEMP):    #/* if Tr = 0*/
                fEnnp = 0.0;
            else:
                fEnnp = np.exp(-Ennp/Tr)/(1-np.exp(-Ennp/Tr))
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
                    BB['up'][n_p][n][l]   = (2*l+3)/(2*l+1) * np.exp(-Ennp/Tr) * BB['dn'][n][n_p][l+1]
                    BB['dn'][n_p][n][l+1] = (2*l+1)/(2*l+3) * np.exp(-Ennp/Tr) * BB['up'][n][n_p][l]

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

EI       = 13.5982860719383 #/*13.60569193 / (1.0 + me_mp)*/       /* Ionization energy of hydrogen in eV, accounting for reduced mass */
alpha_fs = 7.2973525376e-3                                                     #/* Fine structure constant */
hPlanck  = 4.13566733e-15                                                      #/* Planck's constant in eV*s */
cLight   = 2.99792458e10                                                       #/* Speed of light in cm/s */
mue      = 510720.762779219              #/*me / (1.0 + me_mp)*/               #/* Reduced mass of hydrogen in eV*/

# /*********************************************************************************************************************
# Populates two matrices with the coefficients g(n, l; kappa, l+1) and g(n, l; kappa, l-1).
# Input: two [nmax] matrices g_up and g_dn, n, kappa.
# The matrices are populated s.t. g_up[l] = g(n, l; kappa, l+1)
#                                 g_dn[l] = g(n, l; kappa, l-1).
# Reference: Burgess A.,1965, MmRAS..69....1B, Eqs. (28)-(34).
# **********************************************************************************************************************/

def populate_gnlk(nmax, n, kappa):
    gnk_up = np.zeros(nmax)
    gnk_dn = np.zeros(nmax)

    k2 = kappa**2
    n2 = n**2

    log_product = 0.0;

    for s in range(1, n+1):
        log_product = log_product + np.log(1.0 + s*s*k2)
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
    return gnk_up, gnk_dn


# /************************************************************************************************************************
# Populates one array n2k2[Nkappa] and two [nmax+1][Nkappa][nmax] matrices g_up, g_dn, s.t.
# - Nkappa = 10 * NBINS + 1
# - n2k2 has NBINS logarithmically spaced bins between n2k2min and n2k2max, each bin containing 11 linearly spaced values
# for later 11-point Newton-Cotes integration (as in Grin & Hirata, 2010).
# - g_up[n][ik][l] = g(n, l; kappa[ik], l+1)
# - g_dn[n][ik][l] = g(n, l; kappa[ik], l-1)
# *************************************************************************************************************************/

def populate_n2k2_and_g_old(nmax, NBINS):
    Nkappa = 10 * NBINS + 1
    n2k2 = np.zeros(10 * (NBINS-1) + 11)
    g = {key: np.zeros((nmax+1,Nkappa,nmax)) for key in ['up', 'dn']}

    n2k2min = 1e-25                             #/* Min value of n^2 kappa^2 table */
    n2k2max = 4.96e8                            #/* Max value of n^2 kappa^2 table */

    bigBins = np.zeros(NBINS + 1)
    temp = np.zeros(11)

    #/* Populating n2k2: Nbins logarithmically spaced bins, each has 10 sub-bins */
    bigBins = np.logspace(np.log10(n2k2min), np.log10(n2k2max), NBINS + 1)

    for iBig in range(NBINS):
        temp = np.linspace(bigBins[iBig], bigBins[iBig+1], 11)
        for i in range(11):
            n2k2[10 * iBig + i] = temp[i];

    #/* Now populating the g's */
    for n in range(1, nmax+1):
        for ik in range(10*NBINS + 1):
            g['up'][n][ik], g['dn'][n][ik] = populate_gnlk(nmax, n, np.sqrt(n2k2[ik]) / n)

    return n2k2, g


# /********************************************************************************************
# k2[n][ik] because boundaries depend on n
#  ********************************************************************************************/

def populate_k2_and_g(nmax, TM, NBINS):
    Nkappa = 10 * NBINS + 1
    k2_tab = np.zeros((nmax+1,10 * (NBINS-1) + 11))
    g = {key: np.zeros((nmax+1,Nkappa,nmax)) for key in ['up', 'dn']}
    k2max = 7e2*TM/EI

    for n in range(1,nmax+1):
        k2min = 1e-25/n/n
        bigBins = np.logspace(np.log10(k2min), np.log10(k2max), NBINS + 1)
        for iBig in range(NBINS):
            temp = np.linspace(bigBins[iBig], bigBins[iBig+1], 11)
            for i in range(11):
                k2_tab[n][10 * iBig + i] = temp[i]
        for ik in range(10 * NBINS + 1):
            g['up'][n][ik], g['dn'][n][ik] = populate_gnlk(nmax, n, np.sqrt(k2_tab[n][ik]))
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

def populate_beta(TM, Tr, nmax, k2_tab, g, NBINS):
    k2 = np.zeros(11)
    int_b = np.zeros(11)
    common_factor =  2.0/3.0 * alpha_fs**3 * EI/hPlanck
    beta = np.zeros((nmax+1,nmax))

    for n in range(1, nmax+1):
        n2 = n**2
        for l in range(n):
            for iBin in range(NBINS):
                for i in range(11):
                    ik = 10 * iBin + i
                    k2[i] = k2_tab[n][ik]
                    if (Tr < 1e-10):      #/* Flag meaning TR = 0 */
                        int_b[i] = 0.0
                    else:
                        int_b[i] = ((1.0 + k2[i]*n2)**3 / (np.exp(EI / Tr * (k2[i] + 1.0 / n2)) - 1.0) * 
                                      ((l + 1.0) * g['up'][n][ik][l] * g['up'][n][ik][l] 
                                       + l * g['dn'][n][ik][l] * g['dn'][n][ik][l]))
                beta[n][l] +=  Newton_Cotes_11pt(k2, int_b)
            beta[n][l]  *= common_factor / n2 / (2.0 * l + 1.0)
    return beta

# /********************************************************************************************* 
# Populating the recombination coefficients alpha(n, l, Tm, Tr) 
# Input: alpha[nmax+1][nmax], Tm, Tr in eV, nmax
# and the precomputed tables n2k2, g_up[nmax+1][Nkappa][nmax], g_dn[nmax+1][Nkappa][nmax], 
# where Nkappa = 10 * NBINS + 1
# *********************************************************************************************/


# /****************************************************************************************/

def populate_alpha(Tm, Tr, nmax, k2_tab, g, NBINS):
    k2 = np.zeros(11)
    int_a = np.zeros(11)   
    common_factor = (2.0/3.0 * alpha_fs**3 * EI/hPlanck 
                     * (hPlanck**2 * cLight**2/(2.0 * np.pi * mue * Tm))**1.5)
    alpha = np.zeros((nmax+1,nmax))
    
    for n in range(1,nmax+1):
        n2 = n**2
        for l in range(n):
            for iBin in range(NBINS):
                for i in range(11):
                    ik = 10 * iBin + i
                    k2[i] = k2_tab[n][ik]
                    int_a[i] = ((1.0 + n2 *k2[i])**3 * np.exp(-k2[i] * EI / Tm) *
                                 ((l + 1.0) * g['up'][n][ik][l] * g['up'][n][ik][l] 
                                  + l * g['dn'][n][ik][l] * g['dn'][n][ik][l]))
                    if Tr > 1e-10:
                        int_a[i] /= (1. - np.exp(-EI/Tr*(k2[i] + 1./n2)))
                alpha[n][l] += Newton_Cotes_11pt(k2, int_a)
            alpha[n][l] *= common_factor / n2
    return alpha
