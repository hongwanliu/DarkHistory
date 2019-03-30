"""Atomic cooling cross sections.
"""
import numpy
import sys
import darkhistory.physics as physics
from scipy.interpolate import interp1d
import math
import scipy.constants as p
sys.path.append("../..")
from darkhistory.spec.spectrum import *
import darkhistory.spec.spectools as spectools


def thermalize_cs(T, f=0.05, lnV=10):
    '''
    Calculates the heating cross section (xsec) at a particular kinetic energy.

    Parameters
    ----------
    T : float, ndarray
        The electron's initial kinetic energy.
    f : float
        The fraction of energy lost in each interaction.
    lnV : float
        The Coulomb logarithm [ln(Lambda)]

    Returns
    ----------
    float, ndarray (same as T)
        The cross section for heating at energy T
        (given in cm^2).

    See Also
    --------
    heating_dE : Preferred; finds dE/dt
    '''


    sigma_ee=(7.82*10**(-11))*(0.05/f)*lnV*(T)**(-2)
    return sigma_ee

def heating_dE(T, x_e, rs, nH=physics.nH):
    '''
    Calculates the heating loss rate for electrons at a particular
    kinetic energy given some ionization fraction and redshift.

    Parameters
    ----------
    T : float, ndarray
        The electron's initial kinetic energy
    x_e : float
        The ionization fraction for Hydrogen.
    rs : float
        The redshift (1+z) during heating; used for n_e.
    nH : float
        Hydrogen density from physics.py

    Returns
    ----------
    float, ndarray (same as T)
        The energy loss rate from heating (negative).

    '''

    lnV=10
    n_e = x_e*nH*rs**3  #cm^-3
    # from x_e=n_e/n_h/rs^3

    e_charge=4.80326*10**-10 #esu units
    mv=((T**2+2*T*physics.me)**0.5*physics.me/(T+physics.me))
    numfac=(10**-14*physics.ele**-2*physics.c)

    dE_dt = numfac*(-4*math.pi*(e_charge)**4*n_e*lnV)/mv
    

    return dE_dt

def ionize_cs(Energy, atoms):

    '''
    Calculates the ionization cross section (xsec) for electrons
    impacting one of (H, He, He+) at a particular kinetic energy.

    Parameters
    ----------
    Energy : ndarray
        Each electron's initial kinetic energy.
    atoms : ndarray (same size as Energy)
        Indicates a xsec corresponding to each
        element of Energy (1=H, 2=He, 3=He+)

    Returns
    ----------
    ndarray
        The cross section for ionization for
        each pair (Energy[n],atoms[n])
        (given in cm^2).

    See Also
    --------
    ionize_s_cs : Often preferred; gives singly differential xsec
    '''
    #initialize return variable
    sigma = numpy.zeros(len(atoms))

    for n in range(len(Energy)):
        T=Energy[n]
        atom=atoms[n]

        if atom==1: #H
            B=13.6057 #eV: binding energy
            U=13.6057 #eV:
            t=T/B
            D= (2834163/10+2*(-4536259-10736505*t - 7512905*t**(2) + 112365*t**(3))/(5*(1+t)**(5)))/1000000
            N=1 # number of bound electrons in subshell
            N_i= 0.4343 #integral of df/dw from 0 to infinity
        elif atom==2: #He
            B=24.59 #eV
            U=39.51 #eV
            t=T/B
            D= 1/2*(53047/60-(4*(-58971+227814*t-78435*t**2+121780*t**3))/(15*(1+t)**6))/1000
            N=2
            N_i=1.605
        elif atom==3: #He+
            B=13.6057*4 #eV: scaled by Z^2
            U=13.6057*4 #eV: scaled by Z^2
            t=T/B
            D= (2834163/10+2*(-4536259-10736505*t - 7512905*t**(2) + 112365*t**(3))/(5*(1+t)**(5)))/1000000 #same as H
            N=1
            N_i=0.4343 #seems same as H in approx
        else:
            print('error: some atom incorrectly specified')
            return
        u=U/B
        S=4*math.pi*p.value('Bohr radius')**2*N*(13.6057/B)**2 #m^2

        sigma_i=S/(t+u+1)*(D*numpy.log(t)+(2-N_i/N)*((t-1)/t-numpy.log(t)/(t+1)))*(10**4) #cm^2
        #replace negatives with zero
        if sigma_i<0:
            sigma_i=0

        sigma[n]=sigma_i

    return sigma

def ionize_s_cs(E_in, E_sec, atoms):

    '''
    Calculates the singly-differential ionization cross section (xsec)
    for electrons impacting one of (H, He, He+) at a particular
    kinetic energy of the incident and one secondary electron.

    Parameters
    ----------
    E_in : ndarray
        Each electron's initial kinetic energy (eV).
    E_out : ndarray
        The energy of one secondary electron for each initial electron (eV).
    atoms : ndarray
        Atomic xsec relevant to each ionization; (1=H, 2=He, 3=He+)

    Returns
    ----------
    ndarray
        The cross section for ionization at each incident energy (E_in[n]),
        secondary energy (E_out[n]), and atomic xsec (atoms[n]); (given in cm^2/eV).

    See Also
    --------
    ionize_cs : Gives singly-differential ionization xsec (cm^2/eV)
    '''

    #initialize return variable
    sigma=numpy.zeros(len(atoms))

    for n in range(len(atoms)):
        T=E_in[n]
        W=E_sec[n]
        atom=atoms[n]

        if atom==1: #H
            B=13.6057 #eV: binding energy
            U=13.6057 #eV:
            t=T/B
            w=W/B
            y=1/(w+1)
            df_dw=-0.022473*y**2+1.1775*y**3-0.46264*y**4+0.089064*y**5
            N=1 # number of bound electrons in subshell
            N_i= 0.4343 #integral of df/dw from 0 to infinity
        elif atom==2: #He
            B=24.59 #eV
            U=39.51 #eV
            t=T/B
            w=W/B
            y=1/(w+1)
            df_dw=12.178*y**3-29.585*y**4+31.251*y**5-12.175*y**6 #original expression
            #df_dw=8.24012*y**3-10.4769*y**4+3.96496*y**5-0.0445976*y**6
            N=2
            N_i=1.605 #original corresponding
            #N_i=1.610
        elif atom==3: #He+
            B=13.6057*4 #eV: scaled by Z^2
            U=13.6057*4 #eV: scaled by Z^2
            t=T/B
            w=W/B
            y=1/(w+1)
            df_dw=-0.022473*y**2+1.1775*y**3-0.46264*y**4+0.089064*y**5
            N=1
            N_i=0.4343 #seems same as H in approx
        else:
            print('error: atom incorrectly specified')
            return

        u=U/B
        S=4*math.pi*p.value('Bohr radius')**2*N*(13.6057/B)**2*10**(4) #cm^2

        sigma_i=S/(B*(t+(u+1)))*((N_i/N-2)/(t+1)*(1/(w+1)+1/(t-w))+(2-N_i/N)*(1/(w+1)**2+1/(t-w)**2)+ \
                               numpy.log(t)/(N*(w+1))*df_dw) #cm^2/eV
        
        #replace negatives with zero
        if sigma_i < 0:
            sigma_i=0

        sigma[n]=sigma_i

    return sigma


def ionize_s_cs_H(E_in, E_sec):
    
    '''
    Calculates the integrated, singly-differential ionization cross section (xsec) 
    for electrons impacting H at a particular 
    kinetic energy of the incident over a log-space vector of secondary energies. 
    
    Parameters
    ----------
    E_in : float
        Primary electron's initial kinetic energy (eV).
    E_sec : ndarray ([Spectrum].eng)
        The energy of one secondary electron for each initial electron (eV).

    Returns
    ----------
    ndarray
        The cross section for ionization integrated over a log-bin centered at each
        secondary energy (E_sec[n]); (given in cm^2). 
    
    See Also
    --------
    ionize_cs : Gives total ionization xsec
    ionize_s_cs: Gives individual values for multi-atom singly-diff xsecs
    '''
    
    #initialize return variable
    sigma = numpy.zeros(len(E_sec))
    
    #get bin boundaries for integration
    edges = get_bin_bound(E_sec)
    
    def integrand(W): #W=E_sec[n]
        T=E_in 
        B=13.6057 #eV: binding energy
        U=13.6057 #eV: 
        t=T/B
        w=W/B
        y=1/(w+1)
        df_dw=-0.022473*y**2+1.1775*y**3-0.46264*y**4+0.089064*y**5
        N=1 # number of bound electrons in subshell
        N_i= 0.4343 #integral of df/dw from 0 to infinity
        u=U/B 
        S=4*math.pi*p.value('Bohr radius')**2*N*(13.6057/B)**2 #m^2

        return S/(B*(t+(u+1)))*((N_i/N-2)/(t+1)*(1/(w+1)+1/(t-w))*(2-N_i/N)*(1/(w+1)**2+1/(t-w)**2)+ numpy.log(t)/(N*(w+1))*df_dw) #cm^2/eV

    #generates vector for summation
    integrand_vec = np.zeros(len(E_sec))
    for i in range(len(E_sec)):
        integrand_vec[i]=integrand(E_sec[i])
    
    #performs summation
    for n in range(len(E_sec)-1):
        sigma[n] = integrate.trapz(integrand_vec[n:n+2], E_sec[n:n+2])
    
    #integrates continuously
    #for n in range(len(E_sec)):
        #sigma[n] = integrate.quad(integrand, edges[n], edges[n+1])[0]
        
    sigma = sigma.clip(min=0)
    return sigma


def ionize_s_cs_H_2(E_in, E_sec):
    
    '''
    Calculates the integrated, singly-differential ionization cross section (xsec) 
    for electrons impacting H at a particular 
    kinetic energy of the incident over a log-space vector of secondary energies. 
    
    Parameters
    ----------
    E_in : float
        Primary electron's initial kinetic energy (eV).
    E_sec : ndarray ([Spectrum].eng)
        The energy of one secondary electron for each initial electron (eV).

    Returns
    ----------
    sigma : ndarray
        The cross section for ionization integrated over a log-bin centered at each
        secondary energy (E_sec[n]); (given in cm^2). 
    
    See Also
    --------
    ionize_cs : Gives total ionization xsec
    ionize_s_cs: Gives individual values for multi-atom singly-diff xsecs
    '''
    
    #initialize return variable
    sigma = numpy.zeros(len(E_sec))
    
    #get bin boundaries for integration
    edges = spectools.get_bin_bound(E_sec)
    
    def integrand(W): #W=E_sec[n]
        T=E_in 
        B=13.6057 #eV: binding energy
        U=13.6057 #eV: 
        t=T/B
        w=W/B
        y=1/(w+1)
        df_dw=-0.022473*y**2+1.1775*y**3-0.46264*y**4+0.089064*y**5
        N=1 # number of bound electrons in subshell
        N_i= 0.4343 #integral of df/dw from 0 to infinity
        u=U/B 
        S=4*math.pi*p.value('Bohr radius')**2*N*(13.6057/B)**2 #m^2

        return S/(B*(t+(u+1)))*((N_i/N-2)/(t+1)*(1/(w+1)+1/(t-w))*(2-N_i/N)*(1/(w+1)**2+1/(t-w)**2)+ numpy.log(t)/(N*(w+1))*df_dw) #cm^2/eV

    #generates vector for summation
    integrand_vec = np.zeros(len(E_sec))
    
    #change begins
    B=13.6057
    f_max = (E_in - B)/2
    for i,E_s in enumerate(E_sec):
        if E_s>E_in-B:
            integrand_vec[i]=0
        elif E_s > f_max:
            f_delta = E_s-f_max
            integrand_vec[i]=integrand(E_s-2*f_delta)  
        else:
            integrand_vec[i]=integrand(E_s)
    
    #change ends
    
    #performs summation
    for n in range(len(E_sec)-1):
        sigma[n] = integrate.trapz(integrand_vec[n:n+2], E_sec[n:n+2])
    
    #integrates continuously
    #for n in range(len(E_sec)):
        #sigma[n] = integrate.quad(integrand, edges[n], edges[n+1])[0]
        
    sigma = sigma.clip(min=0)
    return sigma



def ionize_s_cs_He(E_in, E_sec):
    
    '''
    Calculates the integrated, singly-differential ionization cross section (xsec) 
    for electrons impacting He at a particular 
    kinetic energy of the incident over a log-space vecotr of secondary energies. 
    
    Parameters
    ----------
    E_in : float
        Primary electron's initial kinetic energy (eV).
    E_sec : ndarray ([Spectrum].eng)
        The energy of one secondary electron for each initial electron (eV).

    Returns
    ----------
    ndarray
        The cross section for ionization integrated over a log-bin centered at each
        secondary energy (E_sec[n]); (given in cm^2). 
    
    See Also
    --------
    ionize_cs : Gives total ionization xsec
    ionize_s_cs: Gives individual values for multi-atom singly-diff xsecs
    '''
    
    #initialize return variable
    sigma = numpy.zeros(len(E_sec))
    
    #get bin boundaries for integration
    edges = get_bin_bound(E_sec)
    
    def integrand(W): #W=E_sec[n]
        T=E_in 
        B=24.59 #eV
        U=39.51 #eV
        t=T/B
        w=W/B
        y=1/(w+1)
        df_dw=12.178*y**3-29.585*y**4+31.251*y**5-12.175*y**6
        N=2
        N_i=1.605
        u=U/B 
        S=4*math.pi*p.value('Bohr radius')**2*N*(13.6057/B)**2 #m^2

        return S/(B*(t+(u+1)))*((N_i/N-2)/(t+1)*(1/(w+1)+1/(t-w))*(2-N_i/N)*(1/(w+1)**2+1/(t-w)**2)+ numpy.log(t)/(N*(w+1))*df_dw) #cm^2/eV

    for n in range(len(E_sec)):
        sigma[n] = integrate.quad(integrand, edges[n], edges[n+1])[0]
        
    sigma = sigma.clip(min=0)
    return sigma