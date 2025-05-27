"""Functions and classes for soft photon spectral distortion evolution."""

import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import Planck18 as cosmo

import darkhistory.physics as phys


#===== Constants =====
m_e = (c.m_e * c.c**2).to(u.eV).value # [eV] | astropy
# m_e = phys.me # [eV]


#===== Functions =====
# Variable order: z, x/xT_e, T_M/theta_e
# Note: x = E/T_CMB, xT_e = x * T_CMB/T_M, theta_e = T_M/m_e.
# we distinguish between x_e (n_e/n_H) and xT_e.

def get_xT_e(z, x, T_M):
    """xT_e = x * T_CMB/T_M, where x = E/T_CMB. [dimensionless]
    
    Args:
        z (float): Redshift.
        x (float or array): x = E/T_CMB.
        T_M (float): Matter temperature in [eV].
    """
    T_CMB = phys.TCMB(1 + z) # [eV]
    return x * T_CMB / T_M

def get_g_ff(xT_e, theta_e):
    """Gaunt factor in absence of Helium [dimensionless]. Eq after (A5) in 2404.11743.
    
    Args:
        xT_e (float or array): x * T_CMB/T_M [dimensionless].
        theta_e (float): Reduced matter temperature [dimensionless].
    """
    Z_H = 1 # charge of hydrogen
    return 1 + np.log(1 + np.exp(np.sqrt(3) / np.pi * (np.log(2.25/(Z_H*xT_e)) + np.log(theta_e)/2) + 1.425))

def get_Lambda_BR(z, x, T_M):
    """Bremsstrahlung emissivity coefficient [dimensionless]. Eq (A7) in 2404.11743.
    
    Args:
        z (float): Redshift.
        xT_e (float or array): x * T_CMB/T_M.
        T_M (float): Matter temperature at z in [eV].
    """
    lambda_Compton_e = c.h / (c.m_e * c.c)
    Np = phys.nH * (1 + z)**3 * (1/u.cm**3)
    xT_e = get_xT_e(z, x, T_M)
    theta_e = T_M / m_e
    prefactor = (c.alpha * lambda_Compton_e**3 * Np / (2*np.pi * np.sqrt(6*np.pi))).to(1).value
    return prefactor * theta_e**(-7/2) * get_g_ff(xT_e, theta_e)

def get_Y(x):
    """Y factor for Y distortion [dimensionless]. Eq (8) in 2404.11743.
    
    Args:
        x (float or array): x = E/T_CMB.
    """
    ex = np.exp(x)
    return x * ex / (ex - 1)**2 * (x * (ex + 1) / (ex - 1) - 4)

def get_S_Y(z, x, T_M):
    """Y distortion source term [dimensionless]. Eq (8) in 2404.11743.
    Args:
        z (float): Redshift.
        x (float or array): x = E/T_CMB.
        T_M (float): Matter temperature at z in [eV].
    """
    T_CMB = phys.TCMB(1 + z) # [eV]
    return (T_M - T_CMB) / m_e * get_Y(x)
    
def get_S_ff_bb(z, x, T_M):
    """Free-free emission and absorption off blackbody photons [dimensionless].
    
    Args:
        z (float): Redshift.
        x (float or array): x = E/T_CMB.
        T_M (float): Matter temperature at z in [eV].
    """
    Lambda_BR = get_Lambda_BR(z, x, T_M)
    xT_e = get_xT_e(z, x, T_M)
    return Lambda_BR * (1-np.exp(-xT_e)) / xT_e**3 * (1/(np.exp(xT_e)-1) - 1/(np.exp(x)-1))


#===== Classes for spectrum and history =====

X_MIN_SOFTPHOT = 1e-8 # [dimensionless]
X_MAX_SOFTPHOT = 1e+2 # [dimensionless]
N_X_BINS = 5000
x_edges_default = np.geomspace(X_MIN_SOFTPHOT, X_MAX_SOFTPHOT, N_X_BINS+1)


class SoftPhotonSpectralDistortion:

    def __init__(
            self,
            x_edges = x_edges_default,
            n = None,
            z = None,
            tau = 0.
        ):
        """
        Soft photon spectral distortion class.
        
        Args:
            x_edges (1D array): Bin edges for the x values. (x = E/T_CMB)
            n (1D array, optional): Phase space density of photons.
            z (float, optional): Redshift.
            tau (float, optional): Optical depth (integrated from early to late times).
        """

        self.x_edges = x_edges
        self.x = (self.x_edges[:-1] + self.x_edges[1:]) / 2
        self.dx = self.x_edges[1:] - self.x_edges[:-1]
        if n is not None:
            self.n = n
        else:
            self.n = np.zeros_like(self.x)
        self.z = z
        self.tau = tau

    def from_point_inj(self, x_cut, gamma, z, rho_frac):
        """
        Initialize the soft photon spectrum from a point injection of form A (x/x_cut)^(-gamma) * exp(-x/x_cut).
        
        Args:
            x_cut (float): Power law cutoff energy E_cut/T_CMB.
            gamma (float): Inverse of power law index for the soft photon spectrum.
            z (float): Redshift of injection.
            rho_frac (float): Fraction of the CMB energy density injected as soft photons.
        """
        self.n = (self.x / x_cut)**(-gamma) * np.exp(-self.x / x_cut) # unnormalized
        T_CMB = phys.TCMB(1 + z) * u.eV
        rho_CMB = (np.pi**2 / 15 * (T_CMB)**4 / (c.hbar**3 * c.c**3)).to(u.eV / u.cm**3)
        rho_target = rho_frac * rho_CMB
        rho_unnorm = self.Etot(z)
        self.n *= (rho_target / rho_unnorm).to(1).value
        self.z = z
        self.tau = 0 # Clear the tau value
        
    def copy(self):
        """Return a copy of the SoftPhotonSpectralDistortion object."""
        return SoftPhotonSpectralDistortion(x_edges=self.x_edges, n=self.n.copy(), z=self.z, tau=self.tau)
    
    def E(self, z):
        """Energy range corresponding to the x values [u.eV]."""
        T_CMB = phys.TCMB(1 + z) * u.eV
        return self.x * T_CMB

    def dNdx(self, z):
        """Physical density (differential against dx) of photons [1/u.cm^3]."""
        T_CMB = phys.TCMB(1 + z) * u.eV
        return (1 / (np.pi**2) * (T_CMB / (c.hbar * c.c))**3).to(1/u.cm**3) * self.x**2 * self.n
    
    def dNdE(self, z):
        """Physical density (differential against dE) of photons [1/(u.cm^3 u.eV)]."""
        T_CMB = phys.TCMB(1 + z) * u.eV
        return self.dNdx(z) / T_CMB
    
    def Etot(self, z):
        """Total energy density of the distortion [u.eV/u.cm^3]."""
        EdNdx = self.E(z) * self.dNdx(z)
        return np.sum(EdNdx * self.dx)
    
    def dTffdz(self, z, state=None):
        """Get the free-free dT_ff/dz [eV]. Eqs (14-15) in 2404.11743.

        Args:
            z (float): Redshift.
            state (dict, optional): State of the universe at redshift z. If None, use default state.
        """
        n_H = phys.nH * (1 + z)**3 * (1/u.cm**3)
        n_He = phys.nHe * (1 + z)**3 * (1/u.cm**3)
        n_e = n_H * (state['xHII'] + state['xHeII'])
        prefactorEq14 = - 1 / (3/2 * (n_H + n_He + n_e))

        T_CMB = phys.TCMB(1 + z) * u.eV
        rho_CMB = (np.pi**2 / 15 * (T_CMB)**4 / (c.hbar**3 * c.c**3)).to(u.eV / u.cm**3)
        T_M = state['Tm'] * u.eV
        H = phys.hubble(z) * u.s**-1
        prefactorEq15 = - rho_CMB / (np.pi**4/15) * (T_M/T_CMB)**3 * c.sigma_T * n_e * c.c / (H * (1 + z))

        Lambda_BR = get_Lambda_BR(z, self.x, T_M.value)
        xT_e = get_xT_e(z, self.x, T_M.value)
        integrand = Lambda_BR * (1 - np.exp(-xT_e)) * (1/(np.exp(xT_e) - 1) - 1/(np.exp(self.x) - 1) - self.n)
        integral = np.trapz(integrand, self.x)

        return (prefactorEq14 * prefactorEq15).to(u.eV).value * integral
        

class SoftPhotonHistory:

    def __init__(self, init_spec=SoftPhotonSpectralDistortion()):
        """
        Soft photon history class.
        
        Args:
            init_spec (SoftPhotonSpectralDistortion): Initial soft photon distortion.
        """
        self.history = [init_spec]
        self.spec = init_spec
        self.dTffdz_arr = [0.] # tmp recorder

    def update(self, spec):
        self.history.append(spec)
        self.spec = spec

    def get_dndtau(self, z, T_M):
        """Get the dN/dtau for the soft photon spectrum. Eq (7) in 2404.11743.

        Args:
            z (float): Redshift.
            T_M (float): Matter temperature at z in [eV].
        """
        x = self.spec.x
        xT_e = get_xT_e(z, x, T_M)
        Lambda_BR = get_Lambda_BR(z, x, T_M)
        return - Lambda_BR * (1-np.exp(-xT_e)) / xT_e**3 * self.spec.n + get_S_Y(z, x, T_M) + get_S_ff_bb(z, x, T_M)
    
    def step(self, z, dz, state):
        """Step the soft photon spectrum forward in time.
        
        Args:
            z (float): Redshift.
            dz (float): Change in redshift.
            state (dict): State of the universe at redshift z.
        """
        
        n_H = phys.nH * (1 + z)**3 * (1/u.cm**3)
        n_e = n_H * (state['xHII'] + state['xHeII'])
        H_z = phys.hubble(z) * u.s**-1
        dtau = (c.sigma_T * c.c * np.abs(dz) * n_e / ((1+z) * H_z)).to(1).value

        # print(f"z={z:.3f} dz={dz:.6f}, dtau={dtau:.6e}, x_e={(state['xHII'] + state['xHeII']):.6f}")
        dn = self.get_dndtau(z, state['Tm']) * dtau

        new_spec = self.spec.copy()
        new_spec.n += dn
        new_spec.z = z
        new_spec.tau += dtau
        self.update(new_spec)
