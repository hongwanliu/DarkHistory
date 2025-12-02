"""Functions and classes for soft photon spectral distortion evolution."""

import numpy as np
import astropy.units as u
import astropy.constants as c
import math
from astropy.cosmology import Planck18 as cosmo

import darkhistory.physics as phys

# ===== Constants =====
m_e = (c.m_e * c.c ** 2).to(u.eV).value  # [eV] | astropy


# m_e = phys.me # [eV]


# ===== Functions =====
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
    T_CMB = phys.TCMB(1 + z)  # [eV]
    return x * T_CMB / T_M


def get_g_ff(xT_e, theta_e):
    """Gaunt factor in absence of Helium [dimensionless]. Eq after (A5)."""
    Z_H = 1.0

    # Clip to avoid log(0) and extremes
    xT_e_clip = np.clip(xT_e, 1e-8, None)
    theta_e_clip = np.clip(theta_e, 1e-6, None)

    arg = (np.sqrt(3) / np.pi) * (
            np.log(2.25 / (Z_H * xT_e_clip)) + 0.5 * np.log(theta_e_clip)
    ) + 1.425

    arg = np.clip(arg, -50.0, 50.0)

    return 1.0 + np.log1p(np.exp(arg))


def get_Lambda_BR(z, x, T_M, xe):
    """Bremsstrahlung emissivity coefficient [dimensionless]."""

    lambda_Compton_e = (c.h / (c.m_e * c.c)).to(u.cm)

    n_H = phys.nH * (1 + z) ** 3 * (1 / u.cm ** 3)
    #n_p = n_H * xHII
    n_e = n_H * xe

    xT_e = get_xT_e(z, x, T_M)
    theta_e = T_M / m_e

    prefactor = (
            c.alpha * lambda_Compton_e ** 3 * n_e
            / (2 * np.pi * np.sqrt(6 * np.pi))
    ).to(1).value

    return prefactor * theta_e ** (-7 / 2) * get_g_ff(xT_e, theta_e)


def get_Y(x):
    """Y factor for Y distortion [dimensionless]. Eq (8) in 2404.11743.

    Args:
        x (float or array): x = E/T_CMB.
    """
    ex = np.exp(x)
    return x * ex / (ex - 1) ** 2 * (x * (ex + 1) / (ex - 1) - 4)


def get_S_Y(z, x, T_M):
    """Y distortion source term [dimensionless]. Eq (8) in 2404.11743.
    Args:
        z (float): Redshift.
        x (float or array): x = E/T_CMB.
        T_M (float): Matter temperature at z in [eV].
    """
    T_CMB = phys.TCMB(1 + z)  # [eV]
    return (T_M - T_CMB) / m_e * get_Y(x)


def get_S_ff_bb(z, x, T_M,xe):
    """Free-free emission and absorption off blackbody photons [dimensionless].

    Args:
        z (float): Redshift.
        x (float or array): x = E/T_CMB.
        T_M (float): Matter temperature at z in [eV].
    """
    Lambda_BR = get_Lambda_BR(z, x, T_M, xe)
    xT_e = get_xT_e(z, x, T_M)
    return Lambda_BR * (1 - np.exp(-xT_e)) / xT_e ** 3 * (1 / (np.exp(xT_e) - 1) - 1 / (np.exp(x) - 1))


#===== Classes for spectrum and history =====

#SOFTPHOT_EDIT

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

    def drhoffdz(self, z, state=None):
        """Get the free-free a^-4 d(a^4 rho_ff)/dz [eV/pcm^3]. Eqs (14-15) in 2404.11743

        Args:
            z (float): Redshift.
            state (dict, optional): State of the universe at redshift z. If None, use default state.
        """
        xHII = state['xHII']
        xHeII = state['xHeII']
        xHeIII = state.get('xHeIII', 0.0)

        xe = xHII + xHeII + 2*xHeIII
        n_H = phys.nH * (1 + z)**3 * (1/u.cm**3)
        n_e = n_H * xe
        #prefactorEq14 = (- 1 / (3/2 * (n_H + n_He + n_e))).to(u.eV).value

        T_CMB = phys.TCMB(1 + z) * u.eV
        rho_CMB = (np.pi**2 / 15 * (T_CMB)**4 / (c.hbar**3 * c.c**3)).to(u.eV / u.cm**3)
        T_M = state['Tm'] * u.eV
        H = phys.hubble(1 + z) * u.s**-1
        prefactorEq15 = - rho_CMB / (np.pi**4/15) * (T_M/T_CMB)**3 * c.sigma_T.to(u.cm**2) * n_e * c.c.to(u.cm / u.s) / (H * (1 + z))

        Lambda_BR = get_Lambda_BR(z, self.x, T_M.value,xe)
        xT_e = get_xT_e(z, self.x, T_M.value)
        integrand = Lambda_BR * (1 - np.exp(-xT_e)) * (1/(np.exp(xT_e) - 1) - 1/(np.exp(self.x) - 1) - self.n)
        integral = np.trapz(integrand, self.x)

        drhoffdz = prefactorEq15.to(u.eV / u.cm**3).value * integral

        return drhoffdz

        

class SoftPhotonHistory:

    def __init__(self, init_spec=SoftPhotonSpectralDistortion(), injection=None):
        """
        Soft photon history class.
        
        Args:s
            init_spec (SoftPhotonSpectralDistortion): Initial soft photon distortion.
        """
        self.history = [init_spec]
        self.spec = init_spec
        self.injection = injection
        self._inj_cached = False
        self._inj_shape = None
        self._inj_A = None
        self.drhoffdz_arr = [] # tmp recorder

    def update(self, spec):
        self.history.append(spec)
        self.spec = spec

    def get_dndtau(self, z, T_M,xe):
        """Get the dN/dtau for the soft photon spectrum. Eq (7) in 2404.11743.

        Args:
            z (float): Redshift.
            T_M (float): Matter temperature at z in [eV].
        """
        x = self.spec.x
        xT_e = get_xT_e(z, x, T_M)
        Lambda_BR = get_Lambda_BR(z, x, T_M, xe)
        return - Lambda_BR * (1-np.exp(-xT_e)) / xT_e**3 * self.spec.n + get_S_Y(z, x, T_M) + get_S_ff_bb(z, x, T_M,xe)

    def S_inj(self, z, x, T_M, state):
        """
        Injection source term S_inj(x, τ) [dimensionless], dn/dτ.

        Implements eqs. (21–22) of Cyr 2404.11743 as a narrow top–hat in
        Thomson optical depth around z_inj, with total photon energy
        injection Δρ/ρ = rho_frac at z_inj.

        Args
        ----
        z (float): Redshift.
        x (ndarray): x = E / T_CMB grid.
        T_M (float): Matter temperature [eV]
        state (dict): Current TLA state.
        """

        if self.injection is None:
            return np.zeros_like(x)

        z_inj = self.injection['z_inj']
        dz_win = self.injection['dz_win']

        if abs(z - z_inj) > dz_win:
            return np.zeros_like(x)

        rho_frac = self.injection['rho_frac']
        x_cut = self.injection['x_cut']
        gamma = self.injection['gamma']

        if not self._inj_cached:

            # ----- eq. (21) -----
            shape = (x / x_cut) ** (-gamma) * np.exp(-x / x_cut)

            nz_win = 200
            zL, zR = z_inj - dz_win, z_inj + dz_win
            zs = np.linspace(zL, zR, nz_win)

            dtau_list = []
            for i in range(len(zs) - 1):
                z_mid = 0.5 * (zs[i] + zs[i + 1])
                rs_mid = 1.0 + z_mid

                xHII_mid = phys.xHII_std(rs_mid)
                xHeII_mid = phys.xHeII_std(rs_mid)
                xHeIII_mid = state.get('xHeIII', 0.0)

                xe_mid = xHII_mid + xHeII_mid + 2.0 * xHeIII_mid

                nH_mid = phys.nH * rs_mid ** 3 * (1.0 / u.cm ** 3)
                ne_mid = nH_mid * xe_mid

                dz_step = zs[i + 1] - zs[i]
                dt_mid = abs(phys.dtdz(rs_mid) * dz_step) * u.s

                dtau_mid = (
                        c.sigma_T.to(u.cm ** 2)
                        * ne_mid
                        * c.c.to(u.cm / u.s)
                        * dt_mid
                ).to(1).value

                dtau_list.append(dtau_mid)

            Delta_tau_window = np.sum(dtau_list)

            # ----- eq. (22) -----
            A_delta = ( rho_frac * (np.pi ** 4 / 15.0) / (x_cut ** 4 * math.gamma(4.0 - gamma)) )

            self._inj_shape = shape
            self._inj_A = A_delta / Delta_tau_window
            self._inj_cached = True

            print("Injecting at z = ", z_inj)

        return self._inj_A * self._inj_shape

    def step(self, z, dz, state):
        rs = state['rs']
        Tm = state['Tm']
        xHII = state['xHII']
        xHeII = state['xHeII']
        xHeIII = state.get('xHeIII', 0.0)

        nH = phys.nH * rs ** 3 * 1 / u.cm**3
        xe = xHII + xHeII + 2*xHeIII
        ne = xe * nH

        TCMB = phys.TCMB(rs)
        x = self.spec.x
        n = self.spec.n.copy()

        xe_photon = x * (TCMB / Tm)

        dt = phys.dtdz(rs) * np.abs(dz) * u.s
        dtau = (c.sigma_T.to(u.cm**2) * ne * c.c.to(u.cm / u.s) * abs(dt)).to(1).value

        Lambda = get_Lambda_BR(z, x, Tm, xe)

        # Cheatsheet Eq. (7)
        Delta_tau_ff = Lambda * (1 - np.exp(-xe_photon)) / xe_photon ** 3 * dtau

        # --- Source term: S_ff + S_Y + S_inj, Eq. (4) ---
        DeltaS = (
                get_S_ff_bb(z, x, Tm, xe)
                + get_S_Y(z, x, Tm)
                + self.S_inj(z, x, Tm, state)  # <--- NEW
        )

        # Cheatsheet Eq. (6)
        S_tilde = xe_photon ** 3 * DeltaS / (Lambda * (1 - np.exp(-xe_photon)) + 1e-30)

        # Cheatsheet Eq. (5)
        exp_term = np.exp(-Delta_tau_ff)
        n_new = n * exp_term + S_tilde * (1 - exp_term)

        new_spec = self.spec.copy()
        new_spec.n = n_new
        new_spec.z = z
        new_spec.tau = self.spec.tau + dtau

        self.update(new_spec)








