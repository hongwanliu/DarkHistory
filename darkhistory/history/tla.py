import numpy as np 
from darkhistory import physics as phys
from scipy.integrate import odeint

def compton_cooling_rate(xe, T_matter, rs):
	"""Returns the Compton cooling rate. 

	Parameters
	----------
	xe : float
		The ionization fraction ne/nH. 
	Tm : float
		The matter temperature. 
	rs : float
		The redshift in 1+z. 

	Returns
	-------
	float
		The Compton cooling rate in eV/s. 
	"""
	return (
		xe / (1 + xe + phys.nHe/phys.nH) * (phys.TCMB(rs) - T_matter)
		* 32 * phys.thomson_xsec * phys.stefboltz
		* phys.TCMB(rs)**4 / (3 * phys.me)
	)

def alpha_recomb(T_matter):
	"""Case-B recombination coefficient. 

	Parameters
	----------
	T_matter : float
		The matter temperature. 

	Returns
	-------
	float
		Case-B recombination coefficient in cm^3/s. 
	"""

	# Fudge factor recommended in 1011.3758
	fudge_fac = 1.126 
	
	return (
		fudge_fac * 1e-13 * 4.309 * (1.16405*T_matter)**(-0.6166)
		/ (1 + 0.6703 * (1.16405*T_matter)**0.5300)
	)

def beta_ion(T_rad):
	"""Case-B photoionization coefficient. 

	Parameters
	----------
	T_rad : float
		The radiation temperature. 

	Returns
	-------
	float
		Case-B photoionization coefficient in s^-1. 

	"""
	reduced_mass = phys.mp*phys.me/(phys.mp + phys.me)
	de_broglie_wavelength = (
		phys.c * 2*np.pi*phys.hbar
		/ np.sqrt(2 * np.pi * reduced_mass * T_rad)
	)
	return (
		(1/de_broglie_wavelength)**3/4 
		* np.exp(-phys.rydberg/4/T_rad) * alpha_recomb(T_rad)
	)


# def betae(Tr):
# 	# Case-B photoionization coefficient
# 	thermlambda = c*(2*pi*hbar)/sqrt(2*pi*(mp*me/(me+mp))*Tr)
# 	return alphae(Tr) * exp(-(rydberg/4)/Tr)/(thermlambda**3) 

def peebles_C(xe, rs):
	"""Returns the Peebles C coefficient. 

	This is the ratio of the total rate for transitions from n = 2 to the ground state to the total rate of all transitions, including ionization.

	Parameters
	----------
	xe : float
		The ionization fraction ne/nH. 
	Tm : float
		The matter temperature. 
	rs : float
		The redshift in 1+z. 

	Returns
	-------
	float
		The Peebles C factor. 
	"""

	# Net rate for 2p to 1s transition. 
	rate_2p1s = (
		8 * np.pi * phys.hubble(rs)
		/(3*(phys.nH*rs**3 * (1-xe) * (phys.c/phys.lya_freq)**3))
	)

	# Net rate for 2s to 1s transition. 
	rate_2s1s = phys.width_2s1s

	# Net rate for ionization. 
	rate_ion = beta_ion(phys.TCMB(rs))

	# Rate is averaged over 3/4 of excited state being in 2p, 1/4 in 2s. 
	return (
		(3*rate_2p1s/4 + rate_2s1s/4)
		/(3*rate_2p1s/4 + rate_2s1s/4 + rate_ion)
	)

def get_history(
	init_cond, f_H_ion, f_H_exc, f_heating, 
	dm_injection_rate, rs_vec
):
	"""Returns the ionization and thermal history of the IGM. 

	Parameters
	----------
	init_cond : array
		Array containing [initial temperature, initial xe]
	fz_H_ion : function
		f(xe, rs) for hydrogen ionization. 
	fz_H_exc : function
		f(xe, rs) for hydrogen Lyman-alpha excitation. 
	f_heating : function
		f(xe, rs) for heating. 
	dm_injection_rate : function
		Injection rate of DM as a function of redshift. 
	rs_vec : ndarray
		Abscissa for the solution. 

	Returns
	-------
	list of ndarray
		[temperature solution (in eV), xe solution]. 

	Note
	----
	The actual differential equation that we solve is expressed in terms of y = arctanh(2*(xe - 0.5)). 

	"""

	def tla_diff_eq(var, rs):
		# Returns an array of values for [dT/dz, dy/dz].
		# var is the [temperature, xe] inputs. 

		def xe(y):
			return 0.5 + 0.5*np.tanh(y)

		def dT_dz(T_matter, y, rs):
			return (
				2*T_matter/rs - phys.dtdz(rs) * (
					compton_cooling_rate(xe(y), T_matter, rs)
					+ (
						1/(1 + xe(y) + phys.nHe/phys.nH)
						* 2/(3 * phys.nH * rs**3)
						* f_heating(rs, xe(y))
						* dm_injection_rate(rs)
					)
				)
			)

		def dy_dz(T_matter, y, rs):
			return (
				2 * np.cosh(y)**2 * phys.dtdz(rs) * (
					peebles_C(xe(y), rs) * (
						alpha_recomb(T_matter) * xe(y)**2 * phys.nH * rs**3
						- (
							beta_ion(phys.TCMB(rs)) * (1 - xe(y))
							* np.exp(-phys.lya_eng/T_matter)
						)
					)
					- (
						f_H_ion(rs, xe(y)) * dm_injection_rate(rs)
						/ (phys.rydberg * phys.nH * rs**3)
					)
					- (1 - peebles_C(xe(y), rs)) * (
						f_H_exc(rs, xe(y)) * dm_injection_rate(rs) 
						/ (phys.lya_eng * phys.nH * rs**3)
					)
				)
			)

		T_matter, y = var[0], var[1]

		return [dT_dz(T_matter, y, rs), dy_dz(T_matter, y, rs)]

	if init_cond[1] == 1:
		init_cond[1] = 1 - 1e-12
	else:
		init_cond[1] = np.arctanh(2*(init_cond[1] - 0.5))

	soln = odeint(tla_diff_eq, init_cond, rs_vec, mxstep = 500)

	soln[:,1] = 0.5 + 0.5*np.tanh(soln[:,1])

	return soln