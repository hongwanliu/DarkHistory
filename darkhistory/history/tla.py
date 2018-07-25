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
	dm_injection_rate, rs_vec, reion_switch=True
):
	"""Returns the ionization and thermal history of the IGM.

	Parameters
	----------
	init_cond : array
		Array containing [initial temperature, initial xHII, initial xHeII, initial xHeIII].
	fz_H_ion : function
		f(rs, x_HI, x_HeI, x_HeII) for hydrogen ionization.
	fz_H_exc : function
		f(rs, x_HI, x_HeI, x_HeII) for hydrogen Lyman-alpha excitation.
	f_heating : function
		f(rs, x_HI, x_HeI, x_HeII) for heating.
	dm_injection_rate : function
		Injection rate of DM as a function of redshift.
	rs_vec : ndarray
		Abscissa for the solution.
	reion_switch : bool
		Reionization model included if true.

	Returns
	-------
	list of ndarray
		[temperature solution (in eV), xHII solution, xHeII, xHeIII].

	Note
	----
	The actual differential equation that we solve is expressed in terms of y = arctanh(f*(x - f)), where f = 0.5 for x = xHII, and f = nHe/nH * 0.5 for x = xHeII or xHeIII, where nHe/nH is approximately 0.083.

	"""

	chi = phys.nHe/phys.nH

	photoion_rate_HI   = reion.photoion_rate('HI')
	photoion_rate_HeI  = reion.photoion_rate('HeI')
	photoion_rate_HeII = reion.photoion_rate('HeII')

	photoheat_rate_HI   = reion.photoheat_rate('HI')
	photoheat_rate_HeI  = reion.photoheat_rate('HeI')
	photoheat_rate_HeII = reion.photoheat_rate('HeII')

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
				- peebles_C(xHII(yHII), rs) * (
					alpha_recomb(T_m) * xHII(yHII)*xe * phys.nH * rs**3
					- beta_ion(phys.TCMB(rs)) * xHI
						* np.exp(-phys.lya_eng/T_m)
				)
				# DM injection. Note that C = 1 at late times.
				+ f_H_ion(rs, xHI, xHeI, xHeII(yHeII)) * dm_injection_rate(rs)
					/ (phys.rydberg * phys.nH * rs**3)
				+ (1 - peebles_C(xHII(yHII), rs)) * (
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
				+ f_H_ion(rs, xHI, xHeI, xHeII(yHeII)) * dm_injection_rate(rs)
					/ (phys.rydberg * phys.nH * rs**3)
				+ (1 - peebles_C(xHII(yHII), rs)) * (
					f_H_exc(rs, xHI, xHeI, xHeII(yHeII)) * dm_injection_rate(rs)
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

	rs_reion = rs_vec[rs_vec <= 16.1]
	rs_before_reion = rs_vec[rs_vec > 16.1]
	rs_before_reion = np.append(rs_before_reion, rs_reion[0])

	if np.size(rs_before_reion) > 1 and reion_switch:
		soln_before_reion = odeint(
			tla_before_reion, init_cond, rs_before_reion, mxstep = 500
		)

		init_cond_reion = [
			soln_before_reion[-1,0],
			soln_before_reion[-1,1],
			np.arctanh(2/(chi)*(1e-12 - chi/2)),
			np.arctanh(2/(chi)*(1e-12 - chi/2))
		]

		soln_reion = odeint(
			tla_reion, init_cond_reion, rs_reion, mxstep = 1000,
		)

		soln = np.vstack((soln_before_reion[:-1,:], soln_reion))

	elif np.size(rs_before_reion) <= 1 and reion_switch:
		soln = odeint(
			tla_reion, init_cond, rs_reion, mxstep = 1000,
		)

	else:
		soln = odeint(
			tla_before_reion, init_cond, rs_vec, mxstep = 500
		)

	soln[:,1] = 0.5 + 0.5*np.tanh(soln[:,1])
	soln[:,2] = (
		chi/2 + chi/2*np.tanh(soln[:,2])
	)
	soln[:,3] = (
		chi/2 + chi/2*np.tanh(soln[:,3])
	)

	return soln
