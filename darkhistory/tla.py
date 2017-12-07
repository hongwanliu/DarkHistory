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
	
	return (
		1e-13 * 4.309 * (1.16405*T_matter)**(-0.6166)
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
		/(3*rate_2p_1s/4 + rate_2s1s/4 + rate_ion)
	)





# def comptonCMB(xe, Tm, rs): 
# 	# Compton cooling rate
# 	return (xe/(1 + xe + nHe/nH)) * (TCMB(rs) - Tm)*32*thomsonXSec*stefBoltz*TCMB(rs)**4/(3*me)

# def KLyman(rs, omegaM=omegaM, omegaRad=omegaRad, omegaLambda=omegaLambda): 
# 	# Rate at which Lya-photons cross the line
# 	return (c/lyaFreq)**3/(8*pi*hubblerates(rs, H0, omegaM, omegaRad, omegaLambda))

# def alphae(Tm): 
# 	# Case-B recombination coefficient
# 	return 1e-13 * (4.309 * (1.16405*Tm)**(-0.6166))/(1 + 0.6703*(1.16405*Tm)**0.5300)

# def betae(Tr):
# 	# Case-B photoionization coefficient
# 	thermlambda = c*(2*pi*hbar)/sqrt(2*pi*(mp*me/(me+mp))*Tr)
# 	return alphae(Tr) * exp(-(rydberg/4)/Tr)/(thermlambda**3) 

# def CPeebles(xe,rs):
# 	# Peebles C-factor 
# 	num = Lambda2s*(1-xe) + 1/(KLyman(rs) * nH * rs**3)
# 	den = Lambda2s*(1-xe) + 1/(KLyman(rs) * nH * rs**3) + betae(TCMB(rs))*(1-xe)
# 	return num/den

def getTLADE(fz, injRate):

	def TLADE(var,rs):

		def xe(y): 
			return 0.5 + 0.5*tanh(y)

		def dTmdz(Tm,y,rs):
			return 2*Tm/rs - dtdz(rs) * (
				comptonCMB(xe(y), Tm, rs) 
				+ 1./(1. + xe(y) + nHe/nH)*2/(3*nH*rs**3)*fz['Heat'](rs,xe(y))*injRate(rs)
				)

		def dydz(Tm,y,rs):
			return (2*cosh(y)**2) * dtdz(rs) * (
				CPeebles(xe(y),rs)*(
					alphae(Tm)*xe(y)**2*nH*rs**3  
					- betae(TCMB(rs))*(1.-xe(y))*exp(-lyaEng/Tm)
					) 
				- fz['HIon'](rs,xe(y))*injRate(rs)/(rydberg*nH*rs**3) 
				- (1 - CPeebles(xe(y),rs))*fz['HLya'](rs,xe(y))*injRate(rs)/(lyaEng*nH*rs**3)
				)

		Tm, y = var

		# dvardz = ([
		# 	(2*Tm/rs - 
		# 	dtdz(rs)*(comptonCMB(xe(y), Tm, rs))),
		# 	(2*cosh(y)**2) * dtdz(rs) * (CPeebles(xe(y),rs)*
		# 		(alphae(Tm)*xe(y)**2*nH*rs**3 - 
		# 			betae(TCMB(rs))*(1-xe(y))*exp(-lyaEng/Tm)))])

	
		dvardz = (
			[dTmdz(Tm,y,rs), dydz(Tm,y,rs)]
			)
		
		return dvardz

	return TLADE

def getIonThermHist(initrs,initCond,fz,injRate,rsVec):

	ionThermHistDE = getTLADE(fz,injRate)
	return odeint(ionThermHistDE,initCond,rsVec,mxstep=500)