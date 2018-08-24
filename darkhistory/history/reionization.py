"""Reionization model.

"""

import numpy as np 
import darkhistory.physics as phys

def photoion_rate(species):
    """ Photoionization rate. 

    Parameters
    ----------
    species : {'HI', 'HeI', 'HeII'}

    Returns
    -------
    function
        Interpolating function which takes redshift as an argument, and returns the photoionization rate of the species in s^-1. See 1801.04931. 
    """
    rs_vec = 1. + np.array([
        0, 0.0491, 0.101, 0.155, 0.211, 0.271, 0.333, 0.399, 0.468, 0.54, 
        0.615, 0.695, 0.778, 0.865, 0.957, 1.05, 1.15, 1.26, 1.37, 1.49,
        1.61, 1.74, 1.87, 2.01, 2.16, 2.32, 2.48, 2.65, 2.83, 3.02, 3.21,
        3.42, 3.64, 3.87, 4.11, 4.36, 4.62, 4.89, 5.18, 5.49, 5.81, 6.14,
        6.49, 6.86, 7.25, 7.65, 8.07, 8.52, 8.99, 9.48, 9.99, 10.5, 11.1,
        11.7, 12.3, 13.0, 13.7, 14.4, 15.1
    ])

    if species == 'HI':
        rate_vec = np.array([
            6060, 7430, 9080, 11000, 13400, 16100, 19400, 23100, 27500, 
            32400, 38100, 44300, 51300, 58800, 66900, 75300, 83800, 92200, 
            100000, 107000, 113000, 117000, 120000, 121000, 121000, 120000,
            117000, 112000, 107000, 102000, 96800, 91800, 87300, 83400, 
            80200, 77700, 75800, 74500, 73200, 70900, 60900, 311, 72.9, 33.7,
            21.5, 15.6, 12.0,9.67, 7.96, 6.67, 5.66, 4.83, 4.15, 3.56, 3.06,
            2.65, 2.30, 2.00, 1.74
        ]) * 1e-17

    elif species == 'HeI':
        rate_vec = np.array([
            3030, 3760, 4630, 5680, 6940, 8440, 10200, 12300, 14600, 17400,
            20500, 23900, 27700, 31900, 36300, 40800, 45400, 49900, 53900,
            57400, 60100, 61900, 62900, 62800, 61700, 59600, 56600, 53000, 
            49300, 46100, 43300, 40900, 39000, 37500, 36500, 36000, 35900,
            36100, 36300, 35500, 29700, 377, 89.4, 41.6, 26.8, 19.5, 15.2, 
            12.4, 10.3, 8.70, 7.43, 6.40, 5.52, 4.74, 4.08, 3.53, 3.06, 
            2.66, 2.32
        ]) * 1e-17

    elif species == 'HeII':
        rate_vec = np.array([
            110, 132, 159, 190, 226, 269, 319, 377, 443, 517, 600, 690, 
            788, 891, 997, 1100, 1190, 1270, 1320, 1330, 1300, 1240, 1150,
            1040, 892, 723, 530, 317, 121, 42.6, 21.0, 13.5, 9.56, 7.04, 
            5.30, 4.04, 2.98, 2.21, 1.66, 1.26, 0.988, 0.783, 0.345, 0.172, 
            0.0934, 0.0521, 0.0294, 0.0168, 0.00997, 0.00627, 0.00428, 
            0.00321, 0.00259, 0.00211, 0.00178, 0.00154, 0.00133, 0.00116, 
            0.00101
        ]) * 1e-17

    else: 
        raise TypeError('invalid species.')

    def ion_rate(rs):
        log10_rate = np.interp(rs, rs_vec, np.log10(rate_vec))
        return 10**log10_rate

    return ion_rate

def photoheat_rate(species):
    """ Photoheating rate. 

    Parameters
    ----------
    species : {'HI', 'HeI', 'HeII'}

    Returns
    -------
    function
        Interpolating function which takes redshift as an argument, and returns the photoionization rate of the species in s^-1. See 1801.04931. 
    """
    rs_vec = 1. + np.array([
        0, 0.0491, 0.101, 0.155, 0.211, 0.271, 0.333, 0.399, 0.468, 0.54, 
        0.615, 0.695, 0.778, 0.865, 0.957, 1.05, 1.15, 1.26, 1.37, 1.49,
        1.61, 1.74, 1.87, 2.01, 2.16, 2.32, 2.48, 2.65, 2.83, 3.02, 3.21,
        3.42, 3.64, 3.87, 4.11, 4.36, 4.62, 4.89, 5.18, 5.49, 5.81, 6.14,
        6.49, 6.86, 7.25, 7.65, 8.07, 8.52, 8.99, 9.48, 9.99, 10.5, 11.1,
        11.7, 12.3, 13.0, 13.7, 14.4, 15.1
    ])

    if species == 'HI':
        rate_vec = np.array([
            2280, 2810, 3440, 4190, 5090, 6150, 7400, 8850, 10500, 12400,
            14600, 17000, 19700, 22600, 25600, 28800, 32000, 35200, 38100,
            40600, 42600, 44200, 45200, 45500, 45200, 44300, 42900, 41000, 
            38900, 36900, 35000, 33200, 31700, 30400, 29400, 28600, 28100, 
            27900, 27600, 26800, 22900, 186, 44.1, 20.5, 13.2, 9.58, 7.44, 
            6.02, 4.99, 4.21, 3.58, 3.08, 2.65, 2.27, 1.96, 1.69, 1.47, 
            1.28, 1.11
        ]) * 1e-16

    elif species == 'HeI':
        rate_vec = np.array([
            2490, 3060, 3750, 4580, 5580, 6760, 8150, 9770, 11700, 13800,
            16200, 18900, 21900, 25100, 28500, 32000, 35500, 38800, 41700, 
            44100, 45800, 46800, 47000, 46300, 44600, 42000, 38600, 34500, 
            30300, 27500, 25400, 23800, 22400, 21500, 20900, 20500, 20400, 
            20400, 20500, 20000, 16600, 321, 76.5, 35.5, 22.6, 16.3, 12.6, 
            10.2, 8.49, 7.19, 6.15, 5.31, 4.59, 3.94, 3.39, 2.93, 2.55, 
            2.21, 1.92
        ]) * 1e-16

    elif species == 'HeII':
        rate_vec = np.array([
            208, 250, 300, 359, 428, 508, 600, 707, 827, 963, 1110, 1280, 
            1450, 1640, 1830, 2010, 2180, 2330, 2430, 2460, 2430, 2330, 2180,
            1980, 1720, 1420, 1070, 699, 345, 158, 90, 61.7, 45.1, 33.8, 
            25.6, 19.4, 14.4, 10.6, 7.72, 5.58, 4.02, 2.82, 1.45, 0.803, 
            0.454, 0.256, 0.143, 0.0795, 0.0446, 0.0258, 0.0156, 0.0101, 
            0.00696, 0.00500, 0.00372, 0.00282, 0.00213, 0.00158, 0.0010
        ]) * 1e-16

    else: 
        raise TypeError('invalid species.')

    def heat_rate(rs):
        log10_rate = np.interp(rs, rs_vec, np.log10(rate_vec))
        return 10**log10_rate

    return heat_rate

def alphaA_recomb(species, T):
    """Case-A recombination coefficient.  

    Parameters
    ----------
    species : {'HII', 'HeIIr', 'HeIId', 'HeIII'}
        Species of interest. 
    T : float
        Matter temperature in eV. 

    Returns
    -------
    float
        Case-A recombination coefficient in cm^3/s. See astro-ph/0607331.
    """
    if species == 'HII':
        return np.exp(
            -28.6130338 - 0.72411256 * np.log(T) 
            - 2.02604473e-2 * np.log(T)**2 - 2.38086188e-3 * np.log(T)**3
            - 3.21260521e-4 * np.log(T)**4 - 1.42150291e-5 * np.log(T)**5
            + 4.98910892e-6 * np.log(T)**6 + 5.75561414e-7 * np.log(T)**7
            - 1.85676704e-8 * np.log(T)**8 - 3.07113524e-9 * np.log(T)**9
        )

    elif species == 'HeIIr':
        return 3.925e-13 * T**-0.6533

    elif species == 'HeIId':
        return (
            1.544e-9 * T**-1.5 * (0.3*np.exp(-48.596/T) + np.exp(-40.496/T))
        )


    elif species == 'HeII':
        return alphaA_recomb('HeIIr', T) + alphaA_recomb('HeIId', T)

    elif species == 'HeIII':
        return 2*alphaA_recomb('HII',T/4)

def coll_ion_rate(species, T_m):
    """Collisional ionization rate.  

    Parameters
    ----------
    species : {'HI', 'HeI', 'HeII'}
        Species of interest. 
    T_m : float
        Matter temperature in eV. 

    Returns
    -------
    float
        Collisional ionization rate in s^-1. See astro-ph/0607331.
    """
    TinK = T_m/phys.kB
    T5_factor = 1/(1 + np.sqrt(TinK/1e5))

    if species == 'HI':
        return 1.17e-10 * np.sqrt(TinK) * np.exp(-157809.1/TinK) * T5_factor
    elif species == 'HeI':
        return 4.76e-11 * np.sqrt(TinK) * np.exp(-285335.4/TinK) * T5_factor
    elif species == 'HeII':
        return 1.14e-11 * np.sqrt(TinK) * np.exp(-631515.0/TinK) * T5_factor
    else:
        raise TypeError('invalid species.')

def recomb_cooling_rate(xHII, xHeII, xHeIII, T_m, rs): 
	"""Recombination cooling rate. 

	Parameters
	----------
	xHII : float
		n_HII/n_H. 
	xHeII : float
		n_HeII/n_H. 
	xHeIII : float
		n_HeIII/n_H. 
	T_m : float
		Matter temperature in eV. 
	rs : float
		Redshift (1+z). 

	Returns
	-------
	float
		Recombination cooling rate in eV s^-1. See astro-ph/0607331.
	"""
	xe   = xHII + xHeII + 2*xHeIII
	xHI  = 1 - xHII
	xHeI = phys.nHe/phys.nH - xHeII - xHeIII 

	return (
		-6.24e11 * (phys.nH * rs**3)**2 * (
			1.036e-16 * T_m/phys.kB * alphaA_recomb('HII', T_m) 
				* xe * xHII
			+ (
				1.036e-16 * T_m/phys.kB * alphaA_recomb('HeIIr', T_m)
				+ 6.526e-11 * alphaA_recomb('HeIId', T_m)
			) * xe * xHeII
			+ 1.036e-16 * T_m/phys.kB * alphaA_recomb('HeIII', T_m)
				* xe * xHeIII
		) 
	)

def coll_ion_cooling_rate(xHII, xHeII, xHeIII, T_m, rs):
	""" Collisional ionization cooling rate.
	
	Parameters
	----------
	xHII : float
		n_HII/n_H. 
	xHeII : float
		n_HeII/n_H. 
	xHeIII : float
		n_HeIII/n_H. 
	T_m : float
		Matter temperature in eV. 
	rs : float
		Redshift (1+z). 

	Returns
	-------
	float
		Collisional ionization cooling rate in eV s^-1. See astro-ph/0607331.

	"""
	xe   = xHII + xHeII + 2*xHeIII
	xHI  = 1 - xHII
	xHeI = phys.nHe/phys.nH - xHeII - xHeIII 

	return (
		-6.24e11 * xe * (phys.nH * rs**3)**2 * (
			2.18e-11 * coll_ion_rate('HI', T_m) * xHI
			+ 3.94e-11 * coll_ion_rate('HeI', T_m) * xHeI
			+ 8.72e-11 * coll_ion_rate('HeII', T_m) * xHeII
		)
	)

def coll_exc_cooling_rate(xHII, xHeII, xHeIII, T_m, rs):
	""" Collisional excitation cooling rate.
	
	Parameters
	----------
	xHII : float
		n_HII/n_H. 
	xHeII : float
		n_HeII/n_H. 
	xHeIII : float
		n_HeIII/n_H. 
	T_m : float
		Matter temperature in eV. 
	rs : float
		Redshift (1+z). 

	Returns
	-------
	float
		Collisional excitation cooling rate in eV s^-1. See astro-ph/0607331.

	"""
	xe   = xHII + xHeII + 2*xHeIII
	xHI  = 1 - xHII
	xHeI = phys.nHe/phys.nH - xHeII - xHeIII 

	T_in_K = T_m/phys.kB
	T_5_factor = (1 + np.sqrt(T_in_K/1e5))**-1

	return (
		-6.24e11 * xe * (phys.nH * rs**3)**2 * (
			7.50e-19 * np.exp(-118348/T_in_K) * T_5_factor * xHI
			+ 9.10e-27 * T_in_K**-0.1687 * np.exp(-13179.0/T_in_K)
				* T_5_factor * xe * (phys.nH * rs**3) * xHeI
			+ 5.54e-17 * T_in_K**-0.397 *  np.exp(-473638 /T_in_K)
				* T_5_factor * xHeII
		)
	)

def brem_cooling_rate(xHII, xHeII, xHeIII, T_m, rs):
	""" Bremsstrahlung cooling rate.
	
	Parameters
	----------
	xHII : float
		n_HII/n_H. 
	xHeII : float
		n_HeII/n_H. 
	xHeIII : float
		n_HeIII/n_H. 
	T_m : float
		Matter temperature in eV. 
	rs : float
		Redshift (1+z). 

	Returns
	-------
	float
		Bremsstrahlung cooling rate in eV s^-1. See astro-ph/0607331.

	"""
	xe   = xHII + xHeII + 2*xHeIII

	T_in_K = T_m/phys.kB
    # See astro-ph/9509107 Eq. 23
	gaunt_fac = 1.1 + 0.34 * np.exp(-(5.5 - np.log10(T_in_K))**2/3.0)

	return (
		-xe * 6.24e11 * (phys.nH*rs**3)**2 * (
			1.43e-27 * np.sqrt(T_in_K) * gaunt_fac
				* (xHII + xHeII + 4*xHeIII)
		)
	)









