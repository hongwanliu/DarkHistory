"""Electron cooling processes.

Electrons cool via inverse Compton scattering (ICS), which is handled by the package :mod:`.ics`. The full electron cooling calculation, including both ICS and atomic cooling, is done in :mod:`.elec_cooling`.

Positrons ultimately annihilate with electrons after first forming positronium: the resulting spectrum is calculated in :mod:`.positronium`. 
"""