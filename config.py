""" Configuration and defaults.

"""

# import pytest

# pytest.skip("Skipping config tests.", allow_module_level=True)


import os
import sys

import numpy as np
import json
import pickle


# # Location of all data files. 

if not 'pytest' in sys.modules:

    data_path = '/Users/hongwan/Dropbox (MIT)/Photon Deposition/DarkHistory_data'
    #data_path = '/Users/gridgway/Dropbox (MIT)/Photon Deposition/DarkHistory_data'
    # data_path = '/Users/gregoryridgway/Dropbox (MIT)/Photon Deposition/DarkHistory_data'

    if data_path == '' or not os.path.isdir(data_path):
        print('NOTE: enter data directory in config.py to avoid this step.')
        data_path = input('Enter the data directory, e.g. /Users/foo/bar: ')

    # # Default binning for photons and low-energy electrons. 

    binning = np.loadtxt(open(data_path+'/default_binning.p', 'rb'))

    photeng = binning[0]
    eleceng = binning[1]

else:

    data_path = ''
    photeng   = np.zeros(500)
    eleceng   = np.zeros(500)

