import numpy as np
import pickle
import sys
import os
#os.system("taskset -p -c 0-15 %d" % os.getpid())
import time
from tqdm import tqdm

DH_DIR = '/zfs/yitians/darkhistory/DarkHistory/'
sys.path.append(DH_DIR)

from config import load_data
import darkhistory.physics as phys
from darkhistory.spec.spectrum import Spectrum
import darkhistory.spec.spectools as spectools
from darkhistory.spec.transferfunclist import TransferFuncInterp
import darkhistory.history.tla as tla

NNDH_DIR = '/zfs/yitians/darkhistory/NNDH/'
sys.path.append(NNDH_DIR)
from common import *

import matplotlib.pyplot as plt
from matplotlib import rc_file
rc_file(NNDH_DIR+'matplotlibrc')

from astropy.io import fits
from scipy.interpolate import interp1d
from pprint import pprint

SOLN_DIR = '/zfs/yitians/darkhistory/DarkHistory/nntf/tests/run_output/'

import tensorflow as tf

import main

soln = main.evolve(
    DM_process='decay', mDM=1e8, lifetime=3e25, primary='phot_delta',
    start_rs = 3000, end_rs = 2900, coarsen_factor=12, backreaction=True, helium_TLA=True, reion_switch=True,
    tf_mode='table', use_tqdm=False
)
pickle.dump(soln, open('test.p','wb'))
#pickle.dump(soln, open(SAVE_DIR+'E8c12_ctf_test.p','wb'))