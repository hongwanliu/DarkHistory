""" Loads the transfer function data.

"""

import pickle

from config import data_path

# Load transfer functions for high-energy photons, low-energy photons, 
# low-energy electrons, high-energy deposition, and CMB upscattered energy.


# print('****** Loading transfer functions... ******')

# try:
#     highengphot_tf_interp
# except:
#     print('    for high-energy photons... ', end =' ')
#     highengphot_tf_interp = pickle.load(
#         open(data_path+'/highengphot_tf_interp.raw', 'rb')
#     )
#     print(' Done!')

# try:
#     lowengphot_tf_interp
# except:
#     print('    for low-energy photons... ', end=' ')
#     lowengphot_tf_interp  = pickle.load(
#         open(data_path+'/lowengphot_tf_interp.raw', 'rb')
#     )
#     print('Done!')

# try:
#     lowengelec_tf_interp
# except:
#     print('    for low-energy electrons... ', end=' ')
#     lowengelec_tf_interp  = pickle.load(
#         open(data_path+"/lowengelec_tf_interp.raw", "rb")
#     )
#     print('Done!')

# try:
#     highengdep_interp
# except:
#     print('    for high-energy deposition... ', end=' ')
#     highengdep_interp     = pickle.load(
#         open(data_path+"/highengdep_interp.raw", "rb")
#     )
#     print('Done!')

# try:
#     CMB_engloss_interp
# except:
#     print('    for total upscattered CMB energy... ', end=' ')
#     CMB_engloss_interp    = pickle.load(
#         open(data_path+"/CMB_engloss_interp.raw", "rb")
#     )
#     print('Done!')

# # Load inverse Compton scattering transfer functions. 

# try:
#     ics_thomson_ref_tf
# except:
#     print('    for inverse Compton (Thomson)... ', end=' ')
#     ics_thomson_ref_tf = pickle.load(
#         open(data_path+"/ics_thomson_ref_tf.raw", "rb")
#     )
#     print('Done!')

# try:
#     ics_rel_ref_tf
# except:
#     print('    for inverse Compton (relativistic)... ', end=' ')
#     ics_rel_ref_tf     = pickle.load(
#         open(data_path+"/ics_rel_ref_tf.raw",     "rb")
#     )
#     print('Done!')

# try:
#     engloss_ref_tf
# except:
#     print('    for inverse Compton (energy loss)... ', end=' ')
#     engloss_ref_tf     = pickle.load(
#         open(data_path+"/engloss_ref_tf.raw",     "rb")
#     )
#     print('Done!')

# print('****** All transfer functions loaded! ******')
