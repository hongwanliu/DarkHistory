""" Module for setting DarkHistory up.

"""

import numpy as np
from numpy.linalg import matrix_power
import pickle

from scipy.interpolate import interp1d

import darkhistory.physics as phys
import darkhistory.utilities as utils
import darkhistory.spec.spectools as spectools
from darkhistory.spec.spectools import EnglossRebinData
import darkhistory.spec.transferfunclist as tflist
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec.spectra import Spectra
from darkhistory.spec import transferfunction as tf
import darkhistory.history.histools as ht
import darkhistory.history.tla as tla

from darkhistory.electrons.ics.ics_spectrum import ics_spec
from darkhistory.electrons.ics.ics_spectrum import nonrel_spec
from darkhistory.electrons.ics.ics_spectrum import rel_spec
from darkhistory.electrons.ics.ics_engloss_spectrum import engloss_spec
from darkhistory.electrons.ics.ics_cooling import get_ics_cooling_tf
from darkhistory.electrons.elec_cooling import get_elec_cooling_tf_fast
from darkhistory.electrons.elec_cooling import \
    get_elec_cooling_tf_fast_linalg

from darkhistory.electrons import positronium as pos

from darkhistory.low_energy.lowE_deposition import compute_fs
from darkhistory.low_energy.lowE_electrons import make_interpolator

import os
cwd = os.getcwd()
abspath = os.path.abspath(__file__)
dir_path = os.path.dirname(abspath)

def load_trans_funcs(
    direc_arr, xes, rs_nodes, string_arr = [''], CMB_subtracted=False
):

    # Load the transfer functions. 

    if isinstance(direc_arr, str):
        direc_arr = [direc_arr]

    num_rs_nodes = len(string_arr)-1
    if num_rs_nodes == 0:
        xes = np.array([xes])


    highengphot_tflistarrs = []
    lowengphot_tflistarrs  = []
    lowengelec_tflistarrs  = []
    highengdep_ionrsarrs   = []
    CMB_engloss_ionrsarrs  = []

    for ii,(string,xe) in enumerate(zip(string_arr, xes)):
        print('Loading transfer functions, set', ii)
        highengphot_tflist_arr = pickle.load(
            open(
                direc_arr[ii]+"tfunclist_photspec_60eV_complete_"
                +string+".raw", "rb"
            )
        )
        print('Loaded high energy photons...')

        lowengphot_tflist_arr = pickle.load(
            open(
                direc_arr[ii]+"tfunclist_lowengphotspec_60eV_complete_"
                +string+".raw", "rb"
            )
        )

        print('Low energy photons...')

        lowengelec_tflist_arr = pickle.load(
            open(
                direc_arr[ii]+"tfunclist_lowengelecspec_60eV_complete_"
                +string+".raw", "rb"
            )
        )

        print('Low energy electrons...')

        highengdep_arr = pickle.load(
            open(
                direc_arr[ii]+"highdeposited_60eV_complete_"
                +string+".raw", "rb"
            )
        )
        highengdep_arr = np.swapaxes(highengdep_arr, -2, -3)
        # (xH, rs, in_eng, channel) or (xH, xHe, rs, in_eng, channel)

        print('High energy deposition...')

        CMB_engloss_arr = pickle.load(
            open(
                direc_arr[ii]+"CMB_engloss_60eV_complete_"
                +string+".raw", "rb"
            )
        )
        CMB_engloss_arr = np.swapaxes(CMB_engloss_arr, -1, -2)
        # (xH, rs, in_eng) or (xH, xHe, rs, in_eng)
        
        print('CMB losses.')

        print('Transfer function lists set ', ii, ' complete.')

        print('Padding transfer functions...')

        try:
            photeng = highengphot_tflist_arr[0].eng
            eleceng = lowengelec_tflist_arr[0].eng
            rs_list = highengphot_tflist_arr[0].rs
        except:
            photeng = highengphot_tflist_arr[0][0].eng
            eleceng = lowengelec_tflist_arr[0][0].eng
            rs_list = highengphot_tflist_arr[0][0].rs 

        #Split photeng into high and low energy.
        photeng_high = photeng[photeng > 60]
        photeng_low  = photeng[photeng <= 60]

        # Split eleceng into high and low energy.
        eleceng_high = eleceng[eleceng > 3000]
        eleceng_low  = eleceng[eleceng <= 3000]

        if isinstance(highengphot_tflist_arr[0], list):
            ndim = 2
        elif isinstance(highengphot_tflist_arr, list):
            ndim = 1
        else:
            raise TypeError('highengphot_tflist_arr must be a list.')


        def pad(tfl, tf_type):

            # Pad with zeros so that it becomes (photeng x photeng/eleceng).
            for tf in tfl:
                tf._grid_vals = np.pad(
                    tf.grid_vals, ((photeng_low.size, 0), (0,0)),
                    'constant'
                )
                if tf_type == 'lowengphot':
                    tf._grid_vals[0:photeng_low.size, 0:photeng_low.size] = (
                        np.identity(photeng_low.size)
                    )

                tf._N_underflow = np.pad(
                    tf._N_underflow, (photeng_low.size, 0), 'constant'
                )
                tf._eng_underflow = np.pad(
                    tf._eng_underflow, (photeng_low.size, 0), 'constant'
                )

                tf._in_eng = photeng
            
                if tf_type == 'lowengelec':
                    tf._eng = eleceng
                else:
                    tf._eng = photeng
                
                tf._rs      = tf.rs[0]*np.ones_like(photeng)

            return None

        if ndim == 1:
            # Array of TransferFuncList objects. 
            print('High energy photons...')
            for highengphot_tflist in highengphot_tflist_arr:
                pad(highengphot_tflist, 'highengphot')
                highengphot_tflist._eng    = photeng
                highengphot_tflist._in_eng = photeng
                highengphot_tflist._grid_vals = np.atleast_3d(
                    np.stack(
                        [tf.grid_vals for tf in highengphot_tflist._tflist]
                    )
                )

            print('Low energy photons...')
            for lowengphot_tflist in lowengphot_tflist_arr:
                pad(lowengphot_tflist, 'lowengphot')
                lowengphot_tflist._eng = photeng
                lowengphot_tflist._in_eng = photeng
                lowengphot_tflist._grid_vals = np.atleast_3d(
                    np.stack(
                        [tf.grid_vals for tf in lowengphot_tflist._tflist]
                    )
                )

            print('Low energy electrons...')
            for lowengelec_tflist in lowengelec_tflist_arr:
                pad(lowengelec_tflist, 'lowengelec')
                lowengelec_tflist._eng = eleceng
                lowengelec_tflist._in_eng = photeng
                lowengelec_tflist._grid_vals = np.atleast_3d(
                    np.stack(
                        [tf.grid_vals for tf in lowengelec_tflist._tflist]
                    )
                )

            print('High energy deposition...')

            highengdep_arr = np.pad(
                highengdep_arr, ((0,0), (0,0), (photeng_low.size, 0), (0,0)),
                'constant'
            )

            ind_below_lya = len(photeng[photeng < phys.lya_eng])
            if CMB_subtracted:
                print('CMB losses and subtracting CMB component...')
            else:
                print('CMB losses...')
            CMB_engloss_arr = np.pad(
                CMB_engloss_arr, ((0,0), (0,0), (photeng_low.size, 0)),
                'constant'
            )

            if CMB_subtracted:
                for i,CMB_engloss_rs in enumerate(CMB_engloss_arr):
                    for j,(rs,CMB_engloss_in_eng) in (
                        enumerate(zip(rs_list, CMB_engloss_rs))
                    ):
                        T_at_rs = phys.TCMB(rs)
                        norm_CMB_spec = ( 
                            Spectrum(
                                photeng, phys.CMB_spec(photeng, T_at_rs)
                            ) / phys.CMB_eng_density(T_at_rs)
                        )
                        norm_CMB_spec = norm_CMB_spec.N
                        CMB_specs = (
                            np.outer(
                                CMB_engloss_in_eng, norm_CMB_spec
                            )
                            * 0.001 / phys.hubble(rs * np.exp(-0.001))
                        ) 
                        tfl = lowengphot_tflist_arr[i]
                        tfl._grid_vals[j,:,:ind_below_lya]-= (
                            CMB_specs[:,:ind_below_lya]
                        )

        elif ndim == 2:
            # 2D array of TransferFuncList objects. 
            print('High energy photons...')
            for highengphot_tflist_xHe in highengphot_tflist_arr:
                for highengphot_tflist in highengphot_tflist_xHe:
                    pad(highengphot_tflist, 'highengphot')
                    highengphot_tflist._eng    = photeng
                    highengphot_tflist._in_eng = photeng
                    highengphot_tflist._grid_vals = np.atleast_3d(
                        np.stack(
                            [
                                tf.grid_vals 
                                for tf in highengphot_tflist._tflist
                            ]
                        )
                    )

            print('Low energy photons...')
            for lowengphot_tflist_xHe in lowengphot_tflist_arr:
                for lowengphot_tflist in lowengphot_tflist_xHe:
                    pad(lowengphot_tflist, 'lowengphot')
                    lowengphot_tflist._eng = photeng
                    lowengphot_tflist._in_eng = photeng
                    lowengphot_tflist._grid_vals = np.atleast_3d(
                        np.stack(
                            [
                                tf.grid_vals 
                                for tf in lowengphot_tflist._tflist
                            ]
                        )
                    )

            print('Low energy electrons...')
            for lowengelec_tflist_xHe in lowengelec_tflist_arr:
                for lowengelec_tflist in lowengelec_tflist_xHe:
                    pad(lowengelec_tflist, 'lowengelec')
                    lowengelec_tflist._eng = eleceng
                    lowengelec_tflist._in_eng = photeng
                    lowengelec_tflist._grid_vals = np.atleast_3d(
                        np.stack(
                            [
                                tf.grid_vals 
                                for tf in lowengelec_tflist._tflist
                            ]
                        )
                    )

            print('High energy deposition...')
            highengdep_arr = np.pad(
                highengdep_arr, 
                ((0,0), (0,0), (0,0), (photeng_low.size, 0), (0,0)),
                'constant'
            )

            ind_below_lya = len(photeng[photeng < phys.lya_eng])
            if CMB_subtracted:
                print('CMB losses and subtracting CMB component...')
            else:
                print('CMB losses...')
            CMB_engloss_arr = np.pad(
                CMB_engloss_arr,
                ((0,0), (0,0), (0,0), (photeng_low.size, 0)),
                'constant'
            )
            if CMB_subtracted:
                for i,CMB_engloss_xHe in enumerate(CMB_engloss_arr):
                    for j,CMB_engloss_rs in enumerate(CMB_engloss_xHe):
                        for k,(rs,CMB_engloss_in_eng) in (
                            enumerate(zip(rs_list, CMB_engloss_rs))
                        ):
                            T_at_rs = phys.TCMB(rs)
                            norm_CMB_spec = ( 
                                Spectrum(
                                    photeng, phys.CMB_spec(photeng, T_at_rs)
                                ) / phys.CMB_eng_density(T_at_rs)
                            )
                            norm_CMB_spec = norm_CMB_spec.N
                            CMB_specs = (
                                np.outer(CMB_engloss_in_eng, norm_CMB_spec)
                                * 0.001/phys.hubble(rs * np.exp(-0.001))
                            )
                            tfl = lowengphot_tflist_arr[i][j]
                            tfl._grid_vals[k, :, :ind_below_lya] -= (
                                CMB_specs[:, :ind_below_lya]
                            )

        print('Creating array objects...')
        highengphot_tflistarr = tflist.TransferFuncListArray(
            highengphot_tflist_arr, xe
        )
        lowengphot_tflistarr = tflist.TransferFuncListArray(
            lowengphot_tflist_arr, xe
        )
        lowengelec_tflistarr = tflist.TransferFuncListArray(
            lowengelec_tflist_arr, xe
        )
        highengdep_ionrsarr  = ht.IonRSArray(
            highengdep_arr, xe, rs_list, in_eng=photeng
        )
        CMB_engloss_ionrsarr = ht.IonRSArray(
            CMB_engloss_arr, xe, rs_list, in_eng=photeng
        )

        # print("Generating TransferFuncInterp objects for each tflist...")
        # highengphot_tf_interp[ii] = tflist.TransferFuncInterp(highengphot_tflist_arr.copy(), xes[ii], log_interp = False)
        # lowengphot_tf_interp[ii]  = tflist.TransferFuncInterp(lowengphot_tflist_arr.copy(), xes[ii], log_interp = False)
        # lowengelec_tf_interp[ii]  = tflist.TransferFuncInterp(lowengelec_tflist_arr.copy(), xes[ii], log_interp = False)
        # highengdep_interp[ii]     = ht.IonRSInterp(xes[ii], rs_list, highengdep_arr.copy(), logInterp=False)
        # CMB_engloss_interp[ii]    = ht.IonRSInterp(xes[ii], rs_list, CMB_engloss_arr.copy(), logInterp=False)

        print('Generating TransferFuncInterp object...')

        highengphot_tflistarrs.append(highengphot_tflistarr)
        lowengphot_tflistarrs.append(lowengphot_tflistarr)
        lowengelec_tflistarrs.append(lowengelec_tflistarr)
        highengdep_ionrsarrs.append(highengdep_ionrsarr)
        CMB_engloss_ionrsarrs.append(CMB_engloss_ionrsarr)

    # if num_rs_nodes == 0:
    #     highengphot_tf_interp = highengphot_tf_interp[0]
    #     lowengphot_tf_interp  = lowengphot_tf_interp[0]
    #     lowengelec_tf_interp  = lowengelec_tf_interp[0]
    #     highengdep_interp     = highengdep_interp[0]
    #     CMB_engloss_interp    = CMB_engloss_interp[0]
    # else:
    #     highengphot_tf_interp = tflist.TransferFuncInterps(highengphot_tf_interp, xes)
    #     lowengphot_tf_interp = tflist.TransferFuncInterps(lowengphot_tf_interp, xes)
    #     lowengelec_tf_interp = tflist.TransferFuncInterps(lowengelec_tf_interp, xes)
    #     highengdep_interp = ht.IonRSInterps(highengdep_interp, xes)
    #     CMB_engloss_interp = ht.IonRSInterps(CMB_engloss_interp, xes)


    highengphot_tf_interp = tflist.TransferFuncInterp(
        highengphot_tflistarrs, rs_nodes, log_interp = False
    )

    lowengphot_tf_interp  = tflist.TransferFuncInterp(
        lowengphot_tflistarrs,  rs_nodes, log_interp = False
    )

    lowengelec_tf_interp  = tflist.TransferFuncInterp(
        lowengelec_tflistarrs,  rs_nodes, log_interp = False
    )
    highengdep_interp = ht.IonRSInterp(
        highengdep_ionrsarrs, rs_nodes, log_interp = False
    )

    CMB_engloss_interp = ht.IonRSInterp(
        CMB_engloss_ionrsarrs, rs_nodes, log_interp = False
    )

    return (
        highengphot_tf_interp, lowengphot_tf_interp, lowengelec_tf_interp, 
        highengdep_interp, CMB_engloss_interp
    )


    


        

        


# def load_trans_funcs(
#     direc_arr, xes, string_arr = [""], inverted=True, CMB_subtracted=False
# ):
#     # Load in the transferfunctions
#     #!!! Should be a directory internal to DarkHistory
#     #If only a string is specified, make it a list of strings
#     if isinstance(direc_arr, str):
#         direc_arr = [direc_arr]

#     num_rs_nodes = len(string_arr)-1
#     if num_rs_nodes == 0:
#         xes = np.array([xes])

#     arr = np.array([None for i in np.arange(num_rs_nodes+1)])

#     highengphot_tf_interp = arr.copy()
#     lowengphot_tf_interp  = arr.copy()
#     lowengelec_tf_interp  = arr.copy()
#     highengdep_interp     = arr.copy()
#     CMB_engloss_interp    = arr.copy()

#     for ii, string in enumerate(string_arr):
#         print('Loading transfer functions...')
#         highengphot_tflist_arr = pickle.load(open(direc_arr[ii]+"tfunclist_photspec_60eV_complete"+string+".raw", "rb"))
#         #highengphot_tflist_arr = pickle.load(open(direc+"tfunclist_photspec_60eV_injE_complete_rs_30_xe_2pts.raw", "rb"))
#         print('Loaded high energy photons...')

#         lowengphot_tflist_arr  = pickle.load(open(direc_arr[ii]+"tfunclist_lowengphotspec_60eV_complete"+string+".raw", "rb"))
#         #lowengphot_tflist_arr  = pickle.load(open(direc+"tfunclist_lowengphotspec_60eV_injE_complete_rs_30_xe_2pts.raw", "rb"))
#         print('Low energy photons...')

#         lowengelec_tflist_arr  = pickle.load(open(direc_arr[ii]+"tfunclist_lowengelecspec_60eV_complete"+string+".raw", "rb"))
#         #lowengelec_tflist_arr  = pickle.load(open(direc+"tfunclist_lowengelecspec_60eV_injE_complete_rs_30_xe_2pts.raw", "rb"))
#         print('Low energy electrons...')

#         highengdep_arr = pickle.load(open(direc_arr[ii]+"highdeposited_60eV_complete"+string+".raw", "rb"))
#         #highengdep_arr = pickle.load(open(direc+"highdeposited_60eV_injE_complete_rs_30_xe_2pts.raw", "rb"))
#         highengdep_arr = np.swapaxes(highengdep_arr, 1, 2)
#         print('high energy deposition.\n')

#         CMB_engloss_arr = pickle.load(open(direc_arr[ii]+"CMB_engloss_60eV_complete"+string+".raw", "rb"))
#         #CMB_engloss_arr = pickle.load(open(direc+"CMB_engloss_60eV_injE_complete_rs_30_xe_2pts.raw", "rb"))
#         CMB_engloss_arr = np.swapaxes(CMB_engloss_arr, 1, 2)
#         print('CMB losses.\n')

#         photeng = highengphot_tflist_arr[0].eng
#         eleceng = lowengelec_tflist_arr[0].eng
#         rs_list = highengphot_tflist_arr[0].rs

#         #Split photeng into high and low energy.
#         photeng_high = photeng[photeng > 60]
#         photeng_low  = photeng[photeng <= 60]

#         # Split eleceng into high and low energy.
#         eleceng_high = eleceng[eleceng > 3000]
#         eleceng_low  = eleceng[eleceng <= 3000]

#         print('Padding tflists with zeros...')
#         for highengphot_tflist in highengphot_tflist_arr:
#             for tf in highengphot_tflist:
#                 # Pad with zeros so that it becomes photeng x photeng.
#                 tf._grid_vals = np.pad(tf.grid_vals, ((photeng_low.size, 0), (0, 0)), 'constant')
#                 tf._N_underflow = np.pad(tf._N_underflow, (photeng_low.size, 0), 'constant')
#                 tf._eng_underflow = np.pad(tf._eng_underflow, (photeng_low.size, 0), 'constant')
#                 tf._in_eng = photeng
#                 tf._eng = photeng
#                 tf._rs = tf.rs[0]*np.ones_like(photeng)

#             highengphot_tflist._eng = photeng
#             highengphot_tflist._in_eng = photeng
#             highengphot_tflist._grid_vals = np.atleast_3d(
#                 np.stack([tf.grid_vals for tf in highengphot_tflist._tflist])
#             )
#         print("high energy photons...")

#         # lowengphot_tflist.in_eng set to photeng_high
#         for lowengphot_tflist in lowengphot_tflist_arr:
#             for tf in lowengphot_tflist:
#                 # Pad with zeros so that it becomes photeng x photeng.
#                 tf._grid_vals = np.pad(tf.grid_vals, ((photeng_low.size,0), (0,0)), 'constant')
#                 # Photons in the low energy bins should be immediately deposited.
#                 tf._grid_vals[0:photeng_low.size, 0:photeng_low.size] = np.identity(photeng_low.size)
#                 tf._N_underflow = np.pad(tf._N_underflow, (photeng_low.size, 0), 'constant')
#                 tf._eng_underflow = np.pad(tf._eng_underflow, (photeng_low.size, 0), 'constant')
#                 tf._in_eng = photeng
#                 tf._eng = photeng
#                 tf._rs = tf.rs[0]*np.ones_like(photeng)

#             lowengphot_tflist._eng = photeng
#             lowengphot_tflist._in_eng = photeng
#             lowengphot_tflist._grid_vals = np.atleast_3d(
#                 np.stack([tf.grid_vals for tf in lowengphot_tflist._tflist])
#             )
#         print("low energy photons...")

#         # lowengelec_tflist.in_eng set to photeng_high
#         for lowengelec_tflist in lowengelec_tflist_arr:
#             for tf in lowengelec_tflist:
#                 # Pad with zeros so that it becomes photeng x eleceng.
#                 tf._grid_vals = np.pad(tf.grid_vals, ((photeng_low.size,0), (0,0)), 'constant')
#                 tf._N_underflow = np.pad(tf._N_underflow, (photeng_low.size, 0), 'constant')
#                 tf._eng_underflow = np.pad(tf._eng_underflow, (photeng_low.size, 0), 'constant')
#                 tf._in_eng = photeng
#                 tf._eng = eleceng
#                 tf._rs = tf.rs[0]*np.ones_like(photeng)

#             lowengelec_tflist._eng = eleceng
#             lowengelec_tflist._in_eng = photeng
#             lowengelec_tflist._grid_vals = np.atleast_3d(
#                 np.stack([tf.grid_vals for tf in lowengelec_tflist._tflist])
#             )
#         print("low energy electrons...\n")

#         if xes[ii] is not None:
#             dim0 = len(xes[ii])
#         else:
#             dim0 = 1

#         tmp = np.zeros((dim0, len(rs_list), len(photeng), 4))
#         for i, highdep in enumerate(highengdep_arr):
#             tmp[i] = np.pad(highdep, ((0,0),(photeng_low.size, 0),(0,0)), 'constant')
#         highengdep_arr = tmp.copy()
#         print("high energy deposition.\n")

#         tmp = np.zeros((dim0, len(rs_list), len(photeng)))
#         for i, engloss in enumerate(CMB_engloss_arr):
#             tmp[i] = np.pad(engloss, ((0,0),(photeng_low.size, 0)), 'constant')
#         CMB_engloss_arr = tmp.copy()
#         print("CMB losses.\n")

#         ind = len(photeng[photeng < phys.lya_eng])
#         if CMB_subtracted:
#             print("Subtracting CMB component from lowengphot")
#             for i, CMB_engloss in enumerate(CMB_engloss_arr):
#                 for j, (rs, CMB_at_rs) in enumerate(zip(rs_list, CMB_engloss)):
#                     T_at_rs = phys.TCMB(rs)
#                     for k, CMB_at_ineng in enumerate(CMB_at_rs):
#                         CMB_spec = Spectrum(photeng, phys.CMB_spec(photeng,T_at_rs))/phys.CMB_eng_density(T_at_rs)*(
#                             CMB_at_ineng*.001/phys.hubble(rs*np.exp(-.001)) #To put in IDL's off-by-one error
#                         )
#                         lowengphot_tflist_arr[i]._grid_vals[j,k][:ind] = lowengphot_tflist_arr[i]._grid_vals[j,k][:ind] - CMB_spec.N[:ind]
#             print("Finished CMB subtraction")


#         print("Generating TransferFuncInterp objects for each tflist...")
#         # BEWARE THAT THE ORDER OF ARGUMENTS MAY HAVE CHANGED!!!!!!!
#         #print(lowengphot_tflist_arr[0][30]._grid_vals[300])
#         highengphot_tf_interp[ii] = tflist.TransferFuncInterp(xes[ii], highengphot_tflist_arr.copy(), log_interp = False)
#         lowengphot_tf_interp[ii]  = tflist.TransferFuncInterp(xes[ii], lowengphot_tflist_arr.copy(), log_interp = False)
#         lowengelec_tf_interp[ii]  = tflist.TransferFuncInterp(xes[ii], lowengelec_tflist_arr.copy(), log_interp = False)
#         highengdep_interp[ii]     = ht.IonRSInterp(xes[ii], rs_list, highengdep_arr.copy(), logInterp=False)
#         CMB_engloss_interp[ii]    = ht.IonRSInterp(xes[ii], rs_list, CMB_engloss_arr.copy(), logInterp=False)

#     print("Done.\n")

#     #print(lowengphot_tflist_arr.copy()[0][30]._grid_vals[300])
#     if num_rs_nodes == 0:
#         highengphot_tf_interp = highengphot_tf_interp[0]
#         lowengphot_tf_interp  = lowengphot_tf_interp[0]
#         lowengelec_tf_interp  = lowengelec_tf_interp[0]
#         highengdep_interp     = highengdep_interp[0]
#         CMB_engloss_interp    = CMB_engloss_interp[0]
#     else:
#         highengphot_tf_interp = tflist.TransferFuncInterps(highengphot_tf_interp, xes, inverted=inverted)
#         lowengphot_tf_interp = tflist.TransferFuncInterps(lowengphot_tf_interp, xes, inverted=inverted)
#         lowengelec_tf_interp = tflist.TransferFuncInterps(lowengelec_tf_interp, xes, inverted=inverted)
#         highengdep_interp = ht.IonRSInterps(highengdep_interp, xes, inverted=inverted)
#         CMB_engloss_interp = ht.IonRSInterps(CMB_engloss_interp, xes, inverted=inverted)

#     return highengphot_tf_interp, lowengphot_tf_interp, lowengelec_tf_interp, highengdep_interp, CMB_engloss_interp



def load_ics_data():
    Emax = 1e20
    Emin = 1e-8
    nEe = 5000
    nEp  = 5000

    dlnEp = np.log(Emax/Emin)/nEp
    lowengEp_rel = Emin*np.exp((np.arange(nEp)+0.5)*dlnEp)

    dlnEe = np.log(Emax/Emin)/nEe
    lowengEe_rel = Emin*np.exp((np.arange(nEe)+0.5)*dlnEe)

    Emax = 1e10
    Emin = 1e-8
    nEe = 5000
    nEp  = 5000

    dlnEp = np.log(Emax/Emin)/nEp
    lowengEp_nonrel = Emin*np.exp((np.arange(nEp)+0.5)*dlnEp)

    dlnEe = np.log(Emax/Emin)/nEe
    lowengEe_nonrel = Emin*np.exp((np.arange(nEe)+0.5)*dlnEe)

    print('********* Thomson regime scattered photon spectrum *********')
    ics_thomson_ref_tf = nonrel_spec(lowengEe_nonrel, lowengEp_nonrel, phys.TCMB(400))
    print('********* Relativistic regime scattered photon spectrum *********')
    ics_rel_ref_tf = rel_spec(lowengEe_rel, lowengEp_rel, phys.TCMB(400), inf_upp_bound=True)
    print('********* Thomson regime energy loss spectrum *********')
    engloss_ref_tf = engloss_spec(lowengEe_nonrel, lowengEp_nonrel, phys.TCMB(400), nonrel=True)
    return ics_thomson_ref_tf, ics_rel_ref_tf, engloss_ref_tf

