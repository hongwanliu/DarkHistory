#!/usr/bin/env python
# coding: utf-8

m_inc   = 0.1
tau_inc = 0.25


import sys
sys.path.append("..")
sys.path.append("../..")
print(sys.path)

import numpy as np
import copy
import pickle
import csv
from scipy.interpolate import interp1d
from scipy.stats import chi2
from scipy import optimize

import darkhistory.physics as phys
from darkhistory.history.tla import get_history
from darkhistory.history import tla
import darkhistory.history.reionization as reion
from tqdm import tqdm_notebook as tqdm


import main
import config

# ### Reionization Data

pri          = sys.argv[1] #'elec' or 'phot'
DM_process   = sys.argv[2] #'decay' or 'pwave
reion_method = sys.argv[3] #'FlexKnot_early', 'FlexKnot_late', 'Tanh_early', 'Tanh_late'
constr_str   = sys.argv[4] #'conservative' or 'photoheated1' or 'photoheated2'
#output_dir   = sys.argv[5] #Try '../Lya_data/'
output_dir = '/Users/gregoryridgway/Desktop/Junk/'

print('LET IT BEGIN')
if constr_str == 'conservative':
    heat_switch = False
    guess_offset = -np.log10(2)
    min_DeltaT = 0
    print("deriving 'conservative' constraints")

elif constr_str == 'photoheated1':
    heat_switch = True
    min_DeltaT = 0
    guess_offset = 0
    print("deriving 'photoheated1' constraints")

elif constr_str == 'photoheated2':
    heat_switch = True
    min_DeltaT = 2e4*phys.kB
    guess_offset = np.log10(2)
    print("deriving 'photoheated2' constraints")

else:
    print('Invalid constr_str, exiting...')
    sys.exit()


#--- Downloading Planck reionization histories ---#
reion_strings = np.array(
    ['FlexKnot_early', 'FlexKnot_late', 'Tanh_early', 'Tanh_late']
)

def make_reion_interp_func(string, He_bump=False):
    Planck_data = []
    with open('Planck_reion_models/Planck_'+string+'.csv') as csvfile:
        reader = csv.reader(csvfile)
        reader = csv.reader(csvfile)
        for row in reader:
            Planck_data.append([float(row[0]),float(row[1])])
    Planck_data = np.array(Planck_data)

    #fix normalization
    if string == 'FlexKnot_early':
        norm_fac = Planck_data[-2,1]
    else:
        norm_fac = Planck_data[0,1]
        
        #I WebPlot Digitized poorly, so I re-zero
        if string == 'FlexKnot_late':
            Planck_data[26:,1] = 0
        elif string == 'Tanh_late':
            Planck_data[63:,1] = 0
            
        
    Planck_data[:,1] = (1+2*phys.chi)*Planck_data[:,1]/norm_fac

    #convert from z to rs
    Planck_data[:,0] = 1+Planck_data[:,0]

    fac = 2
    if He_bump == False:
        Planck_data[Planck_data[:,1]>1+phys.chi,1]=1+phys.chi
        fac = 1

    return interp1d(Planck_data[:,0], Planck_data[:,1], bounds_error=False, fill_value=(1+fac*phys.chi,0), kind='linear')

# Make interpolation functions for each Planck2018 reionization history
reion_interps = {string : make_reion_interp_func(string) for string in reion_strings}
#bump_interp = make_reion_interp_func('Tanh_late', True)


# ### Ly$\alpha$ data

ind = -7

z_entries=np.array([1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.6,5.0,5.4])
rs_entries=1+z_entries[ind:]

new_mids=np.array([.768,.732,1.014,1.165,1.234,
          1.286,1.289,1.186,1.404,1.038,
          1.205,.940,.890,.877,.533,.599])
mids = new_mids[ind:]

high_errs = np.array([.37,.17,.25,.29,.19,.19,.18,.13,.17,.31,.23,.22,.093,.13,.12,.15])
low_errs  = np.array([.22,.091,.15,.19,.14,.15,.14,.12,.16,.27,.19,.17,.073,.11,.091,.13])
sigs = low_errs[ind:]

Gaikwad_data = np.array([[1.1, .16], [1.05, .21], [1.2, .22]])
Gaikwad_rs   = np.array([6.4,6.6,6.8])

default_data = [
    np.concatenate((rs_entries[:-2], Gaikwad_rs)),
    (
        np.concatenate((mids[:-2], Gaikwad_data[:,0])),
        np.concatenate((high_errs[ind:-2], Gaikwad_data[:,1]))
    )
]


# ### Previous Constraints

#old_constraints = pickle.load(open(input_dir+'compiled_data.p','rb'))
log10m_forGuesses = {'phot': np.arange(4.01, 12.76, m_inc), 'elec': np.arange(6.01, 12.76, m_inc)}
guess_funcs = pickle.load(open('guess_funcs.p', 'rb'))

# ### Functions for photoheating constraints

rs_vec = 10**np.arange(np.log10(2.9e3), np.log10(4.55), -.01)

#Given DeltaT and alpha, return the temperature in 10^-4 K
def get_his(DeltaT, alpha, mDM=None, lifetime=None, sigmav=None, fs=[None, None, None, None], DM_process='decay'):
    struct_boost = None
    if (sigmav is not None):# & (): !!!struct_boost is mistakenly folded into the fs
        struct_boost = phys.struct_boost_func(model='pwave_NFW_no_subs')
    elif lifetime is not None:
        struct_boost = None
    else:
        struct_boost = None
        DM_process   = None
 
    tmp = tla.get_history(
        rs_vec,
        DM_process = DM_process, mDM=mDM, lifetime=lifetime, sigmav=sigmav,
        struct_boost = struct_boost,
        f_H_ion=fs[0], f_He_ion=fs[1], f_H_exc=fs[2], f_heating=fs[3],
        reion_switch=True, reion_rs=35, reion_method=None, 
        heat_switch=True, DeltaT = DeltaT, alpha_bk=alpha,
        xe_reion_func=reion_interps[reion_method], helium_TLA=True
    )
    return interp1d(rs_vec, tmp[:,0]/phys.kB*1e-4)

#Given DeltaT and alpha, find (two-sided) chi^2
def get_chisq(var, mDM=None, lifetime=None, sigmav=None, fs=[None, None, None, None], DM_process='decay'):
    DeltaT = var[0]
    alpha=var[1]
    terp = get_his(DeltaT, alpha, mDM=mDM, lifetime=lifetime, sigmav=sigmav, fs=fs, DM_process = DM_process)
    return sum((terp(default_data[0])-default_data[1][0])**2/default_data[1][1]**2)

#Given alpha, optimize DeltaT
def optimize_DeltaT(alpha, tol, mDM=None, lifetime=None, sigmav=None, fs=[None, None, None, None], DM_process='decay', min_DeltaT=0):
    def f(DeltaT):
        return get_chisq([DeltaT,alpha], mDM=mDM, lifetime=lifetime, sigmav=sigmav, fs=fs, DM_process = DM_process)

    return optimize.minimize_scalar(
        f, method='bounded', bounds=[min_DeltaT, 5e4*phys.kB], options={'xatol': tol}
    )

def find_optimum(alpha_list, init, tol=0.5, mDM=None, lifetime=None, sigmav=None, fs=[None, None, None, None], DM_process='decay', output=False, min_DeltaT=0):
    datums = [None for a in alpha_list]
    check_above = False
    check_below = False

    #Initialization Step
    j = init
    out = optimize_DeltaT(alpha_list[j], tol, mDM=mDM, lifetime=lifetime, sigmav=sigmav, fs=fs, DM_process = DM_process, min_DeltaT=min_DeltaT)
    datums[j] = [alpha_list[j], out['x']/phys.kB, out['fun']]
    count = 0
    
    #Search higher alpha
    while not check_above:
        count = count+1
        #At this point, you know the optimal value is above
        if count>1:
            check_below = True

        j = j+1
        out = optimize_DeltaT(alpha_list[j], tol, mDM=mDM, lifetime=lifetime, sigmav=sigmav, fs=fs, DM_process = DM_process, min_DeltaT=min_DeltaT)
        datums[j] = [alpha_list[j], out['x']/phys.kB, out['fun']]
        if output:
            print(datums[j])


        if datums[j][2] > datums[j-1][2]:
            check_above=True
            j = j-1
            break
        elif j==alpha_list.size-1:
            #if output:
            print('Reached maximum alpha!', datums[j])
            return np.array([datums[j],datums[j],datums[j]])
            
    #Search lower alpha
    while not check_below:
        j = j-1
        out = optimize_DeltaT(alpha_list[j], tol, mDM=mDM, lifetime=lifetime, sigmav=sigmav, fs=fs, DM_process = DM_process, min_DeltaT=min_DeltaT)
        datums[j] = [alpha_list[j], out['x']/phys.kB, out['fun']]

        if output:
            print(datums[j])
        if datums[j][2] > datums[j+1][2]:
            check_below=True
            j = j+1
        elif j==0:
            #if output:
            print('Reached minimum alpha!', datums[j])
            return np.array([datums[j],datums[j],datums[j]])
            
#     if output:
    print('Best value: ', datums[j])
#     #filter out all the None entries
#     return list(filter(None,datums))
    return np.array([datums[j-1], datums[j], datums[j+1]])


#Make the f functions
def make_fs(hist, pkl = False):
    channels = {'heat', 'H ion', 'He ion', 'exc'}
    f_interps = {chan: interp1d(
        hist['rs'], 
        hist['f']['low'][chan]+hist['f']['high'][chan],
        bounds_error=False,
        fill_value=(
            (hist['f']['low'][chan]+hist['f']['high'][chan])[-1],
            (hist['f']['low'][chan]+hist['f']['high'][chan])[0]
        )
    ) for chan in channels}
    
    def f_Hion(rs, xHI, xHeI, xHeII):
        return f_interps['H ion'](rs)
    def f_Heion(rs, xHI, xHeI, xHeII):
        return f_interps['He ion'](rs)
    def f_exc(rs, xHI, xHeI, xHeII):
        return f_interps['exc'](rs)
    def f_heat(rs, xHI, xHeI, xHeII):
        return f_interps['heat'](rs)
        
    if not pkl:
        return [f_Hion, f_Heion, f_exc, f_heat]
    else:
        return f_interps


# ## Generate Constraints

max_chisq = 10.1522
def find_param_guess(mDM, log10guess, inc, data, heat_switch=False, 
                     reion_method='FlexKnot_early',pri = 'elec', DM_process='decay',
                     DeltaT=24665*phys.kB, alpha_bk=0.57, min_DeltaT = 0):
    """
    Parameters
    ----------
    inc : float
        amount by which to increment the log10(parameter)
    DeltaT : float
        photoheating parameter, set to best fit value in the absence of DM
    alpha_bk : float
        photoheating parameter, set to best fit value in the absence of DM
    
    """
    below_target = False
    above_target = False
    chisq_list = []
    log10params = [log10guess]
    
    while (
        not below_target or not above_target
    ):

        param = 10**log10params[-1]
        print('log10param: '+str(log10params[-1]))
        if DM_process=='pwave':
            sign=-1.0
            struct_boost = phys.struct_boost_func(model='pwave_NFW_no_subs')
        else:
            sign = 1.0
            struct_boost = None 

        #base_hist = main.evolve(
        #    primary='elec_delta',
        #    DM_process='pwave', mDM=1e9, sigmav=1e-23,
        #    reion_switch=True, reion_rs=35, helium_TLA=True,
        #    xe_reion_func = reion_interps['FlexKnot_late'],
        #    start_rs = 3e3, end_rs=4.3, DeltaT = DeltaT, 
        #    alpha_bk=alpha_bk, heat_switch=False,
        #    coarsen_factor=12, backreaction=True, 
        #    compute_fs_method='He', mxstep=1000, rtol=1e-4,
        #    use_tqdm=False, cross_check = False,
        #    struct_boost=struct_boost
        #)
        #pickle.dump(base_hist, open(output_dir+'comparison.dat','wb'))
        #sys.exit()
       
        log10m_str = str(round(np.log10(mDM)*100)/100)
        log10param_str = str(round(log10params[-1]*100)/100)
        #if heat_switch:
        #    f_interps = pickle.load(open(
        #        output_dir+'/fs/mainEvolveOutput_FreeStream_'+reion_method+'_'+pri+'_'+DM_process
        #        +'log10mDM'+log10m_str+'_log10param'+log10param_str+'.dat','rb'
        #    ))
        #    def f_Hion(rs, xHI, xHeI, xHeII):
        #        return f_interps['H ion'](rs)
        #    def f_Heion(rs, xHI, xHeI, xHeII):
        #        return f_interps['He ion'](rs)
        #    def f_exc(rs, xHI, xHeI, xHeII):
        #        return f_interps['exc'](rs)
        #    def f_heat(rs, xHI, xHeI, xHeII):
        #        return f_interps['heat'](rs)

        #    fs = [f_Hion, f_Heion, f_exc, f_heat]
        #else:
        base_hist = main.evolve(
            primary=pri+'_delta',
            DM_process=DM_process, mDM=mDM, lifetime=param, sigmav=param,
            reion_switch=True, reion_rs=35, helium_TLA=True,
            xe_reion_func = reion_interps[reion_method],
            start_rs = 3e3, end_rs=4.3, DeltaT = DeltaT, 
            alpha_bk=alpha_bk, heat_switch=heat_switch,
            coarsen_factor=12, backreaction=True, 
            compute_fs_method='He', mxstep=1000, rtol=1e-4,
            use_tqdm=False, cross_check = False,
            struct_boost=struct_boost
        )
        f_interps = make_fs(base_hist, pkl=True)
        #pickle.dump(f_interps, open(
        #    output_dir+'/fs/mainEvolveOutput_FreeStream_'+reion_method+'_'+pri+'_'+DM_process
        #    +'log10mDM'+log10m_str+'_log10param'+log10param_str+'.dat','wb'
        #))
        if not heat_switch:
            Tm_interp = interp1d(base_hist['rs'], base_hist['Tm']/phys.kB*1e-4)
            diff = Tm_interp(data[0]) - data[1][0]
            diff[diff<0] = 0
            chisq = sum((diff/data[1][1])**2)
        else:
            fs = make_fs(base_hist)
            alpha_list = np.arange(-0.5,1.5,0.1)
            data = find_optimum(alpha_list, init=18, mDM=mDM, lifetime=param, sigmav=param, fs=fs, DM_process=DM_process, min_DeltaT=min_DeltaT)
            chisq = data[1][-1]
        
        chisq_list.append(chisq)
        print('Test Statistic: {:03.1e}'.format(chisq))

        nan_flag = np.any(np.isnan(base_hist['Tm']))
        if nan_flag:
            print('NAN! increasing heating rate.')
            log10params.append(log10params[-1]-inc*sign)
        elif chisq < max_chisq:
            below_target = True
            log10params.append(log10params[-1]-inc*sign)
        elif chisq > max_chisq:
            above_target = True
            log10params.append(log10params[-1]+inc*sign)

    param_interp = interp1d(chisq_list, log10params[:-1])
    return param_interp(max_chisq), chisq_list, log10params


if pri=='phot':
    log10m = np.arange(4.01, 12.76, m_inc)
else:
    log10m = np.arange(6.01, 12.76, m_inc)
max_chisq_list = np.zeros_like(log10m)
data = [[None for k in max_chisq_list], [None for k in max_chisq_list]]
for i, log10mDM in enumerate(log10m):
        if i>=0:
            print('****** log10(mDM): ', log10mDM, ' ******')
            mDM = 10**log10mDM

            # Format for Wenzer's data
            if reion_method[:4] == 'Flex':
                reion_str = reion_method[:4]+reion_method[8:]
            else:
                reion_str = reion_method

            if DM_process == 'decay':
                #param_str = 'taus'
                #fac = 0
                sign = 1
            else:
                #param_str = 'sigmav_over_ms'
                #fac = log10mDM - 9
                sign = -1

            #string = pri+'_'+DM_process+'_'+reion_str
            string = pri+'_'+DM_process
            log10guess = guess_funcs[string](log10mDM) + sign*guess_offset
            #log10guess = interp1d(old_constraints[string]['log10m'],
            #    old_constraints[string][param_str]
            #)(log10mDM) + fac

            max_chisq_list[i], tmp_chisq_list, tmp_param_list  = find_param_guess(
                mDM, log10guess, tau_inc, default_data, heat_switch=heat_switch,
                reion_method=reion_method, pri=pri, DM_process=DM_process, min_DeltaT=min_DeltaT
            )

            data[0][i] = tmp_chisq_list
            data[1][i] = tmp_param_list
            
            pickle.dump(data, open(output_dir+'dataFreeStream_'+constr_str+'_'+reion_method+'_'+pri+'_'+DM_process+'.dat','wb'))
