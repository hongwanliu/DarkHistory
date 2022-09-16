import sys
sys.path.append('../../..')


import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import pickle



import darkhistory.physics as phys
import darkhistory.history.tla as tla
import darkhistory.low_energy.atomic as atomic
import darkhistory.low_energy.bound_free as bf

output_dir = '/scratch/gpfs/hongwanl/DarkHistory/full_distortion/scratch/'

hplanck = phys.hbar * 2 * np.pi

eng = np.exp(np.linspace(
    np.log(hplanck*1e8), np.log(phys.rydberg), 500))

rs_list = np.exp(np.arange(np.log(2e3), 5., -.001*4))

baseline = tla.get_history(rs_list, high_rs=1.555e3,
                           fudge=1.0, gauss_fudge=False, rtol=1e-8)
fudge_1_14 = tla.get_history(rs_list, high_rs=1.555e3,
                             fudge=1.14, gauss_fudge=False, rtol=1e-8)

dist_eng = np.exp(np.linspace(np.log(hplanck*1e8), np.log(phys.rydberg), 10))

nmax_list = [50,100,200]#,300]
# nmax_list = [10]

nmax_convergence_alt = {}
for nmax in nmax_list:
    R = atomic.populate_radial(nmax)
    Thetas = bf.populate_thetas(nmax)

    MLA_data = np.zeros((rs_list.size, 3))
    for i, rs in enumerate(tqdm(rs_list)):

        MLA_data[i], _ = atomic.process_MLA(
            rs, 1, phys.x_std(rs, 'HI'), 
            phys.Tm_std(rs), nmax, eng, R, Thetas
        )

    MLA_funcs = [interp1d(rs_list, MLA_data[:,i], fill_value='extrapolate')
                 for i in range(3)]

    nmax_convergence_alt[nmax] = tla.get_history(
        rs_list, high_rs=1.555e3,
        recfast_TLA=False, MLA_funcs=MLA_funcs,
        rtol=1e-8
    )
    pickle.dump([baseline, fudge_1_14, nmax_convergence_alt], open(output_dir+'nmax_convergence.dat', 'wb'))