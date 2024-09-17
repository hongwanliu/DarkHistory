import os
import sys
import h5py
from pytest import approx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_evolve():
    from darkhistory.main import evolve
    
    soln = evolve(
        DM_process='decay', mDM=1e8, lifetime=3e25, primary='elec_delta',
        start_rs = 3000,
        coarsen_factor=12, backreaction=True, helium_TLA=True, reion_switch=True
    )
    soln_dict = {
        'rs' : soln['rs'],
        'x' : soln['x'],
        'Tm' : soln['Tm'],
        'highengphot' : soln['highengphot'].grid_vals,
        'lowengphot' : soln['lowengphot'].grid_vals,
        'lowengelec' : soln['lowengelec'].grid_vals,
    }
    with h5py.File(os.path.dirname(os.path.realpath(__file__)) + '/data/test_evolve_5affda21.h5', 'r') as f:
        for key in soln_dict:
            assert soln_dict[key] == approx(f[key][:], rel=1e-3, abs=1e-5)
