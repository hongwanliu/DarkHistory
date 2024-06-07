import os
import sys
from pytest import approx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from darkhistory.spec.spectrum import Spectrum

def test_redshift():

    eng = np.array([3, 10, 29, 3000])
    N = np.array([0.2, 4.6, 3.38e5, 201.041])
    spec = Spectrum(eng, N, spec_type='N', rs=2302.3)
    orig_totN = spec.totN()
    orig_toteng = spec.toteng()
    spec.redshift(842.10)
    assert spec.totN() == approx(orig_totN)
    assert spec.toteng() == approx(orig_toteng*842.10/2302.3)
