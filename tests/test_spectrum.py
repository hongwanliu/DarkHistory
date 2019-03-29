import numpy as np

from pytest import approx

from darkhistory.spec.spectrum import Spectrum

def test_totN():

    eng = np.array([1, 10, 100, 1000])
    N   = np.array([1, 2, 3, 4])
    spec = Spectrum(eng, N, spec_type='N')
    assert spec.totN() == 10.0
    assert spec.totN('bin', np.array([1, 3])) == np.array([5.])
    assert spec.totN('eng', np.array([10, 1e4])) == np.array([8.])

def test_redshift():

    eng = np.array([3, 10, 29, 3000])
    N = np.array([0.2, 4.6, 3.38e5, 201.041])
    spec = Spectrum(eng, N, spec_type='N', rs=2302.3)
    orig_totN = spec.totN()
    orig_toteng = spec.toteng()
    spec.redshift(842.10)
    assert spec.totN() == approx(orig_totN)
    assert spec.toteng() == approx(orig_toteng*842.10/2302.3)
