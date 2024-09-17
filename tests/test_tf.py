import os
import sys
import pytest
import h5py

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from darkhistory.config import load_data


@pytest.fixture(scope='module')
def data_loader():
    data = {
        'dep_tf' : load_data('dep_tf'),
        'tf_helper' : load_data('tf_helper'),
        'ics_tf' : load_data('ics_tf'),
        'expected' : {},
    }
    with h5py.File(os.path.dirname(os.path.realpath(__file__)) + '/data/test_tf_2261700c.h5', 'r') as hf:
        for k in hf.keys():
            data['expected'][k] = hf[k][()]
    return data


def test_dep_tf(data_loader):
    tfs = data_loader['dep_tf']
    for k in ['highengphot', 'lowengphot', 'lowengelec']:
        tf = tfs[k]
        z = tf.get_tf(0.433, 0.302, 2244).sum_specs(np.sin(np.arange(500)))
        z += tf.get_tf(0.760, 0.276, 384).sum_specs(np.sin(np.arange(500)))
        z += tf.get_tf(0.930, 0.088, 18).sum_specs(np.sin(np.arange(500)))
        z = z.N
        assert np.allclose(z, data_loader['expected'][k])
    for k in ['highengdep', 'CMB_engloss']:
        tf = tfs[k]
        z = np.sin(np.arange(500)) @ tf.get_val(0.433, 0.302, 2244)
        z += np.sin(np.arange(500)) @ tf.get_val(0.760, 0.276, 384)
        z += np.sin(np.arange(500)) @ tf.get_val(0.930, 0.088, 18)
        assert np.allclose(z, data_loader['expected'][k])

def test_ics_tf(data_loader):
    tfs = data_loader['ics_tf']
    for k in ['thomson', 'rel', 'engloss']:
        tf = tfs[k]
        z = tf.sum_specs(np.sin(np.arange(5000)))
        z = z.N
        assert np.allclose(z, data_loader['expected'][k])

def test_tf_helper(data_loader):
    tfs = data_loader['tf_helper']
    for k in ['tf_E']:
        tf = tfs[k]
        z = tf.get_val(0.433, 0.302, 2244) @ np.sin(np.arange(500))
        z += tf.get_val(0.760, 0.276, 384) @ np.sin(np.arange(500))
        z += tf.get_val(0.930, 0.088, 18) @ np.sin(np.arange(500))
        assert np.allclose(z, data_loader['expected'][k])

    
