import pytest
import os.path as path
import numpy as np
from flybys.spice import Spice


@pytest.fixture
def spice():
    spice = Spice()
    kernels = ['lsk/naif0012.tls',
               'pck/pck00010.tpc',
               'spk/de432s.bsp',
               'spk/bc_mpo_fcp_00104_20181020_20251101_v01.bsp']
    kernels = [path.join(path.join(path.dirname(path.abspath(__file__)), 'data/kernels'), k) for k in kernels]
    spice.load(kernels)
    # metakernel = path.join(path.join(path.dirname(path.abspath(__file__)), 'data/kernels'), 'mk/vfb2.tm')
    # spice.load_metakernel(metakernel)
    yield spice
    spice.clear()


@pytest.fixture
def et():
    return np.array([681875583.1830401, 681789692.1830631])


@pytest.fixture
def position():
    return np.array([[6491.31632635, -612.03085992, 1047.29988246],
                     [-319631.40822425, -583001.02527864, -239503.49504016]])


@pytest.fixture
def velocity():
    return np.array([[0.53402401, 12.14407062, 3.82433007], [3.75210387, 6.59233577, 2.73393529]])
