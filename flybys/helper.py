import numpy as np
from flybys.spice import Spice


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def find_switch(condition):
    """Finds the indexes where the condition array elements switch values
    Params:
        condition: boolean array
    Returns:
        Two arrays, one with the indexes where condition elements switch from False to True, and another where they
        switch from True to False
    """
    condition_prev = np.roll(condition, 1)
    condition_prev[0] = condition[0]  # to fix circular shift
    switch, = np.where(condition ^ condition_prev)  # true-to-false/false-to-true

    positive = switch[np.where(condition[switch])]
    negative = switch[np.where(~condition[switch])]
    return positive, negative


def closest_approach(body, metakernel, utc_start, utc_end):
    spice = Spice()
    spice.load_metakernel(metakernel)

    etc = spice.closest_approach('MPO', body, utc_start, utc_end, False, 100)
    if etc is not None:
        etc = etc[0]

    utc = spice.et2utc(etc)
    spice.clear()
    return utc

