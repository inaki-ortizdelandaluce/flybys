import numpy as np
from flybys.spice import Spice
from flybys.quaternion import Quaternion
from flybys.helper import normalize, closest_approach, find_switch


# Winslow et al. 2013
_bowshock_models = {"slavin": {"l": 2.40, "eps": 1.07, "x0": 0.5},
                    "winslow": {"l": 2.96, "eps": 1.02, "x0": 0.5}}

_magnetopause_models = {"korth": {"rss": 1.42, "alpha": 0.5}}

_dipole_offset = 479


def _inside_bowshock(vv, model):
    xx = vv[:, 0] - model["x0"]
    yy = np.sqrt(vv[:, 1] ** 2 + vv[:, 2] ** 2)
    rr2 = xx ** 2 + yy ** 2
    theta = np.arctan2(yy, xx)
    rb = np.divide(model["l"], 1 + model["eps"] * np.cos(theta))
    rb2 = rb ** 2
    return rr2 < rb2


def _inside_magnetopause(vv, model):
    xx = vv[:, 0]
    yy = np.sqrt(vv[:, 1] ** 2 + vv[:, 2] ** 2)
    rr2 = xx ** 2 + yy ** 2
    theta = np.arctan2(yy, xx)
    rm = np.multiply(model["rss"], np.power(np.divide(2, 1 + np.cos(theta)), model["alpha"]))
    rm2 = rm ** 2
    return rr2 < rm2


def _mso2msm(vv):
    vv[:, 2] = vv[:, 2] - _dipole_offset
    return vv


def mercury_closest_approach(metakernel, utc_start, utc_end):
    return closest_approach('MERCURY', metakernel, utc_start, utc_end)


def mercury_bowshock_crossings(metakernel, utc_start, utc_end, model='winslow'):
    spice = Spice()
    spice.load_metakernel(metakernel)

    # closest approach
    etc = spice.closest_approach('MPO', 'MERCURY', utc_start, utc_end, False, 100)[0]

    # compute spacecraft positions in Mercury Solar Magnetospheric coordinates
    # starting two hours before the closest approach
    tt = etc - 7200 + np.arange(10000)
    vv = spice.position('MPO', tt, 'BC_MSM', 'MERCURY')
    vv = _mso2msm(vv)
    vv = vv / spice.body_radius('MERCURY')

    bowshock_model = _bowshock_models.get(model)
    if bowshock_model is None:
        raise ValueError("Unknown bowshock model {}".format(model))

    inside = _inside_bowshock(vv, bowshock_model)
    _entry, _exit = find_switch(inside)

    tte = spice.et2utc(tt[_entry])
    ttx = spice.et2utc(tt[_exit])

    spice.clear()
    return tte, ttx


def mercury_magnetopause_crossings(metakernel, utc_start, utc_end, model='korth'):
    spice = Spice()
    spice.load_metakernel(metakernel)

    # closest approach
    etc = spice.closest_approach('MPO', 'MERCURY', utc_start, utc_end, False, 100)[0]

    # compute spacecraft positions in Mercury Solar Magnetospheric coordinates
    # starting two hours before the closest approach
    tt = etc - 7200 + np.arange(10000)
    vv = spice.position('MPO', tt, 'BC_MSM', 'MERCURY')
    vv = _mso2msm(vv)
    vv = vv / spice.body_radius('MERCURY')

    magnetopause_model = _magnetopause_models.get(model)
    if magnetopause_model is None:
        raise ValueError("Unknown magnetopause model {}".format(model))

    inside = _inside_magnetopause(vv, magnetopause_model)
    _entry, _exit = find_switch(inside)

    tte = spice.et2utc(tt[_entry])
    ttx = spice.et2utc(tt[_exit])

    spice.clear()
    return tte, ttx
