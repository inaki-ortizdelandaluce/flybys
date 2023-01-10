import numpy as np
from flybys.quaternion import Quaternion
from flybys.spice import Spice
from flybys.helper import normalize, closest_approach, find_switch


# Martinecz et al. 2008
_bowshock_models = {"martinecz": {"l": 1.303, "eps": 1.056, "x0": 0.788},
                    "slavin": {"l": 1.68, "eps": 1.03, "x0": 0.45},
                    "russell": {"l": 2.14, "eps": 0.609, "x0": 0.},
                    "zhang": {"l": 2.131, "eps": 0.66, "x0": 0.},
                    "tricicle": {"l": 1.515, "eps": 1.018, "x0": 0.664}}


def _inside_bowshock(vv, model):
    xx = vv[:, 0] - model["x0"]
    yy = np.sqrt(vv[:, 1] ** 2 + vv[:, 2] ** 2)
    rr2 = xx ** 2 + yy ** 2
    theta = np.arctan2(yy, xx)
    rb = np.divide(model["l"], 1 + model["eps"] * np.cos(theta))
    rb2 = rb ** 2
    return rr2 < rb2


def venus_closest_approach(metakernel, utc_start, utc_end):
    return closest_approach('VENUS', metakernel, utc_start, utc_end)


def venus_bowshock_crossings(metakernel, utc_start, utc_end, model='martinecz', aberration=False):
    spice = Spice()
    spice.load_metakernel(metakernel)

    # closest approach
    etc = spice.closest_approach('MPO', 'VENUS', utc_start, utc_end, False, 100)[0]
    rs, vs = spice.state('SUN', etc, 'J2000', 'VENUS')

    # build quaternion from rotation matrix to convert to Venus Solar Orbital coordinates
    vx = normalize(rs)  # venus-sun direction
    vy = normalize(vs)  # sun orbital velocity
    vz = normalize(np.cross(vx, vy))
    qv = Quaternion(matrix=np.array([vx, vy, vz]).transpose()).inverse()

    # build quaternion to correct Venus Solar Orbital coordinates from solar-wind aberration
    # assuming average solar wind and mean orbital velocity values (400 and 35 km/s respectively)
    # aberration ~ -5 deg
    qa = Quaternion(axis=np.array([0, 0, 1]), degrees=-5 * aberration).inverse()

    # compute spacecraft positions in Venus Solar Orbital coordinates corrected from solar-wind aberration
    # starting half an hour before the closest approach
    tt = etc - 1800 + np.arange(10000)
    rm = spice.position('MPO', tt, 'J2000', 'VENUS')
    vv = Quaternion.multiply(qa, qv).rotate(rm) / spice.body_radius('VENUS')

    bowshock_model = _bowshock_models.get(model)
    if bowshock_model is None:
        raise ValueError("Unknown bowshock model {}".format(model))

    inside = _inside_bowshock(vv, bowshock_model)
    _entry, _exit = find_switch(inside)

    tte = spice.et2utc(tt[_entry])
    ttx = spice.et2utc(tt[_exit])

    spice.clear()
    return tte, ttx
