import pytest
import numpy as np


def test_version(spice):
    assert spice.version() == 'CSPICE_N0066'


def test_utc2et(spice):
    assert pytest.approx(spice.utc2et('2021-08-10T13:51:54'), 1e-7) == 681875583.1830401


def test_et2utc(spice):
    assert spice.et2utc(681875583.1830401) == '2021-08-10T13:51:54'


def test_body_radius(spice):
    assert spice.body_radius('VENUS') == 6051.8
    assert spice.body_radius('MERCURY') == 2439.7


def test_position(spice, et, position):
    pos = spice.position('MPO', et, 'J2000', 'VENUS')
    assert np.allclose(pos, position)


def test_state(spice, et, position, velocity):
    pos, vel = spice.state('MPO', et, 'J2000', 'VENUS')
    assert np.allclose(pos, position) and np.allclose(vel, velocity)


def test_closest_approach(spice):
    et = spice.closest_approach('MPO', 'VENUS', '2021-08-09T14:00:00', '2021-08-11T14:00:00', False, 100)[0]
    assert et == 681875582.8367949

