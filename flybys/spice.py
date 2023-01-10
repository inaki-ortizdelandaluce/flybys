import spiceypy as spice
import spiceypy.utils.support_types as stypes
import numpy as np
import re
import os.path as path
import tempfile


class Spice:

    def __init__(self):
        pass

    @staticmethod
    def version():
        return spice.tkvrsn('TOOLKIT')

    @staticmethod
    def load(kernels):
        spice.furnsh(kernels)

    @staticmethod
    def load_metakernel(kernel):

        with open(kernel, 'r') as f:
            content = f.read()

            # read kernel path values
            regexp = r'(PATH_VALUES\s+=\s+\(\s+\')(.*?)(\'\s+\))'
            result = re.search(regexp, content)
            path_values = result.group(2)

            if path_values == '..':
                # copy kernel to temporary file with updated path value
                path_values_new = path.abspath(path.join(kernel, '../..'))
                content_new = re.sub(regexp, r'\1' + path_values_new + r'\3', content, flags=re.M)
                kernel_new = tempfile.NamedTemporaryFile(mode='w', delete=False)
                with kernel_new as mk:
                    mk.write(content_new)
                    print('Temporary metakernel {} created'.format(mk.name))
                Spice.load(kernel_new.name)
            else:
                Spice.load(kernel)

    @staticmethod
    def clear():
        spice.kclear()

    @staticmethod
    def et2utc(et):
        return spice.et2utc(et, 'ISOC', 0)

    @staticmethod
    def utc2et(utc):
        return spice.utc2et(utc)

    @staticmethod
    def body_radius(body):
        """Returns body radius in kilometers
        Params:
            body: the body name
        Returns:
            The body radius in kilometers
        """
        dim, radii = spice.bodvrd(body, 'RADII', 3)
        return np.mean(radii)

    @staticmethod
    def position(target, et, frame, observer):
        """Finds the position of the target body relative to the observing body for the times
            specified.
        Params:
            target: name of the target body
            et: ephemeris times for the positions to be computed
            frame: reference frame relative to which the position vector should be expressed
            observer: name of the observing body
        Returns:
            Array of position vectors of the target body relative to an observing body.
        """
        pos, lt = spice.spkpos(target, et, frame, 'NONE', observer)
        return pos

    @staticmethod
    def state(target, et, frame, observer):
        """Finds the state (position and velocity) of the target body relative to the observing body for the times
            specified.
        Params:
            target: name of the target body
            et: ephemeris times for the positions to be computed
            frame: reference frame relative to which the position vector should be expressed
            observer: name of the observing body
        Returns:
            Tuple of position and velocity vectors of the target body relative to an observing body.
        """
        state, lt = spice.spkezr(target, et, frame, 'NONE', observer)

        if isinstance(state, list):
            state = np.asarray(state)
            return state[:, 0:3], state[:, 3:6]
        else:
            return state[0:3], state[3:6]

    def closest_approach(self, target, observer, utc_start, utc_end, multiple, step):
        """Finds closest approaches of the target to the observer during the time period specified.
        Params:
            target: name of the target body
            observer: name of the observing body
            utc_start: start time of the applicable time period in UTC format, e.g. 2021-08-09T14:00:00
            utc_end: end time of the applicable time period in UTC format, e.g. 2021-08-11T14:00:00
            multiple: if true computes all closest distances at a local minima for the applicable time period,
                if false computes the closest approach at the absolute minimum.
            step: step size for this search in seconds. The step must be shorter than the shortest interval over which
                the target-observer distance is increasing or decreasing.
        Returns:
            Array of ephemeris times for the closest approaches matching the search criteria, None if no closest
            approach is found.
        """
        et_start = self.utc2et(utc_start)
        et_end = self.utc2et(utc_end)

        confine = stypes.SPICEDOUBLE_CELL(2)
        spice.wninsd(et_start, et_end, confine)

        ca_win = spice.gfdist(target, 'NONE', observer, 'LOCMIN' if multiple else 'ABSMIN', 0.0, 0.0, step, 1000,
                              confine)
        win_size = spice.wncard(ca_win)

        if win_size == 0:
            return None
        else:
            return [spice.wnfetd(ca_win, i)[0] for i in range(win_size)]

