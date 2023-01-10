import math
import numpy as np


class Quaternion:
    """Class to represent a quaternion.
    Quaternion objects can be used generically as 4D numbers,
    or as unit quaternions to represent rotations in 3D space.
    Attributes:
        q: Quaternion 4-vector represented as a Numpy array
    """

    def __init__(self, *args, **kwargs):
        s = len(args)
        if s == 0:
            if kwargs:
                if ("axis" in kwargs) and (("degrees" in kwargs) or ("radians" in kwargs)):
                    axis = self._to_valid_array(kwargs["axis"], (3, ))
                    angle = kwargs.get("radians") or math.radians(kwargs.get("degrees"))
                    self.q = Quaternion._from_axis_angle(axis, angle).q
                elif "matrix" in kwargs:
                    matrix = self._to_valid_array(kwargs["matrix"], (3, 3))
                    self.q = Quaternion._from_matrix(matrix).q
                else:
                    self.q = np.zeros((4,))
            else:
                self.q = np.zeros((4,))
        elif s == 1:
            self.q = self._to_valid_array(args[0], (4, ))
        else:
            raise TypeError("Quaternion cannot be initialised with multiple arguments")

    def normalize(self):
        n = np.linalg.norm(self.q)
        if n > 0:
            self.q = self.q / n
        else:
            raise ValueError('A zero-norm quaternion cannot be normalized')

    def conjugate(self):
        return Quaternion(np.append(self.q[0], -1 * self.q[1:]))

    def inverse(self):
        q_inverse = self.conjugate()
        n = np.linalg.norm(self.q)
        if n > 0:
            q_inverse.q = q_inverse.q / n**2
            return q_inverse
        else:
            raise ValueError('A zero-norm quaternion cannot be inverted')

    def scalar(self):
        return self.q[0]

    def vector(self):
        return self.q[1:]

    def rotate(self, v):
        """Rotate a 3-vector or a 2d array of 3-vectors by the rotation stored in the Quaternion object.
        Params:
            v: A 2d array of 3-vectors stacked vertically,
                or 3-vector specified as any ordered sequence of 3 real numbers corresponding to
                x, y, and z values (can be either a numpy array, list or tuple).
        Returns:
            The rotated vector returned as the same type it was specified at input.
        """
        vv = np.asarray(v).reshape((1, 3)) if np.shape(v) == (3, ) else v
        return np.apply_along_axis(Quaternion._rotate_vector, 1, vv, self)

    @staticmethod
    def multiply(q1, q2):
        if Quaternion._is_quaternion(q1) and Quaternion._is_quaternion(q2):
            q = np.zeros((4,))
            q[0] = q1.q[0] * q2.q[0] - np.dot(q1.q[1:], q2.q[1:])
            q[1] = q1.q[0] * q2.q[1] + q1.q[1] * q2.q[0] + q1.q[2] * q2.q[3] - q1.q[3] * q2.q[2]
            q[2] = q1.q[0] * q2.q[2] + q1.q[2] * q2.q[0] + q1.q[3] * q2.q[1] - q1.q[1] * q2.q[3]
            q[3] = q1.q[0] * q2.q[3] + q1.q[3] * q2.q[0] + q1.q[1] * q2.q[2] - q1.q[2] * q2.q[1]
            return Quaternion(q)

    @staticmethod
    def _rotate_vector(v, q):
        """Rotate a 3D vector by the rotation stored in the Quaternion object.
        Params:
            vector: A 3-vector specified as any ordered sequence of 3 real numbers corresponding to x, y, and z values.
                Some types that are recognised are: numpy arrays, lists and tuples.
        Returns:
            The rotated vector returned as the same type it was specified at input.
        """
        p = Quaternion(np.hstack((0, np.asarray(v))))
        q.normalize()
        vp = Quaternion.multiply(Quaternion.multiply(q, p), q.inverse()).vector()

        if isinstance(v, list):
            return [x for x in vp]
        elif isinstance(v, tuple):
            return tuple([x for x in vp])
        else:
            return vp

    @staticmethod
    def _is_quaternion(q):
        return isinstance(q, Quaternion)

    @staticmethod
    def _to_valid_array(array, shape):
        if Quaternion._is_numeric_array(array):
            q = np.asarray(array)
            if q.shape == shape:
                return q
        raise TypeError("Input {} is not a valid {} array".format(array, shape))

    @staticmethod
    def _is_numeric_array(array):
        """Checks if the data type of the array is numeric.
        unsigned integer, signed integer, floats are considered numeric.
        """
        numerical_dtypes = {'u',  # unsigned integer
                            'i',  # signed integer
                            'f'}  # float
        try:
            return array.dtype.kind in numerical_dtypes
        except AttributeError:
            return np.asarray(array).dtype.kind in numerical_dtypes

    @classmethod
    def _from_axis_angle(cls, axis, angle):
        n = np.linalg.norm(axis)
        if n == 0.0:
            raise ZeroDivisionError("Rotation axis has zero length")

        # normalize axis
        if abs(1.0 - n) > 1e-12:
            axis = axis / n

        q = np.zeros((4, ))
        theta = angle / 2.0
        q[0] = math.cos(theta)
        q[1:] = math.sin(theta) * axis
        return cls(q)

    @classmethod
    def _from_matrix(cls, matrix):
        q = np.zeros((4, ))
        d = 0.5 * math.sqrt(1 + matrix[0][0] + matrix[1][1] + matrix[2][2])
        tol = 0.01
        if d > tol:
            q[0] = d
            q[1] = 0.25 / d * (matrix[2][1] - matrix[1][2])
            q[2] = 0.25 / d * (matrix[0][2] - matrix[2][0])
            q[3] = 0.25 / d * (matrix[1][0] - matrix[0][1])
        else:
            d = 0.5 * math.sqrt(1 + matrix[0][0] - matrix[1][1] - matrix[2][2])
            if d > tol:
                q[0] = 0.25 / d * (matrix[2][1] - matrix[1][2])
                q[1] = d
                q[2] = 0.25 / d * (matrix[0][1] + matrix[1][0])
                q[3] = 0.25 / d * (matrix[0][2] + matrix[2][0])
            else:
                d = 0.5 * math.sqrt(1 - matrix[0][0] + matrix[1][1] - matrix[2][2])
                if d > tol:
                    q[0] = 0.25 / d * (matrix[0][2] - matrix[2][0])
                    q[1] = 0.25 / d * (matrix[0][1] + matrix[1][0])
                    q[2] = d
                    q[3] = 0.25 / d * (matrix[1][2] + matrix[2][1])
                else:
                    d = 0.5 * math.sqrt(1 - matrix[0][0] - matrix[1][1] + matrix[2][2])
                    if d > tol:
                        q[0] = 0.25 / d * (matrix[1][0] - matrix[0][1])
                        q[1] = 0.25 / d * (matrix[0][2] + matrix[2][0])
                        q[2] = 0.25 / d * (matrix[1][2] + matrix[2][1])
                        q[3] = d
        return cls(q)

    def __eq__(self, other):
        if isinstance(other, Quaternion):
            c = self.q == other.q
            return c.all()
        return False

    def __hash__(self):
        return hash(tuple(self.q))

    def __str__(self):
        """An informal, nicely printable string representation of the Quaternion object.
        """
        return "{:.3f} {:+.3f}i {:+.3f}j {:+.3f}k".format(self.q[0], self.q[1], self.q[2], self.q[3])

    def __repr__(self):
        """The 'official' string representation of the Quaternion object.
        This is a string representation of a valid Python expression that could be used
        to recreate an object with the same value (given an appropriate environment)
        """
        return "Quaternion({!r}, {!r}, {!r}, {!r})".format(self.q[0], self.q[1], self.q[2], self.q[3])

    def __format__(self, format_spec):
        """Inserts a customisable, nicely printable string representation of the Quaternion object The syntax for
        `format_spec` mirrors that of the built in format specifiers for floating point types. Check out the official
        Python [format specification mini-language](https://docs.python.org/3.4/library/string.html#formatspec) for
        details.
        """
        if format_spec.strip() == '':  # Default behaviour mirrors self.__str__()
            format_spec = '+.3f'

        string = \
            "{:" + format_spec + "} " + \
            "{:" + format_spec + "}i " + \
            "{:" + format_spec + "}j " + \
            "{:" + format_spec + "}k"
        return string.format(self.q[0], self.q[1], self.q[2], self.q[3])