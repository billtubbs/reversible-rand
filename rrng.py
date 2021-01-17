# Pseudo-Random Number Generation Algorithms

import numpy as np


def is_power_of_two(x):
    return (x & (x - 1)) == 0


def xgcd_x(a, b):
    """Extended Euclidean algorithm to determine the integer x
    in the equation:
    
        ax + by = gcd(a, b)
    
    where gcd(a, b) is the greatest common divisor of a and b.
    
    x and y are known as Bézout coefficients.
    """
    x_prev, x = 1, 0
    
    while b:
        q = a // b
        x, x_prev = x_prev - q*x, x
        a, b = b, a % b
    
    return x_prev


# Pre-computed values for faster execution
a_inverse_stored = {} 

def get_a_inverse_value(a, b):
    """Returns a pre-calculated value of a_inverse if one
    exists otherwise calculates it using the extended
    Euclidean algoriothm, xgcd_x(a, b).
    """
    if (a, b) in a_inverse_stored:
        return a_inverse_stored[(a, b)]
    else:
        x = xgcd_x(a, b)
        a_inverse_stored[(a, b)] = x
        return x


class ReversibleLCG:
    """Reversible Linear Congruential Generator
    
    The LCG is a type of pseudo-random number generator that
    supports generating random numbers in two directions (i.e.
    forwards and backwards).
    
    The internal state of the generator is a single integer, x.
    The following formula is used to produce the next state
    value in the sequence:
    
        nextx = (a * x + c) % m
    
    where a, c, and m are constants. With appropriate choice
    of these parameters, the period of the sequence is long.
    
    The d least-significant bits of x are removed and the
    output value returned by the generator is given by
    
        x >> d
    
    This truncation technique produces sequences with longer
    periods and statistically better values.  The values
    generated are in the range 0 to ((m-1) >> d).
    
    Arguments
    ----------
    seed : int
        Initial value of x to seed the random number generator.
    m : int
        Modulus.
    a : int
        Multiplicand.
    c : int
        Increment.
    d : int
        Least significant bits to discard.
    a_inverse : int
        The value of `a` used to step in the reverse direction.
    max : int
        Maximum value that can be produced.
    """

    def __init__(self, seed, m=1<<63, a=6364136223846793005, 
                 c=1442695040888963407, d=32):
        self.x = seed
        assert is_power_of_two(m), "`m` must be a power of two."
        self._m = m
        self._a = a
        self._c = c
        self._d = d
        self._a_inverse = get_a_inverse_value(self._a, self._m)
        self.forward = True

    @property
    def m(self):
        return self._m

    @property
    def a(self):
        return self._a

    @property
    def c(self):
        return self._c

    @property
    def d(self):
        return self._d

    @property
    def a_inverse(self):
        return self._a_inverse

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def max(self):
        return (self._m - 1) >> self._d

    def __iter__(self):
        return self

    def __next__(self):
        if self.forward:
            return self.next()
        else:
            return self.prev()

    def _next_state(self):
        # nextx = (a * x + c) % m
        return (self._a * self._x + self._c) & (self._m - 1)

    def _prev_state(self):
        # prevx = (ainverse * (x - c)) mod m
        return self._a_inverse * (self._x - self._c) & (self._m - 1)

    def reverse(self):
        self.forward = not self.forward

    def next(self):
        """Compute and return next value in sequence
        (forwards).
        """
        self._x = self._next_state()
        return self._x >> self._d

    def prev(self):
        """Compute and return previous value in sequence
        (backwards).
        """
        self._x = self._prev_state()
        return self._x >> self._d


class GeneratorLCGReversible(ReversibleLCG):
    """Reversible Linear Congruential Generator with additional
    vectorized methods.
    
    The LCG is a type of pseudo-random number generator that
    supports generating random numbers in two directions (i.e.
    forwards and backwards).
    """

    def __init__(self, seed=0):
        super().__init__(seed)

    @staticmethod
    def _next_state_generator(x, m, a, c):
        while True:
            x = (a*x + c) & (m - 1)
            yield x

    @staticmethod
    def _prev_state_generator(x, m, a_inverse, c):
        while True:
            x = a_inverse * (x - c) & (m - 1)
            yield x

    def random(self, size=None, update=True, increment=None):
        """Return random integers.
        
        Parameters
        ----------
        size : int
            Output size.  Default is None, in which case a single value is 
            returned.
        update : bool
            Determines whether the rng's state is updated after this operation
            or not. Default is True.
        increment : int
            Set to 1 to step fowards in the sequence and -1 to step backwards.
            (Other increments not supported).
    
        Returns
        -------
        out : uint64 or ndarray of uint64
            1-D array of random unsigned integers of size `size` (unless ``size=None``, in which
            case a single unsigned integer is returned).

        Example
        -------
        >>> rng = GeneratorLCGReversible(42)
        >>> rng.random(size=3)
        array([ 293047021,  968358053, 1773127077], dtype=uint64)
        >>> rng.random(size=3, increment=-1)
        array([968358053, 293047021,         0], dtype=uint64)
        """

        if increment is None:
            increment = 1 if self.forward else -1
        else:
            assert increment in (-1, 1), "`increment` must be -1, None, or 1."
            if update:
                self.forward = True if increment == 1 else False
        if size is None:
            if update:
                return self.__next__()
            if increment == 1:
                return self._next_state() >> self._d
            else:
                return self._prev_state() >> self._d
        else:
            if increment == 1:
                gen = self._next_state_generator(self.x, self.m, self.a, self.c)
            else:
                gen = self._prev_state_generator(self.x, self.m, self.a_inverse, self.c)
            assert np.ndim(size) == 0, "only 1-D arrays supported"
            x = np.fromiter(gen, dtype='uint64', count=size)
            if update:
                self._x = int(x[-1])
            return x >> self._d
