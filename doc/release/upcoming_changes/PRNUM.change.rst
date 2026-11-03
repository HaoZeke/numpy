``f2py`` converts ``str_`` arrays passed as ``character`` arguments
-------------------------------------------------------------------
A `numpy.str_` array passed to a scalar ``character`` or
``character(len=n)`` argument of an ``f2py``-wrapped routine is now
converted with NumPy's ASCII cast, as for character array arguments.
Previously ``f2py`` read the UCS4 buffer as bytes, so little-endian
systems passed only the first character, big-endian systems passed an
empty string, and ``intent(inout)`` results corrupted the array.
Non-ASCII characters now raise an error, as for `str` arguments.
