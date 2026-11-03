``f2py`` decodes ``str`` arrays passed to ``character`` arguments
-----------------------------------------------------------------
A NumPy array of dtype ``U`` passed to a scalar ``character`` or ``character(len=n)`` argument of an ``f2py`` wrapper is now read in the byte order of its dtype and encoded as ASCII, as a ``str`` argument is.
Previously its UCS4 data was copied as raw bytes: on big-endian platforms the argument arrived blank, and on little-endian platforms only its first character was passed.
Arrays holding non-ASCII characters are now rejected, like ``str`` arguments.
For ``intent(inout)`` arguments, the result is now written back to such arrays as characters rather than as raw bytes.
