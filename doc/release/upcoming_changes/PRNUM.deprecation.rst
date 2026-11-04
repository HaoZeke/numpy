``f2py`` warns when a scalar argument drops data
------------------------------------------------

An ``f2py`` scalar converter emits a ``DeprecationWarning`` for a sequence longer than one element, for a complex value with a nonzero imaginary part, and for a non-integral float passed to an integer argument.
``Fortran`` receives the same converted value the existing scalar converters produce.
The sequence path forwards the first element.
The complex path forwards the real part.
The integer path warns for NaN, infinity, and any float that differs from its truncation, then applies the existing conversion.
An integral float, a one-element sequence, and a complex value with a zero imaginary part take the quiet conversion path.
An empty sequence raises ``IndexError``.
Turning one of these warnings into an error aborts the conversion and raises that error.
A ``Warning`` raised as an error during numeric conversion stays set for the caller.
