``f2py`` warns when a scalar argument drops data
------------------------------------------------

Converting an ``f2py`` scalar argument from a sequence longer than one
element, from a complex value with a nonzero imaginary part, or from a
non-integral float into an integer argument, now emits a
``DeprecationWarning``. The value passed to Fortran is unchanged.
A longer sequence still contributes its first element.
A complex value still contributes its real part.
A non-integral float, including NaN and infinity, still follows the
existing conversion.
An integral float does not warn.
A one-element sequence and a complex value with a zero imaginary part
do not warn.
An empty sequence still fails.
If one of these warnings is turned into an error, conversion fails and
the error is raised.
A ``Warning`` raised as an error by the numeric conversion is no longer
cleared.
