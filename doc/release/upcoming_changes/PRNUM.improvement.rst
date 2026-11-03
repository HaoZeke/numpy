``f2py`` evaluates kinds and inquiries in parameters as compilers do
--------------------------------------------------------------------

``f2py`` now gives each value in a parameter expression the type and kind that Fortran gives it.
So ``kind(1.0) + kind(1)`` is 8, and ``kind(0.0_qp)`` is ``qp``.
A default real is single precision, which makes ``int(0.29 * 100)`` 29, as gfortran and LLVM ``flang`` evaluate it.
A parameter takes the type and kind of its declaration.
With no declaration, it takes its implicit type.
``real(8), parameter :: one = 1`` makes ``int(one / 2 * 10)`` equal 5.
That value used to be 0.
``parameter`` statements are evaluated by the same rules.
``f2py`` used to evaluate them as Python, so ``parameter (n = 7 / 2)`` made ``n`` 3.5.

``f2py`` now evaluates ``huge``, ``tiny``, ``epsilon``, ``digits`` and ``precision``.
It also evaluates ``range``, ``radix``, ``maxexponent``, ``minexponent``, ``bit_size`` and ``kind``.
The evaluation covers every integer, real and complex kind whose model ``f2py`` knows.
The model numbers come from Fortran 2018.
Real kinds come from the Institute of Electrical and Electronics Engineers (IEEE) 754 formats.
``real(10)`` is included only when C ``long double`` is the 80-bit x87 format.
Kind 16 is that standard's binary128 format.
Kind 16 is omitted for the International Business Machines (IBM) double-double format.
A binary128 ``long double`` has no kind 16 either.
On that ABI, gfortran can number the double-double format as kind 16.
These inquiries also work on a variable, as in ``huge(i)``.
``selected_real_kind`` now considers its range and radix and gives -1 to -5 as the standard says.
So ``selected_real_kind(6, 70)`` is 8.
It used to be 4.
``selected_char_kind`` is understood.
These functions are also evaluated in kinds, lengths and dimensions.
An argument declared ``real(kind=selected_real_kind(15))`` used to be passed as a C ``float``.
``f2py`` evaluates complex constants.
It also evaluates ``cmplx``, ``aimag``, ``conjg`` and the logical operators.

``f2py -h`` writes evaluated parameters, such as ``dp=8``.
It keeps the kind of a real that is not of default kind, as in ``0.1_8``.
It used to write that real as ``0.18``.
A kind that ``f2py`` cannot tell is left as written.
One example is a kind from ``iso_fortran_env``.
``f2py`` used to guess that kind to be 4 or 8.
The kinds ``f2py`` knows are the gfortran kinds it maps to C types, with the exceptions above.
LLVM ``flang`` also has ``real(2)`` and ``real(3)``, so its ``selected_real_kind(p)`` differs for ``p`` up to 3.
