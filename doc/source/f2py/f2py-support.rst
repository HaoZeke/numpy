.. _f2py-support:

========================
Fortran language support
========================

This page lists, standard by standard, the Fortran features that F2PY can
wrap for use from Python, together with the tests in ``numpy/f2py/tests``
that exercise them. The Fortran sources of those tests, in
``numpy/f2py/tests/src``, are small examples of each feature.

F2PY is not a Fortran compiler. It reads the interfaces of procedures (their
names, arguments and declarations) and the data in ``COMMON`` blocks and
modules, and generates wrappers around them. Executable statements are left
to the Fortran compiler, so constructs such as ``SELECT CASE``, ``WHERE`` or
``FORALL`` can be used freely inside wrapped procedures. The tables below list
the features that affect what F2PY wraps.

The status column uses three values:

Yes
   F2PY wraps the feature and the test suite covers it.
Partial
   The feature works with the restrictions given in the notes.
No
   F2PY cannot wrap the feature. Depending on the construct, wrapper generation
   fails, or the wrapper is built but does not work. Exclude such procedures
   with ``skip:`` (see :doc:`usage`), or call them from a Fortran procedure
   whose interface F2PY supports.

The notes link to open issues on the `NumPy issue tracker`_. If a feature is
missing here or does not behave as described, please open an issue with a
small Fortran file and the ``f2py`` command that shows the problem.

Fortran 77
==========

.. list-table::
   :header-rows: 1
   :widths: 30 9 43 18

   * - Feature
     - Status
     - Notes
     - Tests
   * - External subroutines and functions
     - Yes
     - Functions are called through generated Fortran subroutine wrappers,
       unless ``--no-wrap-functions`` is given.
     - ``test_return_*.py``
   * - Fixed-form source, comment lines and continuation lines
     - Yes
     -
     - ``test_regression.py``
   * - ``INTEGER``, ``REAL``, ``DOUBLE PRECISION``, ``COMPLEX``,
       ``DOUBLE COMPLEX``, ``LOGICAL`` and ``CHARACTER``, including the
       ``*n`` forms such as ``REAL*8``
     - Yes
     - See :doc:`python-usage` for the Python objects accepted for each type.
       For ``REAL*16`` see the extended precision row in the Fortran 90 table.
     - ``test_return_*.py``, ``test_array_from_pyobj.py``
   * - Explicit-shape, adjustable and assumed-size (``A(*)``) arrays, and
       explicit lower bounds
     - Yes
     - Arrays are indexed from 0 in Python whatever their Fortran bounds.
     - ``test_array_from_pyobj.py``, ``test_regression.py``
   * - ``CHARACTER*(*)`` arguments
     - Partial
     - For output arguments, give the length with ``f2py_len`` or in a
       signature file, see :ref:`Character strings`.
     - ``test_character.py``, ``test_string.py``
   * - ``COMMON`` blocks
     - Yes
     - Exposed as attributes of the extension module, see
       :doc:`python-usage`.
     - ``test_common.py``
   * - ``BLOCK DATA`` and ``DATA`` statements
     - Yes
     - The initial values are visible from Python.
     - ``test_data.py``
   * - ``PARAMETER`` constants
     - Yes
     - Can be used in array bounds and in other declarations.
     - ``test_parameter.py``, ``test_crackfortran.py``
   * - Implicit typing, default rules and ``IMPLICIT`` statements
     - Yes
     -
     - ``test_language_support.py``
   * - ``ENTRY`` statements
     - Yes
     - Each entry point is wrapped with the signature of its enclosing routine.
     - ``test_language_support.py``
   * - ``EXTERNAL`` dummy arguments
     - Yes
     - Python callables can be passed, see :ref:`Call-back arguments`.
     - ``test_callback.py``
   * - ``INCLUDE`` lines
     - Yes
     -
     - ``test_regression.py``
   * - Statement functions, ``EQUIVALENCE`` and ``SAVE``
     - Yes
     - They do not change the interface; F2PY ignores them and the compiler
       handles them.
     - ``test_language_support.py``
   * - Alternate returns (``*`` dummy arguments)
     - No
     - The ``*`` dummy argument becomes a required argument of the wrapper, and
       statement labels cannot be passed from Python.
     -

Fortran 90 and 95
=================

.. list-table::
   :header-rows: 1
   :widths: 30 9 43 18

   * - Feature
     - Status
     - Notes
     - Tests
   * - Free-form source and ``&`` continuation lines
     - Yes
     -
     - ``test_regression.py``
   * - Module procedures and module variables
     - Yes
     - Available as ``<extension module>.<module>.<name>``. Module variables
       can be read and assigned from Python.
     - ``test_modules.py``, ``test_regression.py``
   * - ``ALLOCATABLE`` module arrays
     - Yes
     - Assigning from Python allocates or reallocates the array, and assigning
       ``None`` deallocates it, see :doc:`python-usage`.
     - ``test_modules.py``
   * - Module ``PARAMETER`` constants
     - Partial
     - Exposed as module attributes. With LLVM ``flang`` the values are wrong,
       see `Compilers`_.
     - ``test_regression.py``
   * - ``USE`` association, ``PUBLIC`` and ``PRIVATE``
     - Yes
     - Private entities are not wrapped.
     - ``test_modules.py``, ``test_crackfortran.py``
   * - ``INTENT(IN)``, ``INTENT(OUT)`` and ``INTENT(INOUT)``
     - Yes
     - Read from the declarations. F2PY adds further intents, see
       :ref:`f2py-attributes`.
     - ``test_regression.py``
   * - Assumed-shape arrays (``A(:)``)
     - Yes
     - F2PY generates a Fortran wrapper with an explicit interface.
     - ``test_assumed_shape.py``
   * - Kind type parameters
     - Partial
     - Literal kinds and many named constants work. Kinds that F2PY cannot
       evaluate are mapped to the default C type unless they are given in an
       f2cmap file, see :ref:`f2py-kind-specifiers`.
     - ``test_kind.py``, ``test_parameter.py``, ``test_f2cmap.py``
   * - Extended precision (``REAL(10)``, ``REAL(16)``)
     - Partial
     - ``complex(c_long_double_complex)`` arrays work, but scalar ``REAL(10)``
       and ``REAL(16)`` arguments crash or give wrong results on x86-64 Linux
       (`gh-9727`_).
     - ``test_isoc.py``
   * - ``CHARACTER(LEN=n)``
     - Yes
     -
     - ``test_character.py``, ``test_string.py``
   * - ``OPTIONAL`` dummy arguments
     - Partial
     - Optional in Python, but an omitted argument is still passed to Fortran
       with a default value, so ``PRESENT()`` returns ``.TRUE.``
       (`gh-4013`_).
     - ``test_character.py``
   * - Functions with a ``RESULT`` clause
     - Yes
     -
     - ``test_crackfortran.py``
   * - ``RECURSIVE`` and ``PURE`` procedures
     - Yes
     -
     - ``test_language_support.py``
   * - ``ELEMENTAL`` procedures
     - Partial
     - Elemental functions are wrapped as scalar functions and do not
       broadcast over arrays. Elemental subroutines in modules, including
       Fortran 2008 ``IMPURE ELEMENTAL`` ones, fail to compile, because F2PY
       passes them as actual arguments when it sets up the module.
     - ``test_language_support.py``
   * - Array-valued functions
     - Partial
     - Functions returning numeric arrays work. Functions returning character
       arrays crash, and the corresponding test is skipped.
     - ``test_language_support.py``, ``test_character.py``
   * - Internal procedures (``CONTAINS`` in a procedure)
     - Partial
     - The host procedure is wrapped and its internal procedures are not
       exposed. If the host needs a Fortran wrapper, for example because it
       has assumed-shape arguments, the generated wrapper does not compile
       (`gh-20103`_).
     - ``test_language_support.py``
   * - Generic interfaces, defined operators and defined assignment
     - Partial
     - Each specific procedure is wrapped under its own name; the generic name
       is not exposed. :ref:`f2py-boilerplating` shows how to generate the
       bindings for many specific procedures.
     - ``test_language_support.py``, ``test_crackfortran.py``
   * - Derived types (``TYPE``)
     - No
     - Variables of derived type are not exposed, and a module that defines a
       derived type is not exposed at all when another wrapped procedure uses
       it. Wrapper generation fails for procedures with derived-type
       arguments, so exclude them with ``skip:`` (`gh-21160`_).
     - ``test_regression.py``
   * - ``ALLOCATABLE`` and ``POINTER`` dummy arguments
     - No
     - The generated wrappers do not compile (`gh-19157`_).
     -

Fortran 2003
============

.. list-table::
   :header-rows: 1
   :widths: 30 9 43 18

   * - Feature
     - Status
     - Notes
     - Tests
   * - ``ISO_C_BINDING`` kinds
     - Yes
     -
     - ``test_isoc.py``
   * - ``BIND(C)`` procedures, including ``BIND(C, NAME=...)``
     - Yes
     -
     - ``test_isoc.py``
   * - ``VALUE`` attribute
     - Yes
     -
     - ``test_value_attrspec.py``
   * - ``USE, INTRINSIC ::``
     - Yes
     -
     - ``test_f2cmap.py``
   * - ``ABSTRACT INTERFACE``, and ``PROCEDURE`` declarations of procedures
       that are not dummy arguments
     - Yes
     -
     - ``test_abstract_interface.py``
   * - ``PROCEDURE(interface)`` dummy arguments
     - No
     - Not recognized as call-back arguments; the implicit typing rules apply
       to the dummy argument instead. Declare call-back arguments with
       ``EXTERNAL``.
     -
   * - Enumerations (``ENUM, BIND(C)``)
     - Partial
     - The enumerators can be used in the Fortran code but are not exposed to
       Python.
     - ``test_language_support.py``
   * - ``PROTECTED`` module variables
     - Yes
     - Readable from Python. F2PY does not enforce ``PROTECTED``, so
       assignments from Python are not rejected.
     - ``test_language_support.py``
   * - ``BIND(C)`` module variables
     - Yes
     -
     - ``test_language_support.py``
   * - ``ASYNCHRONOUS`` and ``VOLATILE`` attributes
     - Yes
     -
     - ``test_language_support.py``
   * - Deferred-length character (``CHARACTER(LEN=:)``)
     - No
     - The generated wrappers do not compile.
     -
   * - ``C_PTR`` and ``C_FUNPTR`` arguments
     - No
     - Handled like derived types: wrapper generation fails, so exclude such
       procedures with ``skip:``.
     -
   * - ``CLASS``, type-bound procedures, type extension and parameterized
       derived types
     - No
     - ``CLASS(...)`` declarations are not recognized and the implicit typing
       rules apply to the dummy argument instead, so the wrapper cannot be
       called correctly. Exclude such procedures with ``skip:``
       (`gh-21160`_).
     -

Fortran 2008
============

.. list-table::
   :header-rows: 1
   :widths: 30 9 43 18

   * - Feature
     - Status
     - Notes
     - Tests
   * - ``ISO_FORTRAN_ENV`` kind constants
     - Partial
     - ``INT8``, ``INT16``, ``INT32``, ``INT64``, ``REAL32`` and ``REAL64``
       are recognized when imported under their own names. Renamed imports
       such as ``dp => real64`` are not (`gh-30352`_), and ``REAL128`` is not
       mapped.
     - ``test_isof.py``, ``test_f2cmap.py``
   * - ``CONTIGUOUS`` attribute
     - Yes
     -
     - ``test_language_support.py``
   * - ``BLOCK`` construct
     - Yes
     - Declarations inside the ``BLOCK`` do not become arguments.
     - ``test_crackfortran.py``
   * - ``DO CONCURRENT``
     - Yes
     -
     - ``test_language_support.py``
   * - Submodules and separate module procedures
     - No
     - F2PY does not recognize ``SUBMODULE`` or ``MODULE FUNCTION`` and
       ``MODULE SUBROUTINE`` interfaces. Wrapper generation fails, or the
       extension module crashes on import.
     -
   * - Coarrays
     - No
     - F2PY drops the codimension with a "could not crack entity declaration"
       message, so a coarray module variable is exposed as an ordinary
       variable.
     -

Fortran 2018
============

.. list-table::
   :header-rows: 1
   :widths: 30 9 43 18

   * - Feature
     - Status
     - Notes
     - Tests
   * - ``IMPLICIT NONE (TYPE, EXTERNAL)``
     - Yes
     -
     - ``test_language_support.py``
   * - ``NON_RECURSIVE`` prefix
     - Partial
     - Accepted on functions. F2PY stops with an error on a
       ``NON_RECURSIVE`` subroutine.
     -
   * - Assumed-type (``TYPE(*)``) and assumed-rank (``DIMENSION(..)``)
       dummy arguments
     - No
     - Wrapper generation fails for ``TYPE(*)``, and the C code generated for
       ``DIMENSION(..)`` does not compile.
     -

Compilers
=========

NumPy's continuous integration runs the F2PY test suite with ``gfortran``.
With recent LLVM ``flang`` releases and the meson backend, the following
problems are known:

* Extension modules are linked without the ``flang`` runtime library, so
  wrapped code that calls into it, for example for input and output, fails to
  import with an ``undefined symbol: _Fortran...`` error (`gh-30260`_,
  `gh-32148`_).
* Module ``PARAMETER`` constants read from Python have wrong values
  (`gh-22511`_).
* F2PY evaluates ``selected_real_kind`` itself, following the kinds of
  ``gfortran``. LLVM ``flang`` returns its half-precision kind for a precision
  of 3 or less, which F2PY maps to the wrong C type (`gh-30260`_).

.. _`NumPy issue tracker`: https://github.com/numpy/numpy/issues?q=is%3Aissue+is%3Aopen+label%3A%22component%3A+numpy.f2py%22
.. _`gh-4013`: https://github.com/numpy/numpy/issues/4013
.. _`gh-9727`: https://github.com/numpy/numpy/issues/9727
.. _`gh-19157`: https://github.com/numpy/numpy/issues/19157
.. _`gh-20103`: https://github.com/numpy/numpy/issues/20103
.. _`gh-21160`: https://github.com/numpy/numpy/issues/21160
.. _`gh-22511`: https://github.com/numpy/numpy/issues/22511
.. _`gh-30260`: https://github.com/numpy/numpy/issues/30260
.. _`gh-30352`: https://github.com/numpy/numpy/issues/30352
.. _`gh-32148`: https://github.com/numpy/numpy/issues/32148
