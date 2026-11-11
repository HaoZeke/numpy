.. _f2py-fortran-support:

=========================
Fortran 90 and 95 support
=========================

F2PY reads Fortran 77, 90 and 95 source in fixed and free form. Fortran 77
code is essentially fully supported. This page describes which Fortran 90/95
constructs F2PY can wrap, how it treats them, and what to do when a construct
is not supported.

What F2PY reads
===============

F2PY does not compile Fortran. It reads the *declarations* of the procedures
and module data that it wraps, generates a C extension module and, when
needed, Fortran wrapper routines, and leaves the original source to the
Fortran compiler. As a consequence:

* Executable statements are not analyzed. Anything the Fortran compiler
  accepts may appear in the body of a procedure: allocation, array syntax,
  intrinsic procedures, local derived types and so on.
* F2PY must understand the interface of each procedure that it wraps: the
  type, kind, rank, shape and intent of every dummy argument, the function
  result, and the named constants used in those declarations.
* Procedures that cannot be wrapped can be left out with ``skip:``, or the
  wanted ones selected with ``only:``, see :doc:`usage`. Any declaration can
  also be corrected in a signature file, see :doc:`signature-file`.

Program units
=============

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Construct
     - Status
     - Notes
   * - External subroutines and functions
     - Wrapped
     -
   * - Module procedures
     - Wrapped
     - Available as ``<extension>.<module>.<procedure>``.
   * - Module variables
     - Wrapped
     - Scalars, explicit-shape arrays, allocatable arrays and strings can be
       read and set from Python, see :doc:`python-usage`.
   * - Named constants (``PARAMETER``) in modules
     - Read only
     - Exposed as module attributes. The values come from the compiled
       module. Do not assign to them.
   * - Generic interfaces, defined operators
     - Partial
     - Each specific procedure is wrapped under its own name. The generic
       name or the operator is not.
   * - Abstract interfaces
     - Parsed
     - Modules that contain them are wrapped as usual. A dummy argument
       declared with ``PROCEDURE(name)`` is not recognized as a call-back,
       see `Dummy arguments`_.
   * - ``ELEMENTAL``, ``PURE`` and ``RECURSIVE`` procedures
     - Wrapped
     - ``ELEMENTAL`` procedures take scalar arguments in Python; F2PY does
       not broadcast them over arrays.
   * - Internal procedures (``CONTAINS`` inside a procedure)
     - Not wrapped
     -
   * - ``PRIVATE`` procedures and variables
     - Not wrapped
     -
   * - Derived types (``TYPE``)
     - Not supported
     - Type definitions are skipped. F2PY stops with an error on a procedure
       that has a ``TYPE(...)`` dummy argument; leave such procedures out with
       ``skip:``.

Dummy arguments
===============

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - Behavior
   * - ``INTENT(IN)``, ``INTENT(INOUT)``
     - Arguments of the Python function.
   * - ``INTENT(OUT)``
     - Returned from the Python function. Arrays need an extent that F2PY
       can compute from the other arguments, see `Array declarations`_.
   * - ``OPTIONAL``
     - Optional in Python. When it is omitted, F2PY still passes a value:
       zero, or the default given in a signature file. ``PRESENT`` is
       therefore true inside the procedure.
   * - ``VALUE``
     - Supported.
   * - ``EXTERNAL``
     - A call-back argument, see :ref:`Call-back arguments`. A dummy
       procedure declared with ``PROCEDURE(name)`` is taken for a scalar
       instead; declare it ``EXTERNAL``.
   * - ``ALLOCATABLE``, ``POINTER``
     - Not supported for dummy arguments: the generated Fortran wrapper does
       not compile. Allocate the array in Python and pass it as an
       explicit-shape or assumed-shape argument, or use an allocatable module
       variable.

Array declarations
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Kind of array
     - Examples
     - Notes
   * - Explicit shape
     - ``x(n)``, ``x(0:n)``, ``x(-n:n)``, ``x(2*n+1, m)``
     - Bounds are integer expressions of literals, named constants and
       scalar integer arguments. If an extent depends linearly on a single
       argument, that argument becomes optional and F2PY computes it from the
       shape of the array; ``x(0:n)`` gives ``n = len(x) - 1``. Otherwise the
       argument is required and F2PY checks that the shapes agree.
   * - Assumed shape
     - ``x(:)``, ``x(0:)``, ``x(n:)``, ``x(0:, 2:)``
     - The procedure sees the lower bounds it declares. An ``INTENT(OUT)``
       assumed-shape array has no extent that F2PY could use: declare it
       ``INTENT(INOUT)`` or give it an explicit shape.
   * - Assumed size
     - ``x(*)``, ``x(0:*)``
     - The array is passed as given; pass its size in another argument.
   * - Bounds that use ``SIZE``
     - ``y(size(x))``, ``y(size(x, 1))``
     - ``x`` has to come before ``y`` in the argument list.
   * - Deferred shape
     - ``real, allocatable :: x(:)``
     - Allocatable module variables only, see :doc:`python-usage`. Module
       ``POINTER`` arrays are not supported.

Types and kinds
---------------

``INTEGER``, ``REAL``, ``DOUBLE PRECISION``, ``COMPLEX``, ``DOUBLE COMPLEX``,
``LOGICAL`` and ``CHARACTER`` are supported. Strings of assumed length,
``character(len=*)``, are described in :ref:`Character strings`.

F2PY translates a kind to a C type in these cases:

* a literal kind, such as ``integer(8)`` or ``real(4)``;
* a named constant defined with ``kind``, ``selected_int_kind`` or
  ``selected_real_kind``, declared in the same procedure or in a module that
  the procedure or its module imports with ``USE``. The used module has to be
  among the files given to F2PY;
* the kinds of the ``iso_c_binding`` and ``iso_fortran_env`` intrinsic
  modules.

F2PY does not resolve a kind expression written in the declaration itself,
such as ``real(kind=kind(0.0d0))``, nor a named constant declared in the module
that contains the procedure. In these cases it warns (``getctype``) and uses
the C type of the default kind, which does not match the data. Move the
constant to a separate module, or map the kind to a C type in a
``.f2py_f2cmap`` file, see "Dealing with KIND specifiers" in
:doc:`advanced/use_cases`.

Named constants and initialization expressions
==============================================

F2PY evaluates a named constant only where it needs a number: in the bounds,
kinds and lengths of the declarations that it wraps. It can evaluate integer
and real literals, arithmetic, references to other named constants, and the
intrinsics ``kind``, ``selected_int_kind`` and ``selected_real_kind``. Other
intrinsic functions, for example ``tiny``, ``huge``, ``epsilon`` or
``digits``, are not evaluated. Named constants that use them work everywhere
else, including module constants read from Python, whose values the Fortran
compiler computes. Do not use them in the bounds or the kind of a wrapped
argument; write a literal there, if necessary in a signature file.

As for kinds, a named constant declared in the module that contains a
procedure is not known to F2PY when it wraps that procedure. Used in a bound
or a length, it makes the generated C code fail to compile. Declare such
constants in a separate module and import it with ``USE``.

Array constructors in declarations have to use the Fortran 95 form
``(/ ... /)``; the Fortran 2003 form ``[ ... ]`` is not understood there.

Finding the Fortran source of an error
======================================

When the generated code does not compile, these help to find the Fortran
declaration that caused it:

* Every wrapped procedure has a C function named
  ``f2py_rout_<extension>_<procedure>``, or
  ``f2py_rout_<extension>_<module>_<procedure>`` for module procedures,
  between the banners ``/*** <procedure> ***/`` and
  ``/*** end of <procedure> ***/``. C compiler errors name this function.
* ``f2py -h <file>.pyf <sources>`` writes the signatures as F2PY understood
  them. Each procedure is annotated with the file and module it comes from,
  for example ``subroutine ortprj(...) ! in :aov:aov.f90:aov``. Compare it
  with the Fortran declarations, correct the signature file if needed, and
  build from it.
* ``--build-dir <dir>`` keeps the generated C and Fortran files, which
  otherwise go to a temporary directory.
* While reading the declarations F2PY prints the scope of its messages on a
  line of the form ``In: :<extension>:<file>:<module>:<procedure>``.
* ``--debug-capi`` builds a module that reports what the wrapper does at run
  time.
