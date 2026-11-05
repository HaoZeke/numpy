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
      - Bounds are integer expressions of literals, lexical named constants,
        scalar integer arguments and integer array elements. Integer division
        truncates towards zero, and exponentiation retains Fortran grouping.
        A nonpositive extent produces a zero-length dimension. If an extent
        depends linearly on a single
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
      - ``y(size(x))``, ``y(size(x, dim=1))``
      - The wrapper converts ``x`` before evaluating the dependent bound.
        ``LBOUND`` and ``UBOUND`` accept a scalar ``DIM`` and retain the
        declared lower bound, including lower bounds of assumed-shape dummies.
        A zero extent has lower bound 1 and upper bound 0.
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

The same expression resolver handles a selector written directly in a
declaration, such as ``real(kind=kind(0.0d0))``, and constants declared in the
containing module. It follows host association, ``USE``, ``ONLY`` lists and
renamed imports through the supplied modules. A local declaration takes
precedence over an imported name or an intrinsic with the same spelling.
The original compiler's kind numbers still need a corresponding entry in
the C type map; ``.f2py_f2cmap`` describes an explicit mapping, see "Dealing
with KIND specifiers" in :doc:`advanced/use_cases`.

Named constants and initialization expressions
==============================================

F2PY needs scalar integer values for kinds, character lengths and constant
array bounds. With a selected compiler it evaluates these expressions by
compiling and running a small Fortran query. The query reconstructs only
the required typed ``PARAMETER`` declarations and intrinsic-module imports;
it does not compile or run the user's procedures. Consequently the compiler
defines the result of ``TINY``, ``HUGE``, ``EPSILON``, ``DIGITS``,
``SELECTED_REAL_KIND`` and other initialization-expression intrinsics, with
the selected compiler flags. Real and complex parameters retain their
declared types when they contribute to an integer expression. Inquiries into
a declared character length or an explicit array shape use the declaration
without allocating a character or array value of that size.

For a Meson build with ``-c``, F2PY uses ``FC`` when it is set and otherwise
asks Meson which Fortran compiler it selects. Set ``FFLAGS`` and the existing
``--f77flags``/``--f90flags`` options consistently with the source. Compiler
selection, the working directory and ``PATH`` participate in query caching.
A compiler, linker or target-execution error is an error; it does not select
a different processor's answer. A configured cross target needs an
executable wrapper. Generation without ``-c`` can use ``FC`` explicitly;
without a selected compiler, processor-dependent expressions remain in the
signature rather than receiving guessed kind numbers.

For example, a native build can wrap this declaration directly::

    module limits
      implicit none
      integer, parameter :: count = merge(7,0,tiny(1.0)>0.0)
    contains
      subroutine fill(values)
        integer, intent(out) :: values(count)
        values = 1
      end subroutine fill
    end module limits

``python -m numpy.f2py -c limits.f90 -m limits_ext --backend meson`` produces
``limits_ext.limits.fill()`` with shape ``(7,)``. Ordinary procedure bodies
can allocate and deallocate local arrays; that allocation is performed by
the original compiled procedure. It does not require translating executable
Fortran statements into C.

Bounds and character lengths that depend on call arguments are evaluated
at the call. Supported integer operations are ``+``, ``-``, ``*``, ``/`` and
``**``; integer ``ABS``, ``SIGN``, ``DIM``, ``MIN``, ``MAX``, ``MOD``,
``MODULO`` and ``MERGE``; scalar integer conversion; and ``LEN``,
``LEN_TRIM``, ``SIZE``, ``LBOUND`` and ``UBOUND`` inquiries on converted
arguments. Constant integer subexpressions, including processor inquiries,
are resolved through the compiler. The C calculation does not cancel
integer division or reassociate powers. The native Fortran wrapper keeps
the corresponding Fortran declaration instead of receiving C helper names.
An invalid runtime divisor, subscript, inquiry dimension or pointer-sized
integer result raises a Python exception before the native call.

These rules do not translate an arbitrary user-written specification
function into a C expression, or a runtime real expression into integer
shape arithmetic. Such interfaces need an explicit integer size argument
or a signature declaration whose size can be computed from the supported
integer arguments. Intrinsic calls in the executable body are handled by
the Fortran compiler without this interface restriction.

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
* A comment immediately preceding that wrapper records the original source
  file and its inclusive procedure line range. For example,
  ``/* Fortran source: 'limits.f90':5-8 */`` links the C wrapper to the
  declaration and body above. Continuation lines, included files and
  signature-file procedures retain their own source ranges. The C line
  number remains a generated-code line; the comment supplies the Fortran
  range without replacing the compiler's C location.
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
