.. _f2py-getting-started:

======================================
 Three ways to wrap - getting started
======================================

Wrapping Fortran or C functions to Python using F2PY has three steps:

* Create a :doc:`signature file <signature-file>` that describes wrappers to
  Fortran or C functions. For Fortran routines, F2PY can generate an initial
  signature file by scanning source code and extracting the information needed
  for wrapper functions.

  * Optionally, edit the generated signature file to refine wrapper behavior
    and produce a more Pythonic interface.

* F2PY reads the signature file and writes a Python C/API module containing
  Fortran/C/Python bindings.

* F2PY compiles all sources and builds an extension module containing
  the wrappers.

  * F2PY uses ``meson`` as the build backend. For other build systems,
    see :ref:`f2py-bldsys`.


.. note::

  * Depending on your operating system, you may need to install the Python
    development headers (which provide ``Python.h``) separately. On
    Debian-based distributions the package is ``python3-dev``; on Fedora-based
    distributions it is ``python3-devel``. On macOS the availability depends on
    how Python was installed. On Windows the headers are typically included;
    see :ref:`f2py-windows`.

.. note::

   F2PY supports all the operating systems SciPy is tested on so their
   `system dependencies panel`_ is a good reference.

These steps can run as a single composite command or one at a time.

The three approaches below use Fortran 77 and progress from least to most
control. Which one to choose depends on whether you can modify the Fortran
source.

Save the following example as ``fib1.f``:

.. literalinclude:: ./code/fib1.f
   :language: fortran

.. note::

  F2PY parses Fortran/C signatures to build wrapper functions for Python.
  It is not a compiler: it does not check for errors in the source code and
  does not implement the full language standards. Some errors may pass silently
  (or as warnings) and must be verified separately.

The quick way
==============

The quickest way to wrap the Fortran subroutine ``FIB`` for use in Python is to
run

::

  python -m numpy.f2py -c fib1.f -m fib1

or, alternatively, if the ``f2py`` command-line tool is available,

::

  f2py -c fib1.f -m fib1

.. note::

  The ``f2py`` command is not available on all systems (notably Windows).
  This guide uses ``python -m numpy.f2py`` throughout.

This compiles and wraps ``fib1.f`` (``-c``) into the extension module
``fib1.so`` (``-m``) in the current directory. Run ``python -m numpy.f2py``
with no arguments to see all command-line options. The Fortran subroutine
``FIB`` is now accessible in Python as ``fib1.fib``::

  >>> import numpy as np
  >>> import fib1
  >>> print(fib1.fib.__doc__)
  fib(a,[n])

  Wrapper for ``fib``.

  Parameters
  ----------
  a : input rank-1 array('d') with bounds (n)

  Other parameters
  ----------------
  n : input int, optional
      Default: len(a)

  >>> a = np.zeros(8, 'd')
  >>> fib1.fib(a)
  >>> print(a)
  [  0.   1.   1.   2.   3.   5.   8.  13.]

.. note::

  * F2PY recognized that the second argument ``n`` is the dimension of
    the first array argument ``a``. Since all arguments default to
    input-only, F2PY makes ``n`` optional with default value ``len(a)``.

  * One can use different values for optional ``n``::

      >>> a1 = np.zeros(8, 'd')
      >>> fib1.fib(a1, 6)
      >>> print(a1)
      [ 0.  1.  1.  2.  3.  5.  0.  0.]

    but an exception is raised when it is incompatible with the input
    array ``a``::

      >>> fib1.fib(a, 10)
      Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
      fib.error: (len(a)>=n) failed for 1st keyword n: fib:n=10
      >>>

    F2PY implements basic compatibility checks between related
    arguments in order to avoid unexpected crashes.

  * When a NumPy array is :term:`Fortran <Fortran order>` :term:`contiguous`
    and has a ``dtype`` matching the expected Fortran type, its C pointer is
    passed directly to Fortran.

    Otherwise, F2PY makes a contiguous copy (with the proper ``dtype``) and
    passes the copy's C pointer to the Fortran subroutine. Changes to this
    copy have no effect on the original argument::

      >>> a = np.ones(8, 'i')
      >>> fib1.fib(a)
      >>> print(a)
      [1 1 1 1 1 1 1 1]

    This is unexpected because Fortran passes by reference. The fact that the
    earlier ``dtype=float`` example worked is accidental.

    F2PY provides an ``intent(inplace)`` attribute that copies changes back
    to the input argument. With ``intent(inplace) a`` (see :ref:`f2py-attributes`),
    the example above would produce::

      >>> a = np.ones(8, 'i')
      >>> fib1.fib(a)
      >>> print(a)
      [  0.   1.   1.   2.   3.   5.   8.  13.]

    The recommended approach is ``intent(out)``, which is both more efficient
    and cleaner.

  * The usage of ``fib1.fib`` in Python resembles using ``FIB`` in Fortran.
    However, *in situ* output arguments are poor style in Python: there are no
    compile-time type checks, so type mismatches surface only at runtime. This
    can produce hard-to-find bugs and requires verbose runtime type checking.

  This approach is straightforward but limited: F2PY cannot determine argument
  intent on its own and treats all arguments as inputs by default. The next two
  approaches remove this ambiguity by declaring argument intent explicitly,
  producing wrappers that are easier to use and less error-prone.

The smart way
==============

For more control over the generated interface, apply the wrapping steps one
at a time.

* First, we create a signature file from ``fib1.f`` by running:

  ::

    python -m numpy.f2py fib1.f -m fib2 -h fib1.pyf

  The signature file is saved to ``fib1.pyf`` (see the ``-h`` flag) and its
  contents are shown below.

  .. literalinclude:: ./code/fib1.pyf
     :language: fortran

* Next, edit the signature file to declare that ``n`` is an input argument
  (``intent(in)``), that ``a`` should be returned to Python (``intent(out)``),
  and that ``a`` depends on ``n`` for its size (``depend(n)``).

  Save this modified version as ``fib2.pyf``:

  .. literalinclude:: ./code/fib2.pyf
     :language: fortran

* Finally, we build the extension module by running:

  ::

    python -m numpy.f2py -c fib2.pyf fib1.f

In Python::

  >>> import fib2
  >>> print(fib2.fib.__doc__)
  a = fib(n)

  Wrapper for ``fib``.

  Parameters
  ----------
  n : input int

  Returns
  -------
  a : rank-1 array('d') with bounds (n)

  >>> print(fib2.fib(8))
  [  0.   1.   1.   2.   3.   5.   8.  13.]

.. note::

  * The signature of ``fib2.fib`` now matches the intent of the Fortran
    subroutine ``FIB``: given ``n``, it returns the first ``n`` Fibonacci
    numbers as a NumPy array. The unexpected behavior from ``fib1.fib`` is
    eliminated.

  * By default, ``intent(out)`` implies ``intent(hide)``. Hidden arguments
    do not appear in the wrapper function's argument list.

  For more details, see :doc:`signature-file`.

The quick and smart way
========================

The "smart way" works well for third-party Fortran code that cannot be
modified. When the source is editable, the intermediate signature file can be
skipped entirely. F2PY-specific attributes go directly into the Fortran source
as special comment lines (starting with ``Cf2py`` or ``!f2py``) that Fortran
compilers ignore but F2PY interprets.

Save the following modified version as ``fib3.f``:

.. literalinclude:: ./code/fib3.f
   :language: fortran

Build the extension module in one command::

  python -m numpy.f2py -c -m fib3 fib3.f

Notice that the resulting wrapper to ``FIB`` is as "smart" (unambiguous) as in
the previous case::

  >>> import fib3
  >>> print(fib3.fib.__doc__)
  a = fib(n)

  Wrapper for ``fib``.

  Parameters
  ----------
  n : input int

  Returns
  -------
  a : rank-1 array('d') with bounds (n)

  >>> print(fib3.fib(8))
  [  0.   1.   1.   2.   3.   5.   8.  13.]

.. _`system dependencies panel`: https://scipy.github.io/devdocs/building/index.html#system-level-dependencies
