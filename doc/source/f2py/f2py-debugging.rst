.. _f2py-debugging:

================================
Debugging F2PY extension modules
================================

When a wrapped routine gives wrong results, crashes, or does not build, look
at each stage that F2PY goes through: reading the Fortran sources, generating
the wrappers, building the extension module, and running it. The examples on
this page use ``fib1.f`` from :ref:`f2py-getting-started`.

Check what F2PY understood
==========================

``f2py -h stdout`` prints the signature that F2PY derived from the sources
without building anything:

.. code-block:: sh

    python -m numpy.f2py -h stdout fib1.f

Compare it with the Fortran code: a missing routine, a wrong type or an
argument with the wrong intent shows up here first. The messages printed while
F2PY reads the sources point to constructs it did not understand. For
example::

    getctype: "real(kind='dp')" is mapped to C "float" (to override define {'real': {'dp': "<C typespec>"}} in /path/to/.f2py_f2cmap file).

means that F2PY does not know the kind ``dp``, see
:ref:`f2py-kind-specifiers`. The docstring of each wrapper, for example
``print(fib1.fib.__doc__)``, shows how it is called from Python, and
:ref:`f2py-support` lists the Fortran constructs that F2PY can wrap.

Keep the generated files
========================

``f2py -c`` generates the C wrapper ``<modulename>module.c`` and, when needed,
the Fortran wrappers ``<modulename>-f2pywrappers.f`` and
``<modulename>-f2pywrappers2.f90``, writes a ``meson.build`` file for them,
and builds the extension module in a temporary directory. Pass
``--build-dir`` to choose that directory instead:

.. code-block:: sh

    python -m numpy.f2py -c fib1.f -m fib1 --build-dir build-fib1

``build-fib1`` then holds the generated sources and ``meson.build``, and
``build-fib1/bbdir`` is the meson build directory. In there,
``meson-logs/meson-log.txt`` records the compilers that meson found, and
``compile_commands.json`` lists the exact compile commands. To read the
wrappers without building them, run ``python -m numpy.f2py -m fib1 fib1.f``,
which writes them to the current directory.

Trace the wrappers at run time
==============================

With ``--debug-capi``, every call of a wrapper prints the arguments it
received, the checks it made and the Fortran routine it called:

.. code-block:: sh

    python -m numpy.f2py -c fib1.f -m fib1 --debug-capi

.. code-block:: python

    >>> import numpy as np
    >>> import fib1
    >>> fib1.fib(np.zeros(4))
    debug-capi:Python C/API function fib1.fib(a,n=shape(a, 0))
    debug-capi:double a=:input,required,array,dims(n|n)
    debug-capi:int n=shape(a, 0):input,optional,scalar
    debug-capi:n=4
    debug-capi:Checking `shape(a, 0) == n'
    debug-capi:Fortran subroutine `fib(a,&n)'
    debug-capi:n=4
    debug-capi:Building return value.
    debug-capi:Python C/API function fib1.fib: successful.
    debug-capi:Freeing memory.

To see when F2PY copies an array argument, for example because the array is
not Fortran-contiguous or has another data type, define the
``F2PY_REPORT_ON_ARRAY_COPY`` macro to the array size above which copies are
reported. Meson reads compiler flags from the ``CFLAGS``, ``FFLAGS`` and
``LDFLAGS`` environment variables when it sets up the build:

.. code-block:: sh

    CFLAGS="-DF2PY_REPORT_ON_ARRAY_COPY=1" python -m numpy.f2py -c fib1.f -m fib1

.. code-block:: python

    >>> import numpy as np
    >>> import fib1
    >>> fib1.fib(np.ones(8, 'i'))
    copied an array: size=8, elsize=8

.. note::

   The meson backend does not pass the ``-D`` command line options on to the
   compiler (`gh-28648`_), and it ignores ``--debug`` (`gh-30353`_), so set
   such flags through the environment variables instead.

Debug the compiled code
=======================

To use a debugger, build without optimization and with debugging information
for both the C wrapper and the Fortran code. ``--f77flags`` and
``--f90flags`` only reach the Fortran compiler, so use the environment
variables:

.. code-block:: sh

    CFLAGS="-g -O0" FFLAGS="-g -O0" python -m numpy.f2py -c fib1.f -m fib1

Then run Python under the debugger. The extension module is only loaded when
it is imported, so let ``gdb`` keep the breakpoints pending until then. The
wrapper for ``fib`` is the C function ``f2py_rout_fib1_fib`` and the Fortran
routine is ``fib_``:

.. code-block:: sh

    gdb -ex "set breakpoint pending on" -ex "break f2py_rout_fib1_fib" \
        -ex "break fib_" -ex run \
        --args python -c "import numpy as np, fib1; fib1.fib(np.zeros(8))"

Compilers can also check array bounds at run time. An out-of-range access in
the Fortran code then stops the Python process with an error message, instead
of silently overwriting memory. With ``gfortran``:

.. code-block:: sh

    FFLAGS="-g -fcheck=bounds" python -m numpy.f2py -c fib1.f -m fib1

See :ref:`debugging` for more on using ``gdb`` with Python, and
:ref:`advanced_debugging` for tools such as ``valgrind`` and sanitizers,
which work for F2PY extension modules as well.

Common problems
===============

``ImportError: ... undefined symbol: fib_``
   The extension module was linked without the compiled Fortran routine.
   Pass all the Fortran sources, object files or libraries that define the
   wrapped routines to ``f2py -c``, also when building from a signature file.

``ImportError: ... undefined symbol: _Fortran...``
   The runtime library of LLVM ``flang`` is missing, see the
   :ref:`known problems with LLVM flang <f2py-compilers>`.

Results are wrong, but there is no error
   Look for ``getctype`` messages: a kind that F2PY could not map passes data
   of the wrong size, see :ref:`f2py-kind-specifiers`. Also check the
   ``intent`` of the arguments in the wrapper's docstring. An input array that
   does not have the type and memory layout the Fortran routine expects is
   passed as a copy, so changes made by Fortran are not visible in Python
   unless the argument is ``intent(out)`` or ``intent(inout)``, see
   :ref:`f2py-getting-started`.

.. _`gh-28648`: https://github.com/numpy/numpy/issues/28648
.. _`gh-30353`: https://github.com/numpy/numpy/issues/30353
