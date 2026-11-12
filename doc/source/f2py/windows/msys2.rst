.. _f2py-win-msys2:

===========================
F2PY and Windows with MSYS2
===========================

Follow the standard `installation instructions`_ and use the ``UCRT64``
`environment`_, which MSYS2 recommends by default. It builds against the
Universal C Runtime (UCRT), the same C runtime the python.org builds of CPython
use. The older ``MINGW64`` environment links against ``msvcrt.dll`` instead and
is deprecated.

From a ``UCRT64`` shell, grab the requisite compilers:

.. code-block:: bash

   pacman -Syu   # repeat after restarting the shell if pacman asks for it
   pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-gcc-fortran

The compilers can then be used in two ways.

Everything inside MSYS2
=======================

MSYS2 also packages Python, NumPy, ``meson`` and ``ninja`` for ``UCRT64``:

.. code-block:: bash

   pacman -S --needed mingw-w64-ucrt-x86_64-python-numpy \
       mingw-w64-ucrt-x86_64-meson mingw-w64-ucrt-x86_64-ninja
   python -m numpy.f2py -c fib1.f -m fib1
   python -c "import fib1, numpy as np; a = np.zeros(8); fib1.fib(a); print(a)"

The MinGW runtime libraries live next to this ``python.exe`` in
``C:\msys64\ucrt64\bin``, so the module imports without further steps.

MSYS2 compilers with python.org Python
======================================

To build modules for a python.org CPython, for example from the virtual
environment of the :ref:`baseline <f2py-win-baseline>`, append the ``UCRT64``
directory to ``PATH`` and select the compilers:

.. code-block:: powershell

   $env:Path += ";C:\msys64\ucrt64\bin"
   $env:CC = 'gcc'
   $env:FC = 'gfortran'
   python -m numpy.f2py -c fib1.f -m fib1

The module depends on the MinGW runtime libraries ``libgfortran-5.dll``,
``libquadmath-0.dll``, ``libgcc_s_seh-1.dll`` and ``libwinpthread-1.dll``.
Python does not search ``PATH`` for them (see :ref:`f2py-win-dlls`), so the
import fails with ``ImportError: DLL load failed`` until one of the following
is done:

- register the directory before the import:

  .. code-block:: python

     import os
     os.add_dll_directory(r"C:\msys64\ucrt64\bin")
     import fib1

- copy the four libraries next to the built module, or,
- link them statically by setting ``LDFLAGS`` for the build:

  .. code-block:: powershell

     $env:LDFLAGS = '-static'
     python -m numpy.f2py -c fib1.f -m fib1

  The resulting module only depends on the Python and Windows system libraries.
  ``--f90flags`` only reaches the compiler, not the linker, and
  ``-static-libgfortran`` has no effect because ``meson`` links through ``gcc``
  and adds ``-lgfortran`` itself.

Standalone Mingw-w64 distributions, such as the UCRT builds from `WinLibs`_,
are used the same way.

.. _`installation instructions`: https://www.msys2.org/
.. _environment: https://www.msys2.org/docs/environments/
.. _WinLibs: https://winlibs.com/
