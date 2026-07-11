.. _f2py-win-msys2:

===========================
F2PY and Windows with MSYS2
===========================

Follow the standard `installation instructions`_. Then, to grab the requisite Fortran compiler with ``MVSC``:

.. code-block:: bash

   # Assuming a fresh install
   pacman -Syu # Restart the terminal
   pacman -Su  # Update packages
   # Get the toolchains
   pacman -S --needed base-devel gcc-fortran
   pacman -S mingw-w64-x86_64-toolchain

Importing MinGW-built modules
=============================

A module built with the MinGW ``gfortran`` links against the MinGW
runtime libraries (``libgfortran``, ``libgcc_s``, ``libwinpthread``).
Python 3.8 and later do not consult ``PATH`` when resolving the
dependent DLLs of an extension module, so importing the module from a
regular (non-MSYS2) Python fails with ``ImportError: DLL load failed``
even when the compiler directory is on ``PATH``. Register the runtime
directory explicitly before the import:

.. code-block:: python

   import os
   os.add_dll_directory(r"C:\msys64\ucrt64\bin")  # or mingw64\bin
   import mymodule

Alternatively, try linking the runtimes statically by passing
``-static-libgfortran -static-libgcc -static-libwinpthread`` through
``--f90flags``, or copy the three runtime libraries next to the built
module. Static ``libwinpthread`` support is not available in every
MinGW layout; when it is not, prefer ``os.add_dll_directory`` (or copy
the libraries next to the module).

.. _`installation instructions`: https://www.msys2.org/
