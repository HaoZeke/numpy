.. _f2py-win-conda:

=========================
F2PY and Conda on Windows
=========================

The `conda-forge`_ channel provides two Fortran toolchains for Windows:

- LLVM Flang (``flang_win-64``), its default Fortran compiler on Windows, which
  pairs with MSVC, and
- the Mingw-w64 ``gcc`` and ``gfortran`` compilers, built against the Universal
  C Runtime.

The two are not ABI compatible with each other, so use one of them per
environment. MSVC cannot be redistributed through conda, which makes the
Mingw-w64 compilers the simpler choice for ``f2py -c``: conda installs the whole
toolchain.

Install `Miniforge`_, which uses conda-forge by default, and create an
environment with the compilers:

.. code-block:: powershell

   conda create -n f2py python numpy meson ninja gcc gfortran
   conda activate f2py
   python -m numpy.f2py -c fib1.f -m fib1
   python -c "import fib1, numpy as np; a = np.zeros(8); fib1.fib(a); print(a)"

In a shell where MSVC is also active, ``meson`` picks ``cl`` first, so set
``CC`` and ``FC`` as shown in :ref:`f2py-win-compilers`.

The compiler runtime libraries, such as ``libgfortran-5.dll``, are installed
into the environment's ``Library\bin`` directory. conda-forge's Python
registers that directory for DLL loading, so the module imports inside the
environment without :func:`os.add_dll_directory`. Elsewhere, it needs those
libraries made available as described in :ref:`f2py-win-dlls`.

.. _conda-forge: https://conda-forge.org/docs/maintainer/infrastructure/#compilers-supplied-by-conda-forge
.. _Miniforge: https://conda-forge.org/download/
