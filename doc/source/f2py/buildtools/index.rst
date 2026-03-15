.. _f2py-bldsys:

=======================
F2PY and build systems
=======================

This section covers popular build systems and their usage with ``f2py``.

.. versionchanged:: NumPy 1.26.x

   The default build system for ``f2py`` was historically the enhanced
   ``numpy.distutils`` module, which is based on ``distutils``.  ``distutils``
   was removed in ``NumPy 2.5.0`` (June 2026). Like the rest of NumPy and
   SciPy, ``f2py`` now uses ``meson``; see :ref:`distutils-status-migration`
   for details.

   All changes to ``f2py`` are tested on SciPy, so their `CI configuration`_ is
   always supported.


.. note::

   See :ref:`f2py-meson-distutils` for migration information.


Basic concepts
===============

Building an extension module which includes Python and Fortran consists of:

- Fortran source(s)
- One or more generated files from ``f2py``

  + A ``C`` wrapper file is always created
  + Code with modules require an additional ``.f90`` wrapper
  + Code with functions generate an additional ``.f`` wrapper

- ``fortranobject.{c,h}``

  + Distributed with ``numpy``
  + Can be queried via ``python -c "import numpy.f2py; print(numpy.f2py.get_include())"``

- NumPy headers

  + Can be queried via ``python -c "import numpy; print(numpy.get_include())"``

- Python libraries and development headers

Three cases arise when considering the outputs of ``f2py``:

Fortran 77 programs
   - Input file ``blah.f``
   - Generates

     + ``blahmodule.c``
     + ``blah-f2pywrappers.f``

   When no ``COMMON`` blocks are present only a ``C`` wrapper file is generated.
   Wrappers are also generated to rewrite assumed shape arrays as automatic
   arrays.

Fortran 90 programs
   - Input file ``blah.f90``
   - Generates:

     + ``blahmodule.c``
     + ``blah-f2pywrappers.f``
     + ``blah-f2pywrappers2.f90``

   The ``f90`` wrapper is used to handle code which is subdivided into
   modules. The ``f`` wrapper makes ``subroutines`` for  ``functions``. It
   rewrites assumed shape arrays as automatic arrays.

Signature files
   - Input file ``blah.pyf``
   - Generates:

     + ``blahmodule.c``
     + ``blah-f2pywrappers2.f90`` (occasionally)
     + ``blah-f2pywrappers.f`` (occasionally)

   Signature files (``.pyf``) do not signal their language standard via the
   file extension. They may generate F90 and F77 specific wrappers depending on
   their contents, which shifts the burden of checking for generated files onto
   the build system.

.. versionchanged:: NumPy ``1.22.4``

   ``f2py`` will deterministically generate wrapper files based on the input
   file Fortran standard (F77 or greater).  ``--skip-empty-wrappers`` can be
   passed to ``f2py`` to restore the previous behaviour of only generating
   wrappers when needed by the input.


In principle, any build system can be adapted to generate ``f2py`` extension
modules given the requirements above. The following sections cover the most
popular systems.

.. note::
   ``make`` has no place in a modern multi-language setup, and so is not
   discussed further.

Build systems
==============

.. toctree::
   :maxdepth: 2

   meson
   meson-python
   cmake
   skbuild
   distutils-to-meson

.. _`issue 20385`: https://github.com/numpy/numpy/issues/20385

.. _`CI configuration`: https://docs.scipy.org/doc/scipy/dev/toolchain.html#official-builds
