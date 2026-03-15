.. _f2py-meson-python:

=====================================================
Distributing F2PY extensions with ``meson-python``
=====================================================

This tutorial produces an installable Python wheel from a Fortran subroutine
using `meson-python <https://meson-python.readthedocs.io/>`_ as the build
backend.  At the end you will have:

.. code-block:: python

   >>> from fib_wrapper import fib
   >>> fib(10)
   array([ 0,  1,  1,  2,  3,  5,  8, 13, 21, 34], dtype=int32)

``meson-python`` is the recommended way to distribute F2PY extensions.
NumPy and SciPy use the same build system.
See :ref:`distutils-status-migration` for migration background.

Prerequisites
=============

* A C compiler
* A Fortran compiler (``gfortran``, ``ifort``, ``ifx``, ``flang-new``, etc.),
  if your package includes Fortran source
* Python >= 3.10
* ``meson``, ``meson-python``, and ``numpy`` (pulled in automatically during
  the build via ``build-system.requires``)

Project layout
==============

Create the following four files::

    fib_project/           # project root
    ├── fib.f90            # Fortran source
    ├── fib_wrapper/       # Python package directory
    │   └── __init__.py
    ├── meson.build
    └── pyproject.toml

Fortran source (``fib.f90``)
-----------------------------

.. literalinclude:: ../code/fib_mesonpy.f90
   :language: fortran

``pyproject.toml``
------------------

.. literalinclude:: ../code/pyproj_mesonpy.toml
   :language: toml

``build-backend = "mesonpy"`` selects ``meson-python`` as the PEP 517 backend.
``numpy >= 2.0`` is required because ``dependency('numpy')`` support in Meson
and the current ``f2py`` code generation both need it.

``meson.build``
---------------

.. literalinclude:: ../code/meson_mesonpy.build

.. note::

   This file is stored as ``meson_mesonpy.build`` in the documentation source
   tree to avoid collisions with other examples.  In your project, name it
   ``meson.build``.

The file:

1. Locates NumPy headers via ``dependency('numpy')`` and adds the F2PY include
   directory (``fortranobject.h``) with ``declare_dependency``.
2. Runs ``f2py`` via ``custom_target`` to generate C wrapper sources.
3. Compiles the generated C code together with the Fortran source into a Python
   extension module (``py.extension_module``).
4. Installs ``__init__.py`` into the package directory.

The ``subdir: 'fib_wrapper'`` argument places the compiled extension inside the
``fib_wrapper/`` package directory, next to ``__init__.py``.  Without it the
extension lands at the top level and ``import fib_wrapper`` cannot find it.
The installed layout::

    site-packages/
    └── fib_wrapper/
        ├── __init__.py        # from .fib import fib
        └── fib.cpython-*.so   # compiled extension module

``fib_wrapper/__init__.py``
----------------------------

.. code-block:: python

   from .fib import fib

Build the wheel
===============

.. code-block:: bash

   # Install pypa/build if you don't have it: pip install build
   python -m build --wheel

The ``.whl`` file in ``dist/`` can be uploaded to PyPI or installed locally
with ``pip install dist/fib_wrapper-0.1.0-*.whl``.

For development iteration, an editable install avoids repeated wheel builds:

.. code-block:: bash

   pip install --no-build-isolation --editable .

This reuses the current environment and requires ``meson-python``, ``meson``,
``ninja``, and ``numpy`` to already be installed.

Verify
======

.. code-block:: python

   >>> from fib_wrapper import fib
   >>> fib(10)
   array([ 0,  1,  1,  2,  3,  5,  8, 13, 21, 34], dtype=int32)

Selecting a Fortran compiler
=============================

By default ``meson`` picks the first Fortran compiler on ``PATH``.
Set ``FC`` to override:

.. code-block:: bash

   FC=ifx python -m build --wheel

For finer control, write a `Meson native file
<https://mesonbuild.com/Native-environments.html>`_:

.. code-block:: ini

   ; native.ini
   [binaries]
   fortran = 'ifx'
   c = 'icx'

.. code-block:: bash

   python -m build --wheel -Csetup-args="--native-file=native.ini"

Adding dependencies (BLAS, LAPACK, etc.)
========================================

Use ``dependency()`` in ``meson.build`` to link against system libraries:

.. code-block:: meson

   lapack_dep = dependency('lapack')

   py.extension_module('mymod',
     [sources, generated, incdir_f2py / 'fortranobject.c'],
     dependencies : [np_dep, f2py_dep, lapack_dep],
     install : true,
   )

``meson`` resolves dependencies through ``pkg-config``, CMake, or its own
detection logic.  See the `Meson dependency documentation
<https://mesonbuild.com/Dependencies.html>`_ for details.

Comparison with ``scikit-build-core``
======================================

:ref:`f2py-skbuild` uses CMake under the hood.  ``meson-python`` provides
native Fortran compiler support without a CMake layer and is the same build
system NumPy and SciPy use.

Further reading
===============

* `meson-python documentation <https://meson-python.readthedocs.io/>`_
* `Meson build system <https://mesonbuild.com/>`_
* `SciPy meson.build <https://github.com/scipy/scipy/blob/main/meson.build>`_ (real-world F2PY usage)
* :ref:`f2py-meson` (building without ``meson-python``)
* :ref:`f2py-skbuild` (alternative with ``scikit-build-core`` / CMake)
* :ref:`f2py-meson-distutils` (migration from ``distutils``)
