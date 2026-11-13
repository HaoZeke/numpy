.. _f2py-meson-python:

=====================================================
Distributing F2PY extensions with ``meson-python``
=====================================================

The :ref:`f2py-meson` page covers building F2PY extensions using raw ``meson``
commands. This page shows how to package those extensions into installable
Python distributions (sdists and wheels) using `meson-python
<https://meson-python.readthedocs.io/>`_ as the PEP 517 build backend.

This is the recommended approach for distributing F2PY-wrapped Fortran code as a
Python package on PyPI or for local ``pip install`` workflows.

.. note::

   ``meson-python`` replaced ``setuptools`` / ``numpy.distutils`` as the
   standard way to build and distribute compiled extensions in the NumPy and
   SciPy ecosystem. See :ref:`distutils-status-migration` for background.

Prerequisites
=============

You need:

* A C compiler
* A Fortran compiler (``gfortran``, ``ifort``, ``ifx``, ``flang-new``, etc.),
  if you use any Fortran code in your package
* Python >= 3.10
* ``meson``, ``meson-python``, and ``numpy`` (installed automatically during the
  build when listed in ``build-system.requires``)

Minimal example
===============

The project below wraps a Fortran ``fib`` subroutine into an importable Python
package called ``fib_wrapper``.

Project layout::

    fib_wrapper/           # project root
    ├── fib.f90            # Fortran source
    ├── fib_wrapper/       # Python package directory
    │   └── __init__.py
    ├── meson.build
    └── pyproject.toml

Fortran source
--------------

Save the following as ``fib.f90``:

.. literalinclude:: ../code/fib_mesonpy.f90
   :language: fortran

``pyproject.toml``
------------------

.. literalinclude:: ../code/pyproj_mesonpy.toml
   :language: toml

Two entries matter here:

* ``build-backend = "mesonpy"`` tells build frontends to use ``meson-python``.
* ``requires`` lists build-time dependencies. ``numpy >= 2.0`` is required so
  that ``f2py``, the NumPy headers, and ``dependency('numpy')`` support in Meson
  are available during compilation.

``meson.build``
---------------

.. literalinclude:: ../code/meson_mesonpy.build

.. note::

   The file is stored as ``meson_mesonpy.build`` in the documentation source
   tree to avoid collisions with other examples. In your project, name it
   ``meson.build``.

The ``meson.build`` file does four things:

1. Uses ``dependency('numpy')`` to locate NumPy headers, and a
   ``declare_dependency`` to add the F2PY include directory (for
   ``fortranobject.h``).
2. Runs ``f2py`` via ``custom_target`` to generate the C wrapper sources.
3. Compiles the generated C code together with the Fortran source into a Python
   extension module using ``py.extension_module``.
4. Installs ``__init__.py`` into the package directory so the result is a proper
   Python package.

The ``subdir: 'fib_wrapper'`` argument on the extension module is required so
that the compiled ``fib`` shared library is installed inside the ``fib_wrapper/``
package directory, next to ``__init__.py``. Without it the extension would
be installed at the top level and ``import fib_wrapper`` would not find the
``fib`` extension. The resulting installed layout is::

    site-packages/
    └── fib_wrapper/
        ├── __init__.py        # from .fib import fib
        └── fib.cpython-*.so   # compiled extension module

``__init__.py``
---------------

A minimal ``__init__.py`` re-exports the wrapped function:

.. code-block:: python

   from .fib import fib

Building and installing
=======================

Editable install (development)
------------------------------

.. code-block:: bash

   pip install --no-build-isolation --editable .

``--no-build-isolation`` reuses the current environment, which is useful when
iterating. This requires ``meson-python``, ``meson``, ``ninja``, and ``numpy``
to already be installed.

Building a wheel
----------------

.. code-block:: bash

   # If you don't yet have `pypa/build` installed: `pip install build`
   python -m build --wheel

The resulting ``.whl`` file in ``dist/`` can be uploaded to PyPI, or installed
elsewhere with ``pip install dist/fib_wrapper-0.1.0-*.whl``.

Verifying the install
---------------------

.. code-block:: python

   >>> from fib_wrapper import fib
   >>> fib(10)
   array([ 0,  1,  1,  2,  3,  5,  8, 13, 21, 34], dtype=int32)

Customizing the Fortran compiler
================================

``meson-python`` delegates compiler selection to ``meson``. By default,
``meson`` will choose the first Fortran compiler it finds on the PATH.
If you want more control over Fortran compiler selection, set the ``FC``
environment variable before building:

.. code-block:: bash

   FC=ifx python -m build --wheel

For more control, use a `Meson native file
<https://mesonbuild.com/Native-environments.html>`_:

.. code-block:: ini

   ; native.ini
   [binaries]
   fortran = 'ifx'
   c = 'icx'

.. code-block:: bash

   python -m build --wheel -Csetup-args="--native-file=native.ini"

Large automatic arrays
======================

An automatic array is a local explicit-shape array whose bounds are not
compile-time constants, for example ``real :: a(n)`` or
``real :: tmp(nvec, ndim)``. Where that storage comes from depends on the
compiler. A multi-megabyte temporary inside a routine that is also an OpenMP
region, such as ``drc3jj_vec`` in py3nj (about 13 MB for the ``wigner3j``
test), overflows the usual 8 MB process stack on compilers that keep the
array there.

``gfortran`` moves an automatic array to the heap when it is larger than
``-fmax-stack-var-size``. The default limit is 32 KiB, so the 13 MB array
runs with no extra flag. A 16 MB automatic array compiled by gfortran 13
also runs under an 8 MB stack.

``ifx`` keeps these arrays on the stack (``-no-heap-arrays`` is the default).
``-heap-arrays`` allocates them on the heap. On the same ``wigner3j`` call
(``n = 10000``), the default build segfaults with an 8 MB stack, and
``FFLAGS=-heap-arrays`` passes. With the stack limit raised so the default
build finishes, seven timed repeats after one warmup average 0.0289 s on
the stack and 0.0364 s with ``-heap-arrays``, about 26% slower.

``flang`` 20 puts the automatic array on the stack. ``-fno-stack-arrays`` is
already the default and only redirects compiler-generated temporaries, not
this array; ``-fstack-arrays`` keeps it on the stack as well. The hidden
option ``-fdynamic-heap-array`` tells the ``memory-allocation-opt`` pass to
allocate every runtime-sized array on the heap. flang 20.1.2 does not accept
it as a bare driver flag. Pass it through LLVM's option parser:

.. code-block:: bash

   FFLAGS="-mllvm -fdynamic-heap-array"

``-mmlir -fdynamic-heap-array`` is accepted by the same driver and has the
same effect. The option ``-mllvm -fstack-array-size=<n>`` reads as a
per-array element limit. Yet flang 20.1.2 still puts a 2,000,000-element
array on the stack. This release has no usable limit. The ``wigner3j``
test passes on an 8 MB stack with ``-mllvm -fdynamic-heap-array``.

``OMP_STACKSIZE`` sizes the stacks of OpenMP worker threads. It does not
enlarge the stack of the thread that entered the routine. For ``drc3jj_vec``
the large array is created before the parallel region, so
``OMP_STACKSIZE=64M`` and ``128M`` still segfault under an 8 MB ``ulimit -s``.
Without a rebuild, raise the process stack with ``ulimit -s`` (in KiB).
Raise ``OMP_STACKSIZE`` too when the large array lives inside the parallel
region.

Adding dependencies (BLAS, LAPACK, etc.)
========================================

Use ``dependency()`` in ``meson.build`` to link against system libraries:

.. code-block:: none

   lapack_dep = dependency('lapack')

   py.extension_module('mymod',
     [sources, generated, incdir_f2py / 'fortranobject.c'],
     dependencies : [np_dep, f2py_dep, lapack_dep],
     install : true,
   )

``meson`` resolves dependencies through ``pkg-config``, CMake, or its own
detection logic. See the `Meson dependency documentation
<https://mesonbuild.com/Dependencies.html>`_ for details.

Differences from the ``scikit-build-core`` workflow
====================================================

The ``scikit-build-core`` approach documented in :ref:`f2py-skbuild` uses CMake
under the hood. ``meson-python`` provides:

* Native Fortran compiler support in ``meson`` (no CMake layer).
* Direct integration with ``pip`` / ``build`` via PEP 517.
* The same build system used by NumPy and SciPy themselves.

Further reading
===============

* `meson-python documentation <https://meson-python.readthedocs.io/>`_
* `Meson build system <https://mesonbuild.com/>`_
* `SciPy's meson build configuration <https://github.com/scipy/scipy/blob/main/meson.build>`_ (real-world F2PY usage)
* :ref:`f2py-meson` (raw meson build without ``meson-python``)
* :ref:`f2py-skbuild` (alternative using ``scikit-build-core`` / CMake)
* :ref:`f2py-meson-distutils` (migration from ``distutils``)
