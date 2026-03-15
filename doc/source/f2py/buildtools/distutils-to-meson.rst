.. _f2py-meson-distutils:

================================================
Migrating from ``distutils`` to ``meson``
================================================

As per the timeline laid out in :ref:`distutils-status-migration`,
``distutils`` has been removed. This page collects common ``meson``-based
workflows that replace the old ``distutils`` patterns.

.. note::

    Contributions welcome via
    `pull requests <https://numpy.org/doc/stable/dev/howto-docs.html>`_.

Baseline
--------

The examples below use a Fibonacci series generator with ``iso_c_binding``:

.. code-block:: fortran

    ! fib.f90
    subroutine fib(a, n)
      use iso_c_binding
       integer(c_int), intent(in) :: n
       integer(c_int), intent(out) :: a(n)
       do i = 1, n
          if (i .eq. 1) then
             a(i) = 0.0d0
          elseif (i .eq. 2) then
             a(i) = 1.0d0
          else
             a(i) = a(i - 1) + a(i - 2)
          end if
       end do
    end

Compilation options
-------------------

Basic usage
~~~~~~~~~~~

.. code-block:: bash

    python -m numpy.f2py -c fib.f90 -m fib
    python -c "import fib; print(fib.fib(30))"
    # [     0      1      1      2      3      5      8     13     21     34
    #       55     89    144    233    377    610    987   1597   2584   4181
    #     6765  10946  17711  28657  46368  75025 121393 196418 317811 514229]

Specify the backend
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  python -m numpy.f2py -c fib.f90 -m fib

``meson`` is the only backend. The ``distutils`` backend was removed in
NumPy 2.5.0.

Pass a compiler name
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  FC=gfortran python -m numpy.f2py -c fib.f90 -m fib

`Meson native files <https://mesonbuild.com/Native-environments.html>`_ can
also be used. ``CC`` sets the C compiler in the same way. The table below
lists the recognized environment variables.

.. table::

    +------------------------------------+-------------------------------+
    | **Name**                           | **What**                      |
    +------------------------------------+-------------------------------+
    | FC                                 | Fortran compiler              |
    +------------------------------------+-------------------------------+
    | CC                                 | C compiler                    |
    +------------------------------------+-------------------------------+
    | CFLAGS                             | C compiler options            |
    +------------------------------------+-------------------------------+
    | FFLAGS                             | Fortran compiler options      |
    +------------------------------------+-------------------------------+
    | LDFLAGS                            | Linker options                |
    +------------------------------------+-------------------------------+
    | LD_LIBRARY_PATH                    | Library file locations (Unix) |
    +------------------------------------+-------------------------------+
    | LIBS                               | Libraries to link against     |
    +------------------------------------+-------------------------------+
    | PATH                               | Search path for executables   |
    +------------------------------------+-------------------------------+
    | CXX                                | C++ compiler                  |
    +------------------------------------+-------------------------------+
    | CXXFLAGS                           | C++ compiler options          |
    +------------------------------------+-------------------------------+


.. note::

    On Windows, environment variables may not propagate reliably.
    Use `native files <https://mesonbuild.com/Native-environments.html>`_
    or direct build customization (see `Customizing builds`_ below) instead.

Dependencies
~~~~~~~~~~~~

.. code-block:: bash

  python -m numpy.f2py -c fib.f90 -m fib --dep lapack

This maps to ``dependency("lapack")`` in the generated ``meson.build`` and
works for any dependency that meson can resolve. See the
`meson dependency documentation <https://mesonbuild.com/Dependencies.html>`_
for CMake-based and other resolution methods.

Libraries
~~~~~~~~~

To link against additional libraries:

.. code-block:: bash

  python -m numpy.f2py -c fib.f90 -m fib -lmylib -L/path/to/mylib

Customizing builds
------------------

.. code-block:: bash

  python -m numpy.f2py -c fib.f90 -m fib --build-dir blah

The ``--build-dir`` flag writes the generated ``meson.build`` and wrapper
sources to the specified directory. From there, standard meson customization
applies; see the `Meson Build How-To Guide <https://mesonbuild.com/howtox.html>`_.
The generated files can also be committed and used as a meson subproject in a
larger codebase.
