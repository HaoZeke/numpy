.. _f2py-cmake:

=========================================
Building an extension with ``CMake``
=========================================

``CMake`` sits between ``make`` and ``meson`` in complexity. The syntax
resembles ``make`` with environment variables rather than Python, but in
exchange ``CMake`` supports most architectures and compilers. For a syntax
primer, see the `extensive CMake collection`_.

.. note::

   ``CMake`` works for mixed-language systems, but its ``f2py`` integration
   requires manual wiring. For a more streamlined approach, see
   :ref:`f2py-skbuild`.

Fibonacci walkthrough (F77)
===========================

We return to the ``fib`` example from :ref:`f2py-getting-started`.

.. literalinclude:: ./../code/fib1.f
    :language: fortran

We do not need to run ``python -m numpy.f2py fib1.f`` manually to produce
``fib1module.c``. The ``CMakeLists.txt`` below handles wrapper generation
as part of the build:

.. literalinclude:: ./../code/CMakeLists.txt
    :language: cmake

``add_custom_command`` generates the wrapper C files, and
``add_custom_target`` registers them as a dependency of the shared library
target so that the command does not re-run on every build. The same approach
used to locate ``fortranobject.c`` also works for finding ``numpy`` headers
on older ``CMake`` versions.

.. code:: bash

    ls .
    # CMakeLists.txt fib1.f
    cmake -S . -B build
    cmake --build build
    cd build
    python -c "import numpy as np; import fibby; a = np.zeros(9); fibby.fib(a); print (a)"
    # [ 0.  1.  1.  2.  3.  5.  8. 13. 21.]

This approach is useful when an existing CMake toolchain is already in place
and adding Python-side build dependencies like ``scikit-build-core`` is
undesirable.

.. _extensive CMake collection: https://cliutils.gitlab.io/modern-cmake/
