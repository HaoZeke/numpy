.. _f2py-skbuild:

========================================================
Distributing F2PY extensions with ``scikit-build-core``
========================================================

``scikit-build-core`` is a build backend that drives ``CMake`` from a
``pyproject.toml`` file, producing wheels and sdists without ``setup.py``.
It also ships ``CMake`` modules that simplify building Python extensions.

.. note::

   The legacy ``scikit-build`` package (which wrapped ``setuptools``) is
   superseded by ``scikit-build-core``. The examples in the ``setuptools``
   replacement section below reflect the old workflow for reference only.

If you do not need wheel packaging and just want to call ``CMake`` directly,
use the plain ``CMake`` setup described in :ref:`f2py-cmake`.

Fibonacci walkthrough (F77)
===========================

We use the ``fib`` example from :ref:`f2py-getting-started`.

.. literalinclude:: ./../code/fib1.f
    :language: fortran

``CMake`` modules only
~~~~~~~~~~~~~~~~~~~~~~

The following ``CMakeLists.txt`` uses ``scikit-build-core``'s CMake modules
directly:

.. literalinclude:: ./../code/CMakeLists_skbuild.txt
   :language: cmake

The logic mirrors :ref:`f2py-cmake`, but here the module suffix comes from
``sysconfig.get_config_var("SO")``. Build and load the extension as usual:

.. code:: bash

    ls .
    # CMakeLists.txt fib1.f
    cmake -S . -B build
    cmake --build build
    cd build
    python -c "import numpy as np; import fibby; a = np.zeros(9); fibby.fib(a); print (a)"
    # [ 0.  1.  1.  2.  3.  5.  8. 13. 21.]


``setuptools`` replacement (legacy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   The ``setup.py``-driven workflow below uses the legacy ``scikit-build``
   package and should not be depended on for new projects. Prefer
   ``scikit-build-core`` with a ``pyproject.toml``-only configuration.

The legacy ``scikit-build`` package could drive ``CMake`` through
``setuptools``, producing wheels for PyPI. The project needed a ``setup.py``:

.. literalinclude:: ./../code/setup_skbuild.py
   :language: python

Along with a matching ``pyproject.toml``:

.. literalinclude:: ./../code/pyproj_skbuild.toml
   :language: toml

Together these build the extension using ``CMake`` alongside standard
``setuptools`` outputs. This pattern was mostly used to integrate with
extension modules not built with ``CMake``.

.. code:: bash

    ls .
    # CMakeLists.txt fib1.f pyproject.toml setup.py
    python setup.py build_ext --inplace
    python -c "import numpy as np; import fibby.fibby; a = np.zeros(9); fibby.fibby.fib(a); print (a)"
    # [ 0.  1.  1.  2.  3.  5.  8. 13. 21.]

The module path differs here because ``--inplace`` places the extension in a
subfolder.

.. _bypass the cmake setup mechanism: https://scikit-build.readthedocs.io/en/latest/cmake-modules/F2PY.html
