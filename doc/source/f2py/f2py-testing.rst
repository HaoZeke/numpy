.. _f2py-testing:

===============
F2PY test suite
===============

The F2PY test suite lives in ``numpy/f2py/tests``. It verifies that Fortran
language features translate correctly to Python. For example, Fortran allows
user-specified array index ranges; the generated CPython extension normalizes
these so arrays always start from index 0.

The directory of the test suite looks like the following::

	./tests/
	├── __init__.py
	├── src
	│   ├── abstract_interface
	│   ├── array_from_pyobj
	│   ├── // ... several test folders
	│   └── string
	├── test_abstract_interface.py
	├── test_array_from_pyobj.py
	├── // ... several test files
	├── test_symbolic.py
	└── util.py

Files starting with ``test_`` contain tests covering F2PY from Fortran parsing
through to module documentation. The ``src`` directory holds the Fortran source
files used by those tests. ``util.py`` provides utility functions for building
and importing Fortran modules into a temporary location at test time.

Adding a test
==============

The F2PY test suite predates ``pytest`` and does not use fixtures. Instead,
test files contain test classes that inherit from the ``F2PyTest`` class in
``util.py``.

.. literalinclude:: ../../../numpy/f2py/tests/util.py
   :language: python
   :lines:  327-336
   :linenos:

This class provides helper functions for parsing and compiling test source
files. Child classes override the ``sources`` data member to supply their own
source files. The superclass compiles these files on object creation and
attaches the resulting functions to ``self.module``. Child classes then call
Fortran routines via ``self.module.<fortran_function_name>``.

.. versionadded:: v2.0.0b1

All ``f2py`` tests must pass even when no Fortran compiler is installed. The
``CompilerChecker`` class provides Meson-based utilities --
``has_{c,f77,f90,fortran}_compiler()`` -- to skip compilation-dependent tests
when appropriate.

For CLI tests in ``test_f2py2e``, flags that invoke ``meson`` or otherwise
require a compiler must call ``compiler_check_f2pycli()`` instead of
``f2pycli()``.

Example
~~~~~~~

Consider the following subroutines, contained in a file named :file:`add-test.f`

.. literalinclude:: ./code/add-test.f
   :language: fortran

The first routine ``addb`` takes an array and increments each element by 1.
The second subroutine ``addc`` assigns to array ``k`` the elements of input
array ``w`` incremented by 1.

A test can be implemented as follows::

	class TestAdd(util.F2PyTest):
	    sources = [util.getpath("add-test.f")]

	    def test_module(self):
	        k = np.array([1, 2, 3], dtype=np.float64)
	        w = np.array([1, 2, 3], dtype=np.float64)
	        self.module.addb(k)
	        assert np.allclose(k, w + 1)
	        self.module.addc([w, k])
	        assert np.allclose(k, w + 1)

The ``sources`` data member points to the source file. On class instantiation,
the source files are compiled and subroutines attached to ``self.module``. The
``test_module`` method calls the subroutines and verifies their results.
