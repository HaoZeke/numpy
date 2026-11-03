.. _f2py-testing:

===============
F2PY test suite
===============

F2PY's test suite is present in the directory ``numpy/f2py/tests``. Its aim
is to ensure that Fortran language features are correctly translated to Python.
For example, the user can specify starting and ending indices of arrays in
Fortran. This behaviour is translated to the generated CPython library where
the arrays strictly start from 0 index.

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

Files starting with ``test_`` contain tests for various aspects of f2py from parsing
Fortran files to checking modules' documentation. ``src`` directory contains the
Fortran source files upon which we do the testing. ``util.py`` contains utility 
functions for building and importing Fortran modules during test time using a 
temporary location.

Adding a test
==============

Most of F2PY's test suite predates ``pytest`` fixtures, so the test files
contain test classes that inherit from the ``F2PyTest`` class in ``util.py``.

.. literalinclude:: ../../../numpy/f2py/tests/util.py
   :language: python
   :pyobject: F2PyTest
   :lines: 1-8
   :linenos:

This class many helper functions for parsing and compiling test source files. Its child 
classes can override its ``sources`` data member to provide their own source files.
This superclass will then compile the added source files upon object creation and their
functions will be appended to ``self.module`` data member. Thus, the child classes will
be able to access the fortran functions specified in source file by calling
``self.module.[fortran_function_name]``.

.. versionadded:: v2.0.0b1

Each of the ``f2py`` tests should run without failure if no Fortran compilers
are present on the host machine. To facilitate this, the ``CompilerChecker`` is
used, essentially providing a ``meson`` dependent set of utilities namely
``has_{c,f77,f90,fortran}_compiler()``.

.. versionchanged:: 2.6.0

The ``CompilerChecker`` runs ``meson``, so it is no longer run when the test
modules are imported. It runs once per test session, when a test first needs a
compiler, through the session-scoped ``check_compilers`` fixture defined in
``conftest.py``; the ``has_{c,f77,f90,fortran}_compiler()`` helpers share its
results. ``F2PyTest`` only requests the fixture for classes which set
``sources`` or ``code``, so tests which do not build an extension module run
without ``meson`` or a Fortran compiler. New tests which build a module outside
of ``F2PyTest`` should request ``check_compilers`` instead of probing for
compilers at import time.

For the CLI tests in ``test_f2py2e``, flags which are expected to call ``meson``
or otherwise depend on a compiler need to call ``compiler_check_f2pycli()``
instead of ``f2pycli()``.

Example
~~~~~~~

Consider the following subroutines, contained in a file named :file:`add-test.f`

.. literalinclude:: ./code/add-test.f
   :language: fortran

The first routine `addb` simply takes an array and increases its elements by 1.
The second subroutine `addc` assigns a new array `k` with elements greater that 
the elements of the input array `w` by 1.

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

We override the ``sources`` data member to provide the source file. The source files
are compiled and subroutines are attached to module data member when the class object
is created. The ``test_module`` function calls the subroutines and tests their results.
