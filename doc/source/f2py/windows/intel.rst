.. _f2py-win-intel:

==============================
F2PY and Windows Intel Fortran
==============================

The supported compiler is the LLVM-based ``ifx`` from the free Intel
oneAPI toolkits. The classic ``ifort`` compiler has been discontinued
and is absent from recent oneAPI releases; the examples below use
``ifx``.

.. note::

	This document does not endorse the usage of Intel in downstream
	projects due to the issues pertaining to `disassembly of components and
	liability`_.

	Neither the Python Intel installation nor the `Classic Intel C/C++
	Compiler` are required.

- The `Intel Fortran Compilers`_ install through the oneAPI HPC toolkit,
  which also takes around a gigabyte and a half or so.

We will consider the classic example of the generation of Fibonnaci numbers,
``fib1.f``, given by:

.. literalinclude:: ../code/fib1.f
   :language: fortran

For ``cmd.exe`` fans, using the Intel oneAPI command prompt is the easiest
approach: ``setvars.bat`` loads the environment for both ``ifx`` and MSVC.
Helper batch scripts are also provided. After the environment is loaded,
``f2py -c`` uses the default meson backend and picks up ``ifx`` from
``PATH``:

.. code-block:: bat

   # cmd.exe
   "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
   python -m numpy.f2py -c fib1.f -m fib1
   python -c "import fib1; import numpy as np; a=np.zeros(8); fib1.fib(a); print(a)"

PowerShell usage is a little less pleasant. Load the oneAPI environment
(so ``ifx`` and the MSVC linker are on ``PATH``), then invoke ``f2py`` as
usual:

.. code-block:: powershell

   # PowerShell: load oneAPI env then open a shell with it
   cmd.exe /k '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'
   python -m numpy.f2py -c fib1.f -m fib1
   python -c "import fib1; import numpy as np; a=np.zeros(8); fib1.fib(a); print(a)"

Note that the path to your local oneAPI install may vary; adjust
``setvars.bat`` accordingly. The ``ifx`` binary is typically under
``C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin``.


Importing ifx-built modules
===========================

The DLL rule from :ref:`f2py-win-msys2` applies here too: a module
built with ``ifx`` depends on the Intel runtime DLLs (``libifcoremd``,
``svml_dispmd``, ...), and Python 3.8+ does not consult ``PATH`` when
resolving them. Register the runtime directory before the import:

.. code-block:: python

   import os
   os.add_dll_directory(r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin")
   import mymodule

Additionally, ``ifx`` on Windows uses uppercase symbol names with no
trailing underscore, so the C wrapper must be compiled with the standard
mangling macros ``UPPERCASE_FORTRAN`` and ``NO_APPEND_FORTRAN`` (see
:doc:`../usage`).

.. note::

   The default meson backend does not currently forward ``-D`` defines from
   the ``f2py -c`` command line into the generated build. Apply those macros
   through your build system's C compiler flags (for example meson's
   ``c_args``) rather than expecting
   ``python -m numpy.f2py -c ... -DUPPERCASE_FORTRAN -DNO_APPEND_FORTRAN``
   to take effect on its own.


.. _disassembly of components and liability: https://www.intel.com/content/www/us/en/developer/articles/license/end-user-license-agreement.html
.. _Intel Fortran Compilers: https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#inpage-nav-6-1
.. _Classic Intel C/C++ Compiler: https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#inpage-nav-6-undefined
