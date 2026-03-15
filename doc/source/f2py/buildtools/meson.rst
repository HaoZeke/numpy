.. _f2py-meson:

=========================================
Building an extension with ``meson``
=========================================

.. note::

   Much of this page is obsoleted by ``f2py --build-dir``, which generates a
   skeleton ``meson`` project with dependencies already configured.

.. versionchanged:: 1.26.x

   The default build system for ``f2py`` is ``meson``. See
   :ref:`distutils-status-migration` for details.

Fibonacci walkthrough (F77)
===========================

Generate the ``C`` wrapper first:

.. code-block:: bash

    python -m numpy.f2py fib1.f -m fib2

The following ``meson.build`` file covers the ``fib`` and ``scalar`` examples
from :ref:`f2py-getting-started`:

.. literalinclude:: ../code/meson.build

The build will complete, but the import fails:

.. code-block:: bash

   meson setup builddir
   meson compile -C builddir
   cd builddir
   python -c 'import fib2'
   Traceback (most recent call last):
   File "<string>", line 1, in <module>
   ImportError: fib2.cpython-39-x86_64-linux-gnu.so: undefined symbol: FIB_
   # Check this isn't a false positive
   nm -A fib2.cpython-39-x86_64-linux-gnu.so | grep FIB_
   fib2.cpython-39-x86_64-linux-gnu.so: U FIB_

The original Fortran source uses SCREAMCASE:

.. literalinclude:: ./../code/fib1.f
   :language: fortran

The subroutine exposed to ``python`` is ``fib``, not ``FIB``. One fix is to
lowercase the source file:

.. code-block:: bash

   tr "[:upper:]" "[:lower:]" < fib1.f > fib1.f
   python -m numpy.f2py fib1.f -m fib2
   meson --wipe builddir
   meson compile -C builddir
   cd builddir
   python -c 'import fib2'

When modifying the source is not an option, pass ``--lower`` to ``f2py``
instead:

.. code-block:: bash

   python -m numpy.f2py fib1.f -m fib2 --lower
   meson --wipe builddir
   meson compile -C builddir
   cd builddir
   python -c 'import fib2'


Automating wrapper generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The workflow above requires manual tracking of inputs. Determining the actual
outputs requires more effort, for reasons discussed in :ref:`f2py-bldsys`.

.. note::

   From NumPy ``1.22.4`` onwards, ``f2py`` deterministically generates wrapper
   files based on the input file Fortran standard (F77 or greater).
   ``--skip-empty-wrappers`` restores the previous behaviour of only generating
   wrappers when the input requires them.

We can augment the build to account for files whose outputs are known at
configuration time:

.. literalinclude:: ../code/meson_upd.build

Compile and run as before:

.. code-block:: bash

    rm -rf builddir
    meson setup builddir
    meson compile -C builddir
    cd builddir
    python -c "import numpy as np; import fibby; a = np.zeros(9); fibby.fib(a); print (a)"
    # [ 0.  1.  1.  2.  3.  5.  8. 13. 21.]

Salient points
===============

* SCREAMCASE symbols are not resolved automatically. Either lowercase the
  ``.f`` source or lowercase the generated ``.c`` wrapper. The ``--lower``
  option of ``F2PY`` handles this.
