.. _f2py-meson:

===================
Using via ``meson``
===================

.. note::

   Much of this document is now obsoleted, one can run ``f2py`` with
   ``--build-dir`` to get a skeleton ``meson`` project with basic dependencies
   setup.

.. versionchanged:: 1.26.x

   The default build system for ``f2py`` is now ``meson``, see
   :ref:`distutils-status-migration` for some more details.

Fibonacci walkthrough (F77)
===========================

We will need the generated ``C`` wrapper before we can use a general purpose
build system like ``meson``. We will acquire this by:

.. code-block:: bash

    python -m numpy.f2py fib1.f -m fib2

Now, consider the following ``meson.build`` file for the ``fib`` and ``scalar``
examples from :ref:`f2py-getting-started` section:

.. literalinclude:: ../code/meson.build

.. note::

   ``meson`` refuses absolute include paths that point into the project
   directory, which is where the NumPy and ``f2py`` headers are when the
   virtual environment is created inside the project, e.g. as ``.venv``. The
   example therefore passes them to ``include_directories`` relative to the
   project with ``fs.relative_to``, which needs ``meson>=1.3.0``.

At this point the build will complete, but the import will fail:

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

Recall that the original example, as reproduced below, was in SCREAMCASE:

.. literalinclude:: ./../code/fib1.f
   :language: fortran

With the standard approach, the subroutine exposed to ``python`` is ``fib`` and
not ``FIB``. This means we have a few options. One approach (where possible) is
to lowercase the original Fortran file with say:

.. code-block:: bash

   tr "[:upper:]" "[:lower:]" < fib1.f > fib1_lower.f
   mv fib1_lower.f fib1.f
   python -m numpy.f2py fib1.f -m fib2
   meson setup --wipe builddir
   meson compile -C builddir
   cd builddir
   python -c 'import fib2'

However this requires the ability to modify the source which is not always
possible. The easiest way to solve this is to let ``f2py`` deal with it:

.. code-block:: bash

   python -m numpy.f2py fib1.f -m fib2 --lower
   meson setup --wipe builddir
   meson compile -C builddir
   cd builddir
   python -c 'import fib2'


Automating wrapper generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A major pain point in the workflow defined above, is the manual tracking of
inputs. Although it would require more effort to figure out the actual outputs
for reasons discussed in :ref:`f2py-bldsys`.

.. note::

   From NumPy ``1.22.4`` onwards, ``f2py`` will deterministically generate
   wrapper files based on the input file Fortran standard (F77 or greater).
   ``--skip-empty-wrappers`` can be passed to ``f2py`` to restore the previous
   behaviour of only generating wrappers when needed by the input .

However, we can augment our workflow in a straightforward to take into account
files for which the outputs are known when the build system is set up.

.. literalinclude:: ../code/meson_upd.build

This can be compiled and run as before.

.. code-block:: bash

    rm -rf builddir
    meson setup builddir
    meson compile -C builddir
    cd builddir
    python -c "import numpy as np; import fibby; a = np.zeros(9); fibby.fib(a); print (a)"
    # [ 0.  1.  1.  2.  3.  5.  8. 13. 21.]

Wrappers from a signature file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use a signature file edited as in :ref:`f2py-getting-started` (``fib2.pyf``
from "The smart way"), generate the wrappers from the ``.pyf`` file alone. The
signature file names the module, so ``-m`` is not needed, and the Fortran
sources go to ``extension_module`` to be compiled:

.. literalinclude:: ../code/meson_pyf.build

.. code-block:: bash

    rm -rf builddir
    meson setup builddir
    meson compile -C builddir
    cd builddir
    python -c "import fib2; print(fib2.fib(9))"
    # [ 0.  1.  1.  2.  3.  5.  8. 13. 21.]

.. versionchanged:: 2.6.0

   ``f2py`` ignores ``-m`` and any Fortran sources passed together with a
   signature file when it only generates wrappers. Earlier versions built a
   second module from them that silently replaced the wrappers described by
   the signature file.

Salient points
===============

It is worth keeping in mind the following:

* It is not possible to use SCREAMCASE in this context, so either the contents
  of the ``.f`` file or the generated wrapper ``.c`` needs to be lowered to
  regular letters; which can be facilitated by the ``--lower`` option of
  ``F2PY``
