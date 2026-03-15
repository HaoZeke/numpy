.. _f2py:

=====================================
F2PY user guide and reference manual
=====================================

The purpose of the ``F2PY`` -- *Fortran to Python interface generator* -- utility
is to provide a connection between Python and Fortran. F2PY is distributed as
part of NumPy_ (``numpy.f2py``) and, once installed, is also available as a
standalone command line tool. It was originally created by Pearu Peterson;
older changelogs are in the `historical reference`_.

F2PY facilitates creating and building native `Python C/API extension modules`_
that make it possible to:

* call Fortran 77/90/95 external subroutines and Fortran 90/95
  module subroutines as well as C functions;
* access Fortran 77 ``COMMON`` blocks and Fortran 90/95 module data,
  including allocatable arrays

from Python.


.. note::

   Fortran 77 support is feature-complete. An increasing amount of
   modern Fortran is supported as well; most ``iso_c_binding`` interfaces
   compile to native extension modules automatically with ``f2py``.
   Bug reports are welcome.

F2PY can be used either as a command line tool ``f2py`` or as a Python
module ``numpy.f2py``. While the command line tool is provided as part
of the NumPy installation, some platforms (notably Windows) make it difficult
to reliably place executables on the ``PATH``. If the ``f2py`` command is not
available on your system, run it as a module::

   python -m numpy.f2py

The ``python -m`` invocation is also useful when multiple Python installations
coexist on the same system (outside of virtual environments) and you need to
target a particular version of Python/F2PY.

If you run ``f2py`` with no arguments, and the line ``numpy Version`` at the
end matches the NumPy version printed from ``python -m numpy.f2py``, then you
can use the shorter version. If not, or if you cannot run ``f2py``, you should
replace all calls to ``f2py`` mentioned in this guide with the longer version.

For Meson build examples, see :doc:`usage`.

.. toctree::
   :maxdepth: 3

   f2py-user
   f2py-reference
   windows/index

.. _Python: https://www.python.org/
.. _NumPy: https://www.numpy.org/
.. _`historical reference`: https://web.archive.org/web/20140822061353/http://cens.ioc.ee/projects/f2py2e
.. _Python C/API extension modules: https://docs.python.org/3/extending/extending.html#extending-python-with-c-or-c
