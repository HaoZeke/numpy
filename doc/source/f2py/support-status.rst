.. _f2py-support-status:

===========================
F2PY support status matrix
===========================

This page records, in one place, which Fortran language features F2PY
supports, how Fortran types map onto NumPy types across platforms, and
which larger features are planned but not yet implemented. Each planned
item links its tracking issue so progress is auditable from the
documentation itself.

Language standard coverage
==========================

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Standard
     - Status
     - Notes
   * - Fortran 77
     - Essentially feature complete
     - Subroutines, functions, ``COMMON`` blocks, ``EXTERNAL``
       callbacks, character handling.
   * - Fortran 90/95
     - Substantially complete
     - Free form, modules (data and procedures), assumed-shape and
       allocatable module arrays, ``optional`` arguments. Built-in
       kind numbers (for example ``integer(8)``, ``real(4)``) map by
       default; non-standard or PARAMETER-named kinds need a
       ``.f2py_f2cmap`` file. See the compatibility statement below
       (`gh-25424 <https://github.com/numpy/numpy/issues/25424>`__).
   * - Fortran 2003
     - Partial
     - ``bind(c)`` interfaces and ``iso_c_binding`` named constants are
       recognized, with forced casting for some kinds rather than full
       native width mapping (`gh-25229 <https://github.com/numpy/numpy/issues/25229>`__).
       Type-bound procedures and derived types are not wrapped
       (`gh-21160 <https://github.com/numpy/numpy/issues/21160>`__).
   * - Fortran 2008+
     - Partial
     - Coarrays, submodules, and assumed-rank (F2018
       ``dimension(..)``) are unsupported.

Fortran 95 compatibility statement
==================================

F2PY targets full Fortran 95 interoperability for procedure interfaces:
any F95 procedure whose dummy arguments use intrinsic types, assumed or
explicit shapes, and ``optional``/``intent`` attributes should wrap
without a hand-written signature file. Known gaps are tracked
individually rather than as a blanket caveat (`gh-25424 <https://github.com/numpy/numpy/issues/25424>`__). Derived
types remain the largest exclusion (`gh-21160 <https://github.com/numpy/numpy/issues/21160>`__).

Type width mapping across platforms
===================================

Fortran default kinds map to fixed-width NumPy types; C-width types
follow the platform model. On LP64 (Linux/macOS) ``long`` is 64-bit
while on LLP64 (64-bit Windows) it is 32-bit, so signatures relying on
``long`` differ across platforms. Removing the remaining
platform-dependent widths from generated code is tracked in
`gh-21409 <https://github.com/numpy/numpy/issues/21409>`__.

``iso_c_binding`` kinds currently use the same platform C types as in
``numpy/f2py/_isocbind.py`` (``c_size_t`` → ``unsigned``,
``c_intptr_t`` / ``c_ptrdiff_t`` → ``long``, each marked "for now" in
the source). True pointer-width mapping (``npy_uintp`` / ``npy_intp``)
is the intended target and is tracked separately (see `gh-21409
<https://github.com/numpy/numpy/issues/21409>`__ and `gh-25229
<https://github.com/numpy/numpy/issues/25229>`__).

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Fortran declaration
     - Generated C type
     - NumPy type
   * - ``integer`` / ``integer(4)``
     - ``int``
     - ``numpy.int32``
   * - ``integer(8)``
     - ``long_long``
     - ``numpy.int64``
   * - ``integer(kind=c_size_t)``
     - ``unsigned``
     - platform ``unsigned`` (today)
   * - ``integer(kind=c_intptr_t)`` / ``integer(kind=c_ptrdiff_t)``
     - ``long``
     - platform ``long`` (LP64 vs LLP64)
   * - ``real`` / ``real(4)``
     - ``float``
     - ``numpy.float32``
   * - ``real(8)`` / ``double precision``
     - ``double``
     - ``numpy.float64``
   * - ``complex(8)``
     - ``complex_double``
     - ``numpy.complex128``
   * - ``character(len=*)``
     - ``char *`` + ``int`` length
     - ``bytes`` / ``str``

Planned features and internals roadmap
======================================

Larger items, in dependency order. The umbrella lists live on the
issues; this table is the durable index (`gh-20202 <https://github.com/numpy/numpy/issues/20202>`__).

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Feature
     - Tracking
     - Summary
   * - Derived type support
     - `gh-21160 <https://github.com/numpy/numpy/issues/21160>`__
     - Wrap ``type :: t`` declarations end-to-end (parser support
       exists in ``crackfortran``; codegen does not emit accessors).
   * - Allocatable arrays as procedure arguments
     - `gh-19157 <https://github.com/numpy/numpy/issues/19157>`__
     - Only allocatable *module* arrays wrap today; the supported
       pattern is documented in :ref:`f2py-allocatable-arrays`.
   * - Interfaces for external modules and module data
     - `gh-19162 <https://github.com/numpy/numpy/issues/19162>`__
     - Re-exporting entities from third-party modules.
   * - ``operator()`` and ``assignment()`` module procedures
     - `gh-19896 <https://github.com/numpy/numpy/issues/19896>`__
     - Requires derived type support first (`gh-21160 <https://github.com/numpy/numpy/issues/21160>`__).
   * - Generated-C cleanup umbrella
     - `gh-20053 <https://github.com/numpy/numpy/issues/20053>`__
     - Deduplicate ``fortranobject.c`` helpers, drop dead code paths.
   * - Harmonization with the NumPy C API
     - `gh-21161 <https://github.com/numpy/numpy/issues/21161>`__
     - Replace bespoke converters with NumPy equivalents where they
       exist.
   * - Python stable-API (HPy/limited API) generation
     - `gh-21300 <https://github.com/numpy/numpy/issues/21300>`__
     - Blocked on the converter harmonization above.
   * - C-level unit testing (cmocka)
     - `gh-21304 <https://github.com/numpy/numpy/issues/21304>`__
     - Test the generated C directly instead of only through Python.
   * - ``f2py2e`` rework for the meson-only world
     - `gh-30353 <https://github.com/numpy/numpy/issues/30353>`__
     - Simplify CLI/state handling now that distutils is gone.
