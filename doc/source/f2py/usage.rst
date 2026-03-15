===========
Using F2PY
===========

This page lists all command-line options for ``f2py`` and the public API of
the ``numpy.f2py`` module.

Using ``f2py`` as a command-line tool
=====================================

As a command-line tool, ``f2py`` has three modes, selected by the ``-c`` and
``-h`` switches.

1. Signature file generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To scan Fortran sources and generate a signature file, use

.. code-block:: sh

  f2py -h <filename.pyf> <options> <fortran files>   \
    [[ only: <fortran functions>  : ]                \
      [ skip: <fortran functions>  : ]]...           \
    [<fortran files> ...]

.. note::

  A Fortran source file can contain many routines, and not all of them need
  Python wrappers. Use the ``only: .. :`` part to select routines to wrap, or
  the ``skip: .. :`` part to exclude routines.

  The ``skip`` and ``only`` lists are global, not per-file: if any functions
  appear in ``only``, no other functions from any file will be wrapped.

If ``<filename.pyf>`` is specified as ``stdout``, then signatures are written to
standard output instead of a file.

Among other options (see below), the following can be used in this mode:

``--overwrite-signature``
  Overwrites an existing signature file.

2. Extension module construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To construct an extension module, use

.. code-block:: sh

  f2py -m <modulename> <options> <fortran files>   \
    [[ only: <fortran functions>  : ]              \
      [ skip: <fortran functions>  : ]]...          \
    [<fortran files> ...]

The constructed extension module is saved as ``<modulename>module.c`` to the
current directory.

Here ``<fortran files>`` may also contain signature files. Among other options
(see below), the following options can be used in this mode:

``--debug-capi``
  Add debugging hooks to the extension module. At runtime the wrapper writes
  diagnostic information (variable values, execution steps) to standard output.

``-include'<includefile>'``
  Add a CPP ``#include`` statement to the extension module source.
  ``<includefile>`` takes one of the following forms:

  .. code-block:: cpp

    "filename.ext"
    <filename.ext>

  The include statement is inserted just before the wrapper functions, which
  allows using arbitrary C functions in F2PY generated wrappers.

  .. note:: This option is deprecated. Use the ``usercode`` statement to
    specify C code snippets directly in signature files.

``--[no-]wrap-functions``
  Create Fortran subroutine wrappers to Fortran functions.
  ``--wrap-functions`` is default because it ensures maximum portability and
  compiler independence.

``--[no-]freethreading-compatible``
  Declare whether the module requires the GIL. The default is
  ``--no-freethreading-compatible`` for backward compatibility. Verify that
  the Fortran code is thread-safe before passing
  ``--freethreading-compatible``; ``f2py`` does not analyze Fortran code for
  thread safety.

``--include-paths "<path1>:<path2>..."``
  Search include files from given directories.

  .. note:: The paths are to be separated by the correct operating system
            separator :py:data:`~os.pathsep`, that is ``:`` on Linux / MacOS
            and ``;`` on Windows. In ``CMake`` this corresponds to using
            ``$<SEMICOLON>``.

3. Building a module
~~~~~~~~~~~~~~~~~~~~

To build an extension module, use

.. code-block:: sh

  f2py -c <options> <fortran files>       \
    [[ only: <fortran functions>  : ]     \
      [ skip: <fortran functions>  : ]]... \
    [ <fortran/c source files> ] [ <.o, .a, .so files> ]
 
If ``<fortran files>`` contains a signature file, then the source for an
extension module is constructed, all Fortran and C sources are compiled, and
finally all object and library files are linked to the extension module
``<modulename>.so`` which is saved into the current directory.

If ``<fortran files>`` does not contain a signature file, then an extension
module is constructed by scanning all Fortran source codes for routine
signatures, before proceeding to build the extension module.

.. warning::
   ``distutils`` has been removed. Use environment
   variables or native files to interact with ``meson`` instead. See its `FAQ
   <https://mesonbuild.com/howtox.html>`__ for more information.

In addition to the options described above, the following apply in this mode.

.. note::

   .. versionchanged:: 2.5.0
      The ``distutils`` backend has been removed.

Common build flags:

``--backend <backend_type>``
  Legacy option, only ``meson`` is supported.
``--f77flags=<string>``
  Specify F77 compiler flags
``--f90flags=<string>``
  Specify F90 compiler flags
``--debug``
  Compile with debugging information
``-l<libname>``
  Use the library ``<libname>`` when linking.
``-D<macro>[=<defn=1>]``
  Define macro ``<macro>`` as ``<defn>``.
``-U<macro>``
  Define macro ``<macro>``
``-I<dir>``
  Append directory ``<dir>`` to the list of directories searched for include
  files.
``-L<dir>``
  Add directory ``<dir>`` to the list of directories to be searched for
  ``-l``.

``--dep <dependency>``
  Specify a Meson dependency for the module. Pass this option multiple times
  for multiple dependencies.
  Example: ``--dep lapack --dep scalapack``.

.. note::
  
  The ``f2py -c`` option must be applied either to an existing ``.pyf`` file
  (plus the source/object/library files) or one must specify the
  ``-m <modulename>`` option (plus the sources/object/library files). Use one of
  the following options:

  .. code-block:: sh
    
    f2py -c -m fib1 fib1.f

  or

  .. code-block:: sh

    f2py -m fib1 fib1.f -h fib1.pyf
    f2py -c fib1.pyf fib1.f

  For more information, see the `Building C and C++ Extensions`__ Python
  documentation for details.

  __ https://docs.python.org/3/extending/building.html


Non-GCC Fortran compilers may require one or more of the following macros:

.. code-block:: sh

  -DPREPEND_FORTRAN
  -DNO_APPEND_FORTRAN
  -DUPPERCASE_FORTRAN
 
To profile F2PY generated interfaces, use ``-DF2PY_REPORT_ATEXIT``. A timing
report is printed when Python exits. Currently only Linux is supported.

To detect array copies, use ``-DF2PY_REPORT_ON_ARRAY_COPY=<int>``. When the
size of an array argument exceeds ``<int>``, a message is sent to ``stderr``.

Other options
~~~~~~~~~~~~~

``-m <modulename>``
  Name of an extension module. Default is ``untitled``.

.. warning::
   Don't use this option if a signature file (``*.pyf``) is used.

   .. versionchanged:: 1.26.3
      Will ignore ``-m`` if a ``pyf`` file is provided.

``--[no-]lower``
  Do [not] lower the cases in ``<fortran files>``. By default, ``--lower`` is
  assumed with ``-h`` switch, and ``--no-lower`` without the ``-h`` switch.
``-include<header>``
  Writes additional headers in the C wrapper, can be passed multiple times,
  generates #include <header> each time. Note that this is meant to be passed
  in single quotes and without spaces, for example ``'-include<stdbool.h>'``
``--build-dir <dirname>``
  All F2PY generated files are created in ``<dirname>``. Default is
  ``tempfile.mkdtemp()``.
``--f2cmap <filename>``
  Load Fortran-to-C ``KIND`` specifications from the given file.
``--quiet``
  Run quietly.
``--verbose``
  Run with extra verbosity.
``--skip-empty-wrappers``
  Do not generate wrapper files unless required by the inputs.
  This is a backwards compatibility flag to restore pre 1.22.4 behavior.
``-v``
  Print the F2PY version and exit.

Execute ``f2py`` without any options to get an up-to-date list of available
options.

.. _python-module-numpy.f2py:

Python module ``numpy.f2py``
============================

.. warning::

   .. versionchanged:: 2.0.0

      The ``f2py.compile`` function has been removed. Use
      ``subprocess.run`` to call ``python -m numpy.f2py`` and set environment
      variables to interact with ``meson`` as needed.

The following functions are available when ``numpy.f2py`` is used as a module.

.. automodule:: numpy.f2py
    :members:

Building with Meson (Examples)
==============================

Using f2py with Meson
~~~~~~~~~~~~~~~~~~~~~

Meson is the recommended build system for Python extension modules starting
with Python 3.12 and NumPy 2.x.

The example below builds the ``add`` extension from the ``add.f`` and
``add.pyf`` files described in the :ref:`f2py-examples`. A ``.pyf`` file is
not always necessary; in many cases ``f2py`` can infer the annotations.

Project layout::

    f2py_examples/
      meson.build
      add.f
      add.pyf (optional)
      __init__.py  (can be empty)

Example ``meson.build``:

.. code-block:: meson

   project('f2py_examples', 'fortran')

   py = import('python').find_installation()

   # List your Fortran source files
   sources = files('add.pyf', 'add.f')

   # Build the extension by invoking f2py via a custom target
   add_mod = custom_target(
     'add_extension',
     input: sources,
     output: ['add' + py.extension_suffix()],
     command: [
       py.full_path(), '-m', 'numpy.f2py',
       '-c', 'add.pyf', 'add.f',
       '-m', 'add'
     ],
     build_by_default: true
   )

   # Install into site-packages under the f2py_examples package
   install_subdir('.', install_dir: join_paths(py.site_packages_dir(), 'f2py_examples'),
                  strip_directory: false,
                  exclude_files: ['meson.build'])

   # Also install the built extension (place it beside __init__.py)
   install_data(add_mod, install_dir: join_paths(py.site_packages_dir(), 'f2py_examples'))

For advanced usage, see the Meson build guide in the user documentation or
SciPy's build files: https://github.com/scipy/scipy/tree/main/meson.build
