``f2py -c`` passes ``-D`` and ``-U`` macros to the compilers again
------------------------------------------------------------------
The ``meson`` backend of ``f2py -c`` silently dropped preprocessor macros given
as ``-D<macro>[=<defn>]`` and ``-U<macro>``. They now reach both the C and the
Fortran compiler, as they did with the removed ``distutils`` backend. ``#ifdef``
blocks in preprocessed Fortran sources such as ``.F90`` files and the
documented C macros such as ``-DF2PY_REPORT_ON_ARRAY_COPY=<int>`` work again.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
