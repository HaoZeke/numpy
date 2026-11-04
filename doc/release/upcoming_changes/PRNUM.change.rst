``f2py`` usercode tests run on macOS again
------------------------------------------
The ``f2py`` tests built from ``.pyf`` files with ``usercode`` blocks, in
``test_semicolon_split.py`` and ``TestCReturnReal``, are no longer skipped on
macOS. They were disabled in 2018 for import failures that only appeared when
the whole ``f2py`` test suite ran together.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
