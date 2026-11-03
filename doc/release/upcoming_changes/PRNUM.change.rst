``f2py`` usercode tests run on macOS again
------------------------------------------
The ``f2py`` tests built from ``.pyf`` files with ``usercode`` blocks, in
``test_semicolon_split.py`` and ``TestCReturnReal``, are no longer skipped on
macOS. They were disabled in 2018 for import failures that only appeared when
the whole ``f2py`` test suite ran together.
