``f2py`` usercode tests run on macOS again
------------------------------------------
The ``f2py`` ``.pyf`` usercode tests are no longer skipped on macOS.
That covers ``test_semicolon_split.py`` and ``TestCReturnReal``.
They were disabled in 2018 for import failures.
They showed up only when the whole ``f2py`` suite ran together.
