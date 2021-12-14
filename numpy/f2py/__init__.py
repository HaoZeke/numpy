"""Fortran to Python Interface Generator."""

import sys

from numpy.f2py import frontend
from numpy.f2py import __version__
# Helpers
from numpy.f2py.utils.pathhelper import get_include
# Build tool
from numpy.f2py.utils.npdist import compile
from numpy.f2py.frontend.f2py2e import main

run_main = frontend.f2py2e.run_main

if __name__ == "__main__":
    sys.exit(main())

if sys.version_info[:2] >= (3, 7):
    # module level getattr is only supported in 3.7 onwards
    # https://www.python.org/dev/peps/pep-0562/
    def __getattr__(attr):

        # Avoid importing things that aren't needed for building
        # which might import the main numpy module
        if attr == "f2py_testing":
            import numpy.f2py.f2py_testing as f2py_testing
            return f2py_testing

        elif attr == "test":
            from numpy._pytesttester import PytestTester
            test = PytestTester(__name__)
            return test

        else:
            raise AttributeError("module {!r} has no attribute "
                                 "{!r}".format(__name__, attr))

    def __dir__():
        return list(globals().keys() | {"f2py_testing", "test"})

else:
    from . import f2py_testing

    from numpy._pytesttester import PytestTester
    test = PytestTester(__name__)
    del PytestTester
