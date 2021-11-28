import os
import pytest

import numpy.f2py
import numpy as np

from . import util


class TestIntentInOut(util.F2PyTest):
    # Check that intent(in out) translates as intent(inout)
    sources = [util.getpath("tests", "src", "regression", "inout.f90")]

    @pytest.mark.slow
    def test_inout(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float32)[::2]
        pytest.raises(ValueError, self.module.foo, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float32)
        self.module.foo(x)
        assert np.allclose(x, [3, 1, 2])


class TestNumpyVersionAttribute(util.F2PyTest):
    # Check that th attribute __f2py_numpy_version__ is present
    # in the compiled module and that has the value np.__version__.
    sources = [util.getpath("tests", "src", "regression", "inout.f90")]

    @pytest.mark.slow
    def test_numpy_version_attribute(self):

        # Check that self.module has an attribute named "__f2py_numpy_version__"
        assert hasattr(self.module, "__f2py_numpy_version__")

        # Check that the attribute __f2py_numpy_version__ is a string
        assert isinstance(self.module.__f2py_numpy_version__, str)

        # Check that __f2py_numpy_version__ has the value numpy.__version__
        assert np.__version__ == self.module.__f2py_numpy_version__


def test_include_path():
    incdir = numpy.f2py.get_include()
    fnames_in_dir = os.listdir(incdir)
    for fname in ("fortranobject.c", "fortranobject.h"):
        assert fname in fnames_in_dir
