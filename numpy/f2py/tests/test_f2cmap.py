import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestF2Cmap(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "f2cmap", "isoFortranEnvMap.f90"),
        util.getpath("tests", "src", "f2cmap", ".f2py_f2cmap")
    ]

    # gh-15095
    def test_gh15095(self):
        inp = np.ones(3)
        out = self.module.func1(inp)
        exp_out = 3
        assert out == exp_out


@pytest.mark.slow
@pytest.mark.skipif(
    np.finfo(np.longdouble).nmant != 63,
    reason="needs 80-bit long double",
)
class TestKindParameterMap(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "f2cmap", "kindparam", "kinds.f90"),
        util.getpath("tests", "src", "f2cmap", "kindparam", "calc.f90"),
        util.getpath("tests", "src", "f2cmap", "kindparam", ".f2py_f2cmap"),
    ]

    @pytest.mark.parametrize("name", ["wider", "widerep"])
    def test_parameter_name(self, name):
        x = np.longdouble(1) + np.longdouble(2) ** -60
        assert getattr(self.module.calc, name)(x) == 1
