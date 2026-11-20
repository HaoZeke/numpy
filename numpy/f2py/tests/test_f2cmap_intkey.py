import pytest

from . import util


@pytest.mark.slow
class TestF2CMapIntegerKey(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "f2cmap_intkey", "echo.f90"),
        util.getpath("tests", "src", "f2cmap_intkey", ".f2py_f2cmap"),
    ]

    @pytest.mark.parametrize("value,expected", [(1.25, 2.25), (2.5, 3.5)])
    def test_real10_integer_key(self, value, expected):
        assert self.module.echo(value) == expected
