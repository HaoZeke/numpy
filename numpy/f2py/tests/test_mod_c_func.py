import pytest

from . import util


@pytest.mark.slow
class TestModCFunc(util.F2PyTest):
    module_name = "mod_c_func"
    sources = [
        util.getpath("tests", "src", "mod_c_func", "add.f90"),
        util.getpath("tests", "src", "mod_c_func", "add.pyf"),
    ]

    @pytest.mark.parametrize("x,expected", [(3, 4), (0, 1)])
    def test_callstatement_return(self, x, expected):
        assert self.module.m.add(x) == expected
