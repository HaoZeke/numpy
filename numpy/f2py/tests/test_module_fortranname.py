import pytest

from . import util


@pytest.mark.slow
class TestModuleFortranName(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "routines", "modfortranname.f90"),
        util.getpath("tests", "src", "routines", "modfortranname.pyf"),
    ]
    module_name = "modfortranname"

    @pytest.mark.parametrize(("name", "args", "expected"), [
        ("pyinc", (4,), 5),
        ("pyadd", (2, 3), 5),
    ])
    def test_renamed_module_subroutine(self, name, args, expected):
        routine = getattr(self.module.modn, name)
        assert routine(*args) == expected
