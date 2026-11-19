import pytest

from . import util


@pytest.mark.slow
class TestModuleDummyFortranName(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "routines", "moddummy.f90"),
        util.getpath("tests", "src", "routines", "moddummy.pyf"),
    ]
    module_name = "moddummy"

    @pytest.mark.parametrize(("name", "args", "expected"), [
        ("add", (2, 3), 5),
        ("realinc", (4,), 5),
    ])
    def test_dummy_module_subroutine(self, name, args, expected):
        routine = getattr(self.module.modd, name)
        assert routine(*args) == expected
