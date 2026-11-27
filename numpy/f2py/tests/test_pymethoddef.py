import pytest

from . import util


@pytest.mark.slow
class TestPyMethodDef(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "pymethoddef", "twometh.pyf"),
    ]
    module_name = "twometh"

    def test_both_methods(self):
        assert self.module.a() == 2
        assert self.module.b() == 3
