import pytest

from . import util


@pytest.mark.slow
class TestBindCNameCase(util.F2PyTest):
    sources = [util.getpath("tests", "src", "bindc", "name_case.f90")]

    @pytest.mark.parametrize("fn,inc", [("f", 2), ("g", 3)])
    def test_written_c_name(self, fn, inc):
        assert getattr(self.module, fn)(4) == 4 + inc
