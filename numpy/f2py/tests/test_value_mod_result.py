import pytest

from . import util


@pytest.mark.slow
class TestValueModuleResult(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "value_mod_result", "value_mod_result.f90")
    ]

    @pytest.mark.parametrize(
        ("name", "value"),
        [("echo", 1 + 2j), ("scale", 3.5)],
    )
    def test_value_result(self, name, value):
        assert getattr(self.module.valmod, name)(value) == value
