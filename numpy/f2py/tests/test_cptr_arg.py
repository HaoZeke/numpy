import pytest

from . import util


@pytest.mark.slow
class TestCptrArg(util.F2PyTest):
    sources = [util.getpath("tests", "src", "cptr", "cptr_arg.f90")]

    @pytest.mark.parametrize("name", ["ptr_bits", "funptr_bits"])
    def test_cptr_arg(self, name):
        fn = getattr(self.module.ptrmod, name)
        assert fn(0) == 0
        assert fn(0x1234) == 0x1234
