import pytest

from numpy.f2py import crackfortran

from . import util


@pytest.mark.slow
class TestImplicitEntry(util.F2PyTest):
    # gh-13356: the IMPLICIT statement types the ENTRY result
    sources = [util.getpath("tests", "src", "entry", "gh13356.f")]

    def test_entry(self):
        assert self.module.eta0(2.0, [3.0]) == 5.0
        # 0.1 * 3 rounds differently in single precision
        assert self.module.eta1(0.1, 3) == 0.1 * 3


@pytest.mark.slow
class TestDefaultTypedEntry(util.F2PyTest):
    # The default implicit rules make this ENTRY real in an integer function
    sources = [util.getpath("tests", "src", "entry", "default_typing.f")]

    def test_entry(self):
        assert self.module.ifun(3) == 6
        assert self.module.rfun(3) == 1.5

    def test_signature(self):
        mod = crackfortran.crackfortran([str(self.sources[0])])
        assert mod[0]["vars"]["rfun"]["typespec"] == "real"
        assert "real :: rfun" in crackfortran.crack2fortran(mod)


@pytest.mark.slow
class TestEntryInResultFunction(util.F2PyTest):
    # Each ENTRY returns its own type, not that of the function RESULT
    sources = [util.getpath("tests", "src", "entry", "result_entry.f90")]

    def test_entry(self):
        assert self.module.f(1.5) == 3.0
        assert self.module.g(1.5) == 4
        assert self.module.h(1.5) == 6.0

    def test_docstring(self):
        assert self.module.g.__doc__.startswith("g = g(x)")
        assert "g : int" in self.module.g.__doc__
