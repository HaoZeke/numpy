import pytest

from . import util


@pytest.mark.slow
class TestRenamedFunc(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "routines", "funcfortranname.f"),
        util.getpath("tests", "src", "routines", "funcfortranname.pyf"),
    ]
    module_name = "funcfortranname"

    def test_gh25799(self):
        assert dir(self.module)
        assert self.module.funcfortranname_default(200, 12) == 212


@pytest.mark.slow
class TestRenamedSubroutine(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "routines", "subrout.f"),
        util.getpath("tests", "src", "routines", "subrout.pyf"),
    ]
    module_name = "subrout"

    def test_renamed_subroutine(self):
        assert dir(self.module)
        assert self.module.subrout_default(200, 12) == 212


@pytest.mark.slow
class TestDummyFunction(util.F2PyTest):
    suffix = ".pyf"
    module_name = "dummyfunc"
    code = f"""
python module {module_name}
    interface
        function triple(x)
            fortranname
            callstatement triple = 3*x
            real*8 intent(in) :: x
            real*8 :: triple
        end function triple
    end interface
end python module {module_name}
    """

    @pytest.mark.parametrize("x", [2.0, -1.5])
    def test_dummy_function(self, x):
        assert self.module.triple(x) == 3 * x
