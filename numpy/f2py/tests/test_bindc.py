import numpy as np
import numpy.f2py.crackfortran as crackfortran
import pytest

from . import util


def _routine(src, name):
    mod = crackfortran.crackfortran([src])

    def walk(block):
        if isinstance(block, list):
            for item in block:
                found = walk(item)
                if found is not None:
                    return found
            return None
        if not isinstance(block, dict):
            return None
        if block.get('name') == name and block.get('block') in ('function', 'subroutine'):
            return block
        return walk(block.get('body') or [])

    return walk(mod)


def test_free_bindc_uses_c_symbol(tmp_path):
    src = tmp_path / "free.f90"
    src.write_text(
        'function f(x) bind(c, name="f_c")\n'
        "  use iso_c_binding\n"
        "  real(c_double), intent(in) :: x\n"
        "  real(c_double) :: f\n"
        "  f = x\n"
        "end function\n"
    )
    rout = _routine(str(src), "f")
    assert rout["f2pyenhancements"]["fortranname"] == "f_c"
    assert "c" in rout["intent"]


def test_module_bindc_stays_a_fortran_call(tmp_path):
    src = tmp_path / "mod.f90"
    src.write_text(
        "module m\n"
        "contains\n"
        '  function g(x) bind(c, name="g_c")\n'
        "    use iso_c_binding\n"
        "    real(c_double), intent(in) :: x\n"
        "    real(c_double) :: g\n"
        "    g = x\n"
        "  end function\n"
        "end module\n"
    )
    rout = _routine(str(src), "g")
    assert "c" not in rout.get("intent", [])
    assert "fortranname" not in rout.get("f2pyenhancements", {})


@pytest.mark.slow
class TestFreeBindC(util.F2PyTest):
    sources = [util.getpath("tests", "src", "bindc", "free.f90")]

    def test_function_and_subroutine(self):
        assert self.module.f(2.0) == 3.0
        x = np.array(1.0)
        assert self.module.s(x) is None
        assert x == 4.0
