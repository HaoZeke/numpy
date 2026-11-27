import numpy as np

from numpy.f2py import crackfortran

from . import util


class TestParamIdiv(util.F2PyTest):
    suffix = ".f"
    code = (
        "      subroutine fill(a)\n"
        "      integer n\n"
        "      parameter (n = (7/2)*2)\n"
        "      integer a(n)\n"
        "Cf2py intent(out) a\n"
        "      a(1) = 7\n"
        "      a(2) = 9\n"
        "      end\n"
    )

    def test_extent(self):
        a = self.module.fill()
        assert a.shape == (6,)
        assert a.dtype == np.int32
        assert int(a[0]) == 7
        assert int(a[1]) == 9


class TestParamIdivFree(util.F2PyTest):
    suffix = ".f90"
    code = (
        "subroutine fill(a)\n"
        "  integer, parameter :: n = (7/2)*2\n"
        "  integer, intent(out) :: a(n)\n"
        "  a(1) = 7\n"
        "  a(2) = 9\n"
        "end subroutine fill\n"
    )

    def test_extent(self):
        a = self.module.fill()
        assert a.shape == (6,)
        assert a.dtype == np.int32
        assert int(a[0]) == 7
        assert int(a[1]) == 9


class TestParamEvalIdiv:
    def test_quotients(self):
        g_params = {
            "kind": crackfortran._kind_func,
            "selected_int_kind": crackfortran._selected_int_kind_func,
            "selected_real_kind": crackfortran._selected_real_kind_func,
        }
        assert crackfortran.param_eval("(7/2)*2", g_params, {}) == 6
        assert crackfortran.param_eval("(-7)/2", g_params, {}) == -3
        assert crackfortran.param_eval("7/(-2)", g_params, {}) == -3
        assert crackfortran.param_eval("8/3", g_params, {}) == 2
        assert crackfortran.param_eval("1.0/2", g_params, {}) == 0.5
        assert crackfortran.param_eval(
            "selected_int_kind(4)", g_params, {}) == 2
        assert crackfortran.param_eval(
            "(/7/2, 1/)", g_params, {}, dimspec="(2)") == {1: 3, 2: 1}
        assert crackfortran._eval_scalar("(7/2)*2", {}) == "6"
        assert crackfortran._eval_scalar("1.0/2", {}) == "0.5"
        assert crackfortran.param_eval("3.14_dp", g_params, {}) == "3.14_dp"
