import numpy as np
import pytest

from numpy.f2py.func2subr import fortran_wrapper_name

from . import util


def test_short_wrapper_name_is_unchanged():
    assert fortran_wrapper_name("m", "f") == "f2pywrap_m_f"


def test_long_wrapper_name_is_stable_and_legal():
    raw = "f2pywrap_stdlib_specialfunctions_activations_gelu_approx_grad_sp"
    assert len(raw) > 63
    wrapped = fortran_wrapper_name(
        "stdlib_specialfunctions_activations", "gelu_approx_grad_sp")
    assert len(wrapped) <= 63
    assert wrapped != raw
    assert wrapped == fortran_wrapper_name(
        "stdlib_specialfunctions_activations", "gelu_approx_grad_sp")
    other = fortran_wrapper_name(
        "stdlib_specialfunctions_activations", "gelu_approx_grad_dp")
    assert other != wrapped


@pytest.mark.slow
class TestLongWrapperName(util.F2PyTest):
    # The full wrapper symbol is 64 characters and is not a legal
    # Fortran name. The shortened symbol still calls the function.
    suffix = ".f90"
    code = """
module stdlib_specialfunctions_activations
contains
  real function gelu_approx_grad_sp(x)
    real, intent(in) :: x
    gelu_approx_grad_sp = x + 1.0
  end function
end module
"""

    def test_long_name_is_callable(self):
        got = self.module.stdlib_specialfunctions_activations.gelu_approx_grad_sp(
            np.float32(2.0))
        assert got == np.float32(3.0)
