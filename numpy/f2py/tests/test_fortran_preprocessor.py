import numpy as np
import pytest

from numpy.f2py import crackfortran

from . import util


def _parameter(blocks, module, name):
    found = next(b for b in blocks if b.get("name") == module)
    return found["vars"][name]["="]


def test_preprocessor_keeps_else_branch(tmp_path):
    src = tmp_path / "kinds.F90"
    src.write_text(
        "module kinds\n"
        "#ifdef USE32\n"
        "  integer, parameter :: wp = 4\n"
        "#else\n"
        "  integer, parameter :: wp = 8\n"
        "#endif\n"
        "end module kinds\n"
    )
    blocks = crackfortran.crackfortran([str(src)])
    assert str(_parameter(blocks, "kinds", "wp")) == "8"


def test_preprocessor_honors_define(tmp_path):
    src = tmp_path / "kinds.F90"
    src.write_text(
        "module kinds\n"
        "#ifdef USE32\n"
        "  integer, parameter :: wp = 4\n"
        "#else\n"
        "  integer, parameter :: wp = 8\n"
        "#endif\n"
        "end module kinds\n"
    )
    crackfortran.cpp_macros = [("USE32", None)]
    try:
        blocks = crackfortran.crackfortran([str(src)])
    finally:
        crackfortran.cpp_macros = []
    assert str(_parameter(blocks, "kinds", "wp")) == "4"


def test_lowercase_suffix_is_not_preprocessed(tmp_path):
    # .f90 is not a preprocessed suffix. Both assignments are visible and
    # the first one is kept.
    src = tmp_path / "kinds.f90"
    src.write_text(
        "module kinds\n"
        "  integer, parameter :: wp = 4\n"
        "  integer, parameter :: wp = 8\n"
        "end module kinds\n"
    )
    blocks = crackfortran.crackfortran([str(src)])
    assert str(_parameter(blocks, "kinds", "wp")) == "4"


@pytest.mark.slow
class TestPreprocessedKind(util.F2PyTest):
    # No -D: gfortran keeps the #else parameter, real64.
    suffix = ".F90"
    code = """
module kinds
  use, intrinsic :: iso_fortran_env
#ifdef USE32
  integer, parameter :: wp = real32
#else
  integer, parameter :: wp = real64
#endif
end module kinds
module m
  use kinds, only: wp
contains
  subroutine reveal(x)
    real(wp), intent(out) :: x
    x = 1.0_wp
  end subroutine reveal
end module m
"""

    def test_else_branch_is_real64(self):
        x = self.module.m.reveal()
        assert np.dtype(np.asarray(x).dtype) == np.dtype(np.float64)
        assert float(x) == 1.0
