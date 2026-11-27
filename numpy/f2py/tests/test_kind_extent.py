import contextlib
import io
import warnings

import pytest

from numpy.f2py import crackfortran
from numpy.f2py.symbolic import ExprWarning

from . import util


def _dims(tmp_path, code):
    path = tmp_path / "kind_extent.f90"
    path.write_text(code)
    with contextlib.redirect_stdout(io.StringIO()):
        mod = crackfortran.crackfortran([str(path)])
    return mod[0]["vars"]


def test_selected_real_kind_extent_not_folded(tmp_path):
    # gfortran gives 10 for selected_real_kind(15, 308). Folding the
    # call with the local map yields 8, so the extent keeps the call.
    vs = _dims(tmp_path, """
subroutine srk_extent(a)
  integer, intent(out) :: a(selected_real_kind(15, 308))
  a = 1
end subroutine srk_extent
""")
    dim = vs["a"]["dimension"][0]
    assert dim != "8"
    assert "selected_real_kind" in dim


@pytest.mark.parametrize(("code", "expected"), [
    (
        """
subroutine lit_extent(a, b, c, d, e, f, g, mx_supply_curves)
  integer, parameter :: n = 10_8
  integer, intent(in) :: mx_supply_curves
  integer, intent(out) :: a(10_8)
  integer, intent(out) :: b(81_8)
  integer, intent(out) :: c(2*10_8)
  integer, intent(out) :: d(mx_supply_curves)
  integer, intent(out) :: e(kind(10_8))
  integer, intent(out) :: f(n)
  integer, intent(out) :: g(n + 1_8)
  a = 1
  b = 1
  c = 1
  d = 1
  e = 1
  f = 1
  g = 1
end subroutine lit_extent
""",
        {
            "a": ["10"],
            "b": ["81"],
            "c": ["20"],
            "d": ["mx_supply_curves"],
            "e": ["kind(10_8)"],
            "f": ["10"],
            "g": ["11"],
        },
    ),
    (
        # ik is not the default integer kind, so a leftover suffix
        # would remain in the dimension expression.
        """
subroutine named_kind_extent(a, b, c, d)
  use iso_fortran_env, only: int64
  use iso_c_binding
  integer, parameter :: ik = 8
  integer, intent(out) :: a(10_int64)
  integer, intent(out) :: b(3_ik)
  integer, intent(out) :: c(0_ik:9_ik)
  integer, intent(out) :: d(10_c_int)
  a = 1
  b = 1
  c = 1
  d = 1
end subroutine named_kind_extent
""",
        {
            "a": ["10"],
            "b": ["3"],
            "c": ["10"],
            "d": ["10"],
        },
    ),
    (
        """
subroutine quoted_kind_extent(a)
  integer, intent(out) :: a(len("10_8"))
  a = 1
end subroutine quoted_kind_extent
""",
        {"a": ['len("10_8")']},
    ),
    (
        """
subroutine kind_call_extent(a)
  integer, intent(out) :: a(kind(10_8) + 2_8)
  a = 1
end subroutine kind_call_extent
""",
        {"a": ["2 + kind(10_8)"]},
    ),
])
def test_portable_kind_suffix_extent(tmp_path, code, expected):
    vs = _dims(tmp_path, code)
    for name, dim in expected.items():
        assert vs[name]["dimension"] == dim


@pytest.mark.slow
class TestKindExtent(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine literal_kind_extent(a)
  integer, intent(out) :: a(10_8)
  a = 3
end subroutine literal_kind_extent
"""

    def test_literal_kind_extent(self):
        out = self.module.literal_kind_extent()
        assert out.shape == (10,)
        assert int(out[0]) == 3


def test_backslash_quote_kind_suffix():
    fold = crackfortran._fold_extent_kind_suffixes
    expr = 'len("10_8\\") + 2_8'
    assert fold(expr) == 'len("10_8\\") + 2'
    expr = "len('10_8\\') + 2_8"
    assert fold(expr) == "len('10_8\\') + 2"
    expr = "len('10_8''x') + 2_8"
    assert fold(expr) == "len('10_8''x') + 2"


def test_dotted_operator_kind_suffix(tmp_path):
    fold = crackfortran._fold_extent_kind_suffixes
    assert fold('n.eq.3_8') == 'n.eq.3'
    assert fold('3.0_8') == '3.0_8'
    assert fold('1.e5_8') == '1.e5_8'
    assert fold('n.ne.-3_ik') == 'n.ne.-3'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ExprWarning)
        vs = _dims(tmp_path, """
subroutine dotted_kind_extent(a, b, c)
  integer, intent(out) :: a(n.eq.3_8)
  integer, intent(out) :: b(3.0_8)
  integer, intent(out) :: c(1.e5_8)
  a = 1
  b = 1
  c = 1
end subroutine dotted_kind_extent
""")
    assert vs['a']['dimension'] == ['n.eq.3']
    assert vs['b']['dimension'] == ['3.0_8']
    assert vs['c']['dimension'] == ['100000.0_8']


def test_quoted_mask_token_kind_suffix():
    expr = '"@__f2py_lit1__@" + "@__f2py_lit0__@" + 4_8'
    assert crackfortran._fold_extent_kind_suffixes(expr) == (
        '"@__f2py_lit1__@" + "@__f2py_lit0__@" + 4'
    )


@pytest.mark.slow
class TestCIntKindExtent(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine c_int_kind_extent(a)
  use iso_c_binding
  integer, intent(out) :: a(10_c_int)
  a = 4
end subroutine c_int_kind_extent
"""

    def test_c_int_kind_extent(self):
        out = self.module.c_int_kind_extent()
        assert out.shape == (10,)
        assert int(out[0]) == 4
