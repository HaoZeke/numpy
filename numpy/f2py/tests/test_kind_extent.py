import contextlib
import io

import pytest

from numpy.f2py import crackfortran

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


def test_portable_kind_suffix_extent(tmp_path):
    vs = _dims(tmp_path, """
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
""")
    assert vs["a"]["dimension"] == ["10"]
    assert vs["b"]["dimension"] == ["81"]
    assert vs["c"]["dimension"] == ["20"]
    assert vs["d"]["dimension"] == ["mx_supply_curves"]
    assert vs["e"]["dimension"] == ["kind(10_8)"]
    assert vs["f"]["dimension"] == ["10"]
    assert vs["g"]["dimension"] == ["11"]


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
