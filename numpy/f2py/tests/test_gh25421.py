import pytest

from numpy.f2py import crackfortran

from . import util


def test_odd_bound_keeps_truncating_division(tmp_path):
    # gh-25421. ``0:n/2*2`` has extent ``n/2*2+1``, not ``n+1``.
    src = tmp_path / "bound.f90"
    src.write_text(
        "subroutine probe(n, a)\n"
        "  integer, intent(in) :: n\n"
        "  real, intent(out) :: a(0:n/2*2)\n"
        "end subroutine\n"
    )
    block = crackfortran.crackfortran([str(src)])[0]
    dim = block["vars"]["a"]["dimension"][0].replace(" ", "")
    assert "n/2*2" in dim
    assert dim != "1+n"


@pytest.mark.slow
class TestIntegerExtent(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "gh25421.f90")]

    def test_odd_n_matches_fortran(self):
        # n=5 -> upper bound 5/2*2 = 4, so 0:4 has 5 elements.
        assert self.module.probe(5).shape == (5,)
        assert self.module.probe(4).shape == (5,)
