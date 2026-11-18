import pytest

from numpy.f2py.tests import util


@pytest.mark.slow
@pytest.mark.parametrize(
    ("tag", "suffix", "source"),
    [
        (
            "free",
            ".f90",
            "subroutine nonzero(n)\n"
            "  integer, intent(in) :: n\n"
            "!f2py check(n!=0) n\n"
            "end subroutine nonzero\n",
        ),
        (
            "fixed",
            ".f",
            "      subroutine nonzero(n)\n"
            "      integer n\n"
            "Cf2py intent(in) n\n"
            "Cf2py check(n!=0) n\n"
            "      end\n",
        ),
    ],
)
def test_check_not_equal(tag, suffix, source):
    mod = util.build_code(
        source, suffix=suffix, module_name=f"check_neq_{tag}")
    mod.nonzero(2)
    with pytest.raises(Exception, match="failed for"):
        mod.nonzero(0)
