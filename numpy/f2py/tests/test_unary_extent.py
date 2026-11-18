import pytest

from . import util


@pytest.mark.slow
@pytest.mark.parametrize(
    "suffix, source",
    [
        (
            ".f90",
            """
subroutine negdim(a)
  integer, parameter :: n = -3
  integer, intent(out) :: a(2 * n + 10)
  a = [1, 2, 3, 4]
end subroutine negdim
""",
        ),
        (
            ".f",
            """      subroutine negdim(a)
      integer n,i
      parameter (n=-3)
      integer a(2*n+10)
Cf2py intent(out) a
      do 10 i=1,4
      a(i)=i
 10   continue
      end
""",
        ),
    ],
    ids=["free", "fixed"],
)
def test_negative_parameter_extent(suffix, source):
    module = util.build_code(
        source, suffix=suffix, module_name=f"negative_extent_{suffix[2:]}"
    )
    assert module.negdim().tolist() == [1, 2, 3, 4]
