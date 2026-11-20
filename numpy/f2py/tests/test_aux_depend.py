import pytest

from . import util


@pytest.mark.slow
class TestAuxDepend(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine aux_shape(n, y)
  integer, intent(in) :: n
  !f2py integer, intent(aux), depend(n), check(m > 2) :: m = n + 2
  !f2py real(8), intent(out), dimension(m), depend(m) :: y
  real(8), intent(out) :: y(n + 2)
  y(1) = dble(n)
end subroutine aux_shape
"""

    @pytest.mark.parametrize(("n", "expected_shape"), [
        (3, (5,)),
        (0, None),
    ])
    def test_aux_depend_and_check(self, n, expected_shape):
        if expected_shape is not None:
            out = self.module.aux_shape(n)
            assert out.shape == expected_shape
        else:
            with pytest.raises(Exception, match="auxiliary m"):
                self.module.aux_shape(n)


@pytest.mark.slow
class TestAuxIndependent(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine aux_limit(x, y)
  integer, intent(in) :: x
  !f2py integer, check(x > m) :: x
  !f2py integer, intent(aux) :: m = 5
  integer, intent(out) :: y
  y = x
end subroutine aux_limit
"""

    def test_check_uses_aux_declared_later(self):
        assert self.module.aux_limit(6) == 6
        with pytest.raises(Exception, match=r"x > m"):
            self.module.aux_limit(3)
