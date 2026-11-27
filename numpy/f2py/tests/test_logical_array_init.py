import pytest

from . import util


@pytest.mark.slow
class TestLogicalArrayDefault(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine count_l1(n, k, a)
  integer, intent(in) :: n
  integer, intent(out) :: k
  logical(kind=1), intent(in) :: a(n)
  !f2py logical(kind=1), optional, intent(in), depend(n) :: a = .true.
  integer :: i
  k = 0
  do i = 1, n
    if (a(i)) k = k + 1
  end do
end subroutine count_l1

subroutine count_l8(n, k, a)
  integer, intent(in) :: n
  integer, intent(out) :: k
  logical(kind=8), intent(in) :: a(n)
  !f2py logical(kind=8), optional, intent(in), depend(n) :: a = .false.
  integer :: i
  k = 0
  do i = 1, n
    if (a(i)) k = k + 1
  end do
end subroutine count_l8

subroutine count_l1k(n, k, a)
  integer, intent(in) :: n
  integer, intent(out) :: k
  logical(kind=1), intent(in) :: a(n)
  !f2py logical(kind=1), optional, intent(in), depend(n) :: a = .TRUE._1
  integer :: i
  k = 0
  do i = 1, n
    if (a(i)) k = k + 1
  end do
end subroutine count_l1k

subroutine count_l8k(n, k, a)
  integer, intent(in) :: n
  integer, intent(out) :: k
  logical(kind=8), intent(in) :: a(n)
  !f2py logical(kind=8), optional, intent(in), depend(n) :: a = .false._8
  integer :: i
  k = 0
  do i = 1, n
    if (a(i)) k = k + 1
  end do
end subroutine count_l8k
"""

    @pytest.mark.parametrize(("name", "expected"), [
        ("count_l1", 4),
        ("count_l8", 0),
        ("count_l1k", 4),
        ("count_l8k", 0),
    ])
    def test_omitted_logical_array_default(self, name, expected):
        assert getattr(self.module, name)(4) == expected
