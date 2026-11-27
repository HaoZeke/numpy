import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestCharLenDefault(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine dclen(s, n, out)
  character(len=*), intent(in) :: s
  integer, intent(out) :: out
  integer :: n
  !f2py integer optional, intent(in), depend(s) :: n = len(s)
  out = n
end

subroutine dclen1(s, n, out)
  character(len=*), intent(in) :: s
  integer, intent(out) :: out
  integer :: n
  !f2py integer optional, intent(in), depend(s) :: n = len(s) + 1
  out = n
end

subroutine alen(a, n, out)
  integer, intent(in) :: n
  real, intent(in) :: a(n)
  integer, intent(out) :: out
  !f2py integer optional, intent(in), depend(a) :: n = len(a)
  out = n
end

module mlen
contains
  subroutine mclen(s, n, out)
    character(len=*), intent(in) :: s
    integer, intent(out) :: out
    integer :: n
    !f2py integer optional, intent(in), depend(s) :: n = len(s)
    out = n
  end subroutine mclen
end module mlen
"""

    def test_scalar_character_len(self):
        assert self.module.dclen("ab") == 2
        assert self.module.dclen("ab", 1) == 1
        assert self.module.dclen(s="abcd", n=3) == 3
        assert self.module.dclen1("ab") == 3
        assert self.module.dclen1(n=4, s="ab") == 4

    def test_array_len(self):
        a = np.zeros(4, dtype=np.float32)
        assert self.module.alen(a) == 4
        assert self.module.alen(a, 2) == 2
        assert self.module.alen(n=3, a=a) == 3

    def test_module_character_len(self):
        assert self.module.mlen.mclen("xyz") == 3
        assert self.module.mlen.mclen("xyz", 1) == 1
        assert self.module.mlen.mclen(n=2, s="xyz") == 2
