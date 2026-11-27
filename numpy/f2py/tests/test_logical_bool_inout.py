import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestLogicalKind1Bool(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine flip1(a)
  implicit none
  logical(kind=1), intent(inout) :: a(3)
  a = .not. a
end subroutine flip1

subroutine copy1(a, b)
  implicit none
  logical(kind=1), intent(in) :: a(3)
  logical(kind=1), intent(out) :: b(3)
  b = a
end subroutine copy1

subroutine flip2(a)
  implicit none
  logical(kind=2), intent(inout) :: a(3)
  a = .not. a
end subroutine flip2
"""

    def test_inout_bool(self):
        a = np.array([True, False, True])
        self.module.flip1(a)
        assert a.dtype == np.bool_
        assert a.tolist() == [False, True, False]

    def test_inout_int8(self):
        a = np.array([1, 0, 1], dtype=np.int8)
        self.module.flip1(a)
        assert a.tolist() == [0, 1, 0]

    def test_in_bool_out_byte(self):
        a = np.array([True, False, True])
        b = self.module.copy1(a)
        assert b.dtype == np.int8
        assert b.tolist() == [1, 0, 1]
        assert a.tolist() == [True, False, True]

    def test_kind2_bool_rejected(self):
        a = np.array([True, False, True])
        with pytest.raises(ValueError):
            self.module.flip2(a)
