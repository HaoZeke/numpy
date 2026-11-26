import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestUsedModuleDefinesType(util.F2PyTest):
    suffix = ".f90"
    code = """
module m
  implicit none
  type :: pair
    integer :: pub
    integer, private :: hid
  end type pair
  integer :: n = 1
contains
  subroutine inc()
    n = n + 1
  end subroutine inc
end module m

subroutine peek_only(k)
  use m, only: n
  integer, intent(out) :: k
  k = n
end subroutine peek_only

subroutine peek_renamed(k)
  use m, only: val => n
  integer, intent(out) :: k
  k = val
end subroutine peek_renamed

module types_only
  implicit none
  type :: pair
    integer :: pub
  end type pair
end module types_only

subroutine touch_types(k)
  use types_only
  integer, intent(out) :: k
  k = 1
end subroutine touch_types

module with_arg
  implicit none
  type :: pair
    integer :: pub
  end type pair
  integer :: n = 1
contains
  subroutine take_pair(p)
    type(pair), intent(in) :: p
  end subroutine take_pair
  subroutine bump()
    n = n + 1
  end subroutine bump
end module with_arg

subroutine peek_arg(k)
  use with_arg, only: n
  integer, intent(out) :: k
  k = n
end subroutine peek_arg

module with_class
  implicit none
  type :: pair
    integer :: pub
  end type pair
  integer :: n = 1
contains
  subroutine take_class(p)
    class(pair), intent(in) :: p
  end subroutine take_class
  subroutine bump()
    n = n + 1
  end subroutine bump
end module with_class

subroutine peek_class(k)
  use with_class, only: n
  integer, intent(out) :: k
  k = n
end subroutine peek_class

module with_priv
  implicit none
  type :: pair
    integer :: pub
  end type pair
  type(pair), private :: hidden
  integer :: n = 1
contains
  subroutine inc()
    n = n + 1
  end subroutine inc
end module with_priv

subroutine peek_priv(k)
  use with_priv, only: n
  integer, intent(out) :: k
  k = n
end subroutine peek_priv
"""

    @pytest.mark.parametrize(
        ("modname", "reader"),
        [
            ("m", "peek_only"),
            ("m", "peek_renamed"),
            ("with_priv", "peek_priv"),
        ],
    )
    def test_public_data_stays_available(self, modname, reader):
        mod = getattr(self.module, modname)
        assert not hasattr(mod, "hidden")
        mod.n = np.array(1, dtype=np.int32)
        assert int(np.array(mod.n)) == 1
        mod.inc()
        assert int(np.array(mod.n)) == 2
        assert int(getattr(self.module, reader)()) == 2

    @pytest.mark.parametrize(
        ("absent", "reader"),
        [
            ("types_only", "touch_types"),
            ("with_arg", "peek_arg"),
            ("with_class", "peek_class"),
        ],
    )
    def test_skipped_module_leaves_the_rest(self, absent, reader):
        assert not hasattr(self.module, absent)
        assert int(getattr(self.module, reader)()) == 1
