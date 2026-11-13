import pytest

from . import util


@pytest.mark.slow
class TestDeferredCharInout(util.F2PyTest):
    # character(len=:) allocatable is a descriptor, not a fixed buffer.
    # Passing a char* and a length by value segfaults when Fortran allocates.
    suffix = ".f90"
    code = """
subroutine grow(s)
  character(len=:), allocatable, intent(inout) :: s
  if (allocated(s)) deallocate(s)
  allocate(character(len=5) :: s)
  s = 'hello'
end subroutine

subroutine keep(s)
  character(len=:), allocatable, intent(inout) :: s
  if (.not. allocated(s)) then
    allocate(character(len=3) :: s)
    s = 'abc'
  else
    s = 'xyz'
  end if
end subroutine
"""

    def test_reallocate(self):
        got = self.module.grow("hi")
        assert got == "hello"

    def test_reallocate_from_empty(self):
        got = self.module.grow("")
        assert got == "hello"

    def test_overwrite_allocated(self):
        got = self.module.keep("ab")
        assert got == "xyz"

    def test_allocate_when_unallocated(self):
        got = self.module.keep("")
        assert got == "abc"
