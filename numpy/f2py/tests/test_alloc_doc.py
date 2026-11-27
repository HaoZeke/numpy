import pytest

from . import util


@pytest.mark.slow
class TestAllocatableDoc(util.F2PyTest):
    suffix = ".f90"
    code = """
module m
  integer, allocatable :: a(:,:)
contains
  subroutine fill(n)
    integer, intent(in) :: n
    if (allocated(a)) deallocate(a)
    allocate(a(n, n + 1))
    a = n
  end subroutine fill
end module
"""

    @pytest.mark.parametrize("n,shape", [(2, "(2,3)"), (4, "(4,5)")])
    def test_doc_follows_allocation(self, n, shape):
        _ = self.module.m.__doc__
        self.module.m.fill(n)
        line = next(ln for ln in self.module.m.__doc__.splitlines()
                    if ln.strip().startswith("a :"))
        assert line.strip() == f"a : 'i'-array{shape}"
