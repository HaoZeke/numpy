import pytest

from . import util


@pytest.mark.slow
class TestSizeParameter(util.F2PyTest):
    suffix = ".f90"
    code = """
      subroutine from_size(a)
        integer, parameter :: p(3) = (/2, 4, 6/)
        integer, intent(out) :: a(size(p))
        a = p
      end subroutine from_size

      subroutine from_bounds(a)
        integer, parameter :: p(0:4) = (/1, 2, 3, 4, 5/)
        integer, intent(out) :: a(size(p, 1))
        a = p
      end subroutine from_bounds

      subroutine from_negative_bounds(a)
        integer, parameter :: p(-2:2) = (/1, 2, 3, 4, 5/)
        integer, intent(out) :: a(size(p))
        a = p
      end subroutine from_negative_bounds

      subroutine from_expression(a)
        integer, parameter :: n = 2
        integer, parameter :: p(n + 1) = (/1, 2, 3/)
        integer, intent(out) :: a(size(p))
        a = p
      end subroutine from_expression

      subroutine from_kind(a)
        integer, parameter :: p(3_4) = (/1, 2, 3/)
        integer, intent(out) :: a(size(p))
        a = p
      end subroutine from_kind

      subroutine twice_size(a)
        integer, parameter :: p(3) = (/1, 2, 3/)
        integer, intent(out) :: a(2 * size(p))
        a = (/p, p/)
      end subroutine twice_size

      subroutine rank_two(total, columns)
        integer, parameter :: p(2, 3) = reshape((/1, 2, 3, 4, 5, 6/), (/2, 3/))
        integer, intent(out) :: total(size(p)), columns(size(p, 2))
        total = reshape(p, (/size(p)/))
        columns = p(1, :)
      end subroutine rank_two

      module parameter_source
        integer, parameter :: p(3) = (/7, 8, 9/)
      end module parameter_source

      subroutine used_parameter(a)
        use parameter_source
        integer, intent(out) :: a(size(p))
        a = p
      end subroutine used_parameter

      subroutine from_character(a)
        character(len=2), parameter :: p(3) = (/'aa', 'bb', 'cc'/)
        character(len=2), intent(out) :: a(size(p))
        a = p
      end subroutine from_character

      subroutine dummy_size(x, y)
        integer, intent(in) :: x(:)
        integer, intent(out) :: y(size(x))
        y = x
      end subroutine dummy_size
    """

    @pytest.mark.parametrize("name,expect", [
        ("from_size", [2, 4, 6]),
        ("from_bounds", [1, 2, 3, 4, 5]),
        ("from_negative_bounds", [1, 2, 3, 4, 5]),
        ("from_expression", [1, 2, 3]),
        ("from_kind", [1, 2, 3]),
        ("twice_size", [1, 2, 3, 1, 2, 3]),
        ("rank_two", ([1, 2, 3, 4, 5, 6], [1, 3, 5])),
        ("used_parameter", [7, 8, 9]),
        ("from_character", [b"aa", b"bb", b"cc"]),
    ])
    def test_extent(self, name, expect):
        got = getattr(self.module, name)()
        if isinstance(expect, tuple):
            got = tuple(list(item) for item in got)
        else:
            got = list(got)
        assert got == expect

    def test_dummy_size(self):
        assert list(self.module.dummy_size([1, 2, 3])) == [1, 2, 3]
