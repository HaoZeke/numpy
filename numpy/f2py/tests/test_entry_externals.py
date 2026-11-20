import pytest

from . import util


@pytest.mark.slow
class TestEntryExternals(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine run_cb(f, x, y)
  integer, external :: f
  integer, intent(in) :: x
  integer, intent(out) :: y
  y = f(x) + 1
  return
entry run_plain(x, y)
  y = x + 2
end subroutine run_cb

integer function fn_cb(f, x)
  integer, external :: f
  integer, intent(in) :: x
  integer :: fn_plain
  fn_cb = f(x) + 1
  return
entry fn_plain(x)
  fn_plain = x + 3
end function fn_cb
"""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("run_plain", 7),
            ("fn_plain", 8),
        ],
    )
    def test_entry_without_callback(self, name, expected):
        assert getattr(self.module, name)(5) == expected
