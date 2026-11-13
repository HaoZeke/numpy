import pytest

from numpy.f2py import crackfortran

from . import util


_SRC = """
subroutine s(n, a)
  integer, intent(out) :: n
  class(*), intent(in) :: a
  select type (a)
  type is (integer)
    n = 1
  type is (real)
    n = 2
  class default
    n = 0
  end select
end subroutine

subroutine t(n)
  integer, intent(out) :: n
  n = 7
end subroutine
"""


def _routine_names(blocks):
    names = []

    def walk(node):
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        if node.get("block") in ("function", "subroutine"):
            names.append(node.get("name"))
        walk(node.get("body") or [])

    walk(blocks)
    return names


def test_class_star_is_skipped(tmp_path, capfd):
    src = tmp_path / "class_star.f90"
    src.write_text(_SRC)
    blocks = crackfortran.crackfortran([str(src)])
    out, err = capfd.readouterr()
    names = _routine_names(blocks)
    assert "s" not in names
    assert "t" in names
    assert 'routine "s" has unsupported polymorphic argument a (class(*))' in err
    assert "Skipping" in err
    # The warning is the skip. The routine must not be described as a float.
    assert "input float" not in out


@pytest.mark.slow
class TestClassStarSkipped(util.F2PyTest):
    suffix = ".f90"
    code = _SRC

    def test_other_routine_still_callable(self):
        assert self.module.t() == 7
        assert not hasattr(self.module, "s")
