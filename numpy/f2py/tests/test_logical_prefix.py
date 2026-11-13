import pytest

from numpy.f2py import crackfortran

from . import util


def _result_typespec(path):
    crackfortran.verbose = 0
    post = crackfortran.crackfortran([str(path)])

    def walk(block):
        if isinstance(block, list):
            for item in block:
                found = walk(item)
                if found:
                    return found
            return None
        if not isinstance(block, dict):
            return None
        if block.get("block") == "function" and block.get("name") == "is_ok":
            result = block.get("result") or block.get("name")
            return (block.get("vars") or {}).get(result, {}).get("typespec")
        for child in block.get("body") or []:
            found = walk(child)
            if found:
                return found
        return None

    return walk(post)


def test_elemental_logical_prefix(tmp_path):
    # The name starts with `i`. Dropping `elemental logical` makes the
    # result an integer.
    src = tmp_path / "m.f90"
    src.write_text(
        "module m\n"
        "contains\n"
        "  elemental logical function is_ok(n)\n"
        "    integer, intent(in) :: n\n"
        "    is_ok = n > 0\n"
        "  end function is_ok\n"
        "end module m\n"
    )
    assert _result_typespec(src) == "logical"


@pytest.mark.slow
class TestLogicalPrefix(util.F2PyTest):
    suffix = ".f90"
    code = """
module m
contains
  elemental logical function is_ok(n)
    integer, intent(in) :: n
    is_ok = n > 0
  end function is_ok
end module m
"""

    def test_elemental_logical_value(self):
        assert self.module.m.is_ok(2) == 1
        assert self.module.m.is_ok(-1) == 0
