import textwrap

import pytest

import numpy as np
from numpy.f2py import crackfortran

from . import util


def _routine(block, name):
    if block.get("name") == name and block.get("block") in (
            "function", "subroutine"):
        return block
    for child in block.get("body") or []:
        found = _routine(child, name)
        if found:
            return found
    return None


def test_spec_statements_keep_attributes(tmp_path):
    src = tmp_path / "attr.f90"
    src.write_text(textwrap.dedent("""\
        module m
          implicit none
          integer :: x
          integer :: y
          integer :: z
          protected :: x
          protected z
          volatile :: y
          asynchronous :: y
        contains
          subroutine copy(a, b)
            real, intent(in) :: a(:)
            real, intent(out) :: b
            contiguous :: a
            b = a(1)
          end subroutine copy
        end module m
        """))
    blocks = crackfortran.crackfortran([str(src)])
    assert len(blocks) == 1
    mod = blocks[0]
    assert mod["vars"]["x"]["attrspec"] == ["protected"]
    assert mod["vars"]["z"]["attrspec"] == ["protected"]
    assert mod["vars"]["y"]["attrspec"] == ["asynchronous", "volatile"]
    routine = _routine(mod, "copy")
    assert routine is not None
    assert "contiguous" in routine["vars"]["a"]["attrspec"]
    text = crackfortran.crack2fortran(blocks)
    for word in ("protected", "volatile", "asynchronous", "contiguous"):
        assert word in text


@pytest.mark.slow
class TestAttrStmtRuntime(util.F2PyTest):
    suffix = ".f90"
    code = textwrap.dedent("""\
        module attrstmt
          implicit none
          integer :: x = 3
          protected :: x
          integer :: y = 1
          volatile :: y
          asynchronous :: y
        contains
          integer function readx()
            readx = x + y
          end function readx
          real function head(a)
            real, intent(in) :: a(:)
            contiguous :: a
            head = a(1) + a(2)
          end function head
        end module attrstmt
        """)

    def test_values(self):
        mod = self.module.attrstmt
        assert mod.readx() == 4
        assert int(mod.x) == 3
        assert int(mod.y) == 1
        base = np.array([10.0, 0.0, 20.0, 0.0], dtype=np.float32)
        assert mod.head(base[::2]) == np.float32(30.0)
