import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestCharArrayIoBlanks(util.F2PyTest):
    suffix = ".f90"
    code = """
      subroutine swap_one(x, flag)
        character(len=3), intent(inout) :: x(2)
        integer, intent(out) :: flag
        flag = 0
        if (x(1) == 'a') flag = flag + 1
        if (x(1) == 'a  ') flag = flag + 2
        if (x(2) == 'b ') flag = flag + 4
        x(1) = 'q'
      end subroutine swap_one

      subroutine mark(c, flag)
        character(len=3) :: c(2, 3)
        !f2py intent(in,out) c
        integer, intent(out) :: flag
        flag = 0
        if (c(1, 2) == 'cd') flag = flag + 1
        if (c(2, 1) == 'ef') flag = flag + 2
        c(2, 3) = 'Z'
      end subroutine mark

      subroutine peek(x, flag)
        character(len=3), intent(in) :: x(2)
        integer, intent(out) :: flag
        flag = 0
        if (x(1) == 'a') flag = flag + 1
        if (x(1) == 'a  ') flag = flag + 2
        if (x(2) == 'b ') flag = flag + 4
      end subroutine peek

      subroutine fill_out(x, flag)
        character(len=3), intent(out) :: x(2)
        integer, intent(out) :: flag
        flag = 0
        x(1) = 'q'
        x(2) = 'zz'
      end subroutine fill_out
    """

    def test_inout_exact(self):
        x = np.array([b"a", b"b "], dtype="S3")
        flag = self.module.swap_one(x)
        assert flag == 7
        assert x.tobytes() == b"q\x00\x00b \x00"

    def test_in_out_exact(self):
        c = np.empty((2, 3), dtype="S3", order="F")
        c[...] = b"..."
        c[0, 1] = b"cd"
        c[1, 0] = b"ef"
        assert c.flags.f_contiguous
        out, flag = self.module.mark(c)
        assert flag == 3
        assert out.shape == (2, 3)
        assert out.tobytes() == b"...cd\x00...ef\x00...Z\x00\x00"
        assert c.tobytes() == b"...cd\x00...ef\x00...Z\x00\x00"

    def test_intent_in_unchanged(self):
        x = np.array([b"a", b"b "], dtype="S3")
        before = x.tobytes()
        flag = self.module.peek(x)
        assert flag == 7
        assert x.tobytes() == before

    def test_intent_out_exact(self):
        x, flag = self.module.fill_out()
        assert flag == 0
        assert x.shape == (2,)
        assert x.tobytes() == b"q\x00\x00zz\x00"


@pytest.mark.slow
class TestCharArrayIoBlanksPyf(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "char_array_io", "blank.pyf"),
        util.getpath("tests", "src", "char_array_io", "blank.f90"),
    ]
    module_name = "_char_io_blank_pyf"

    def test_pyf_signature(self):
        x = np.array([b"a", b"b "], dtype="S3")
        before = x.tobytes()
        flag = self.module.see_in(x)
        assert flag == 7
        assert x.tobytes() == before

        y = np.array([b"a", b"b "], dtype="S3")
        flag = self.module.see_inout(y)
        assert flag == 7
        assert y.tobytes() == b"q\x00\x00b \x00"

        z, flag = self.module.see_out()
        assert flag == 0
        assert z.tobytes() == b"q\x00\x00zz\x00"
