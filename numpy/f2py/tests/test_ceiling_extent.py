import textwrap

from numpy.f2py import capi_maps, cfuncs, crackfortran
from numpy.f2py.auxfuncs import getargs2

from . import util


class TestCeilingExtent(util.F2PyTest):
    suffix = ".f90"
    code = textwrap.dedent("""
        subroutine cout(a, b, c, x, s, sb, sc)
          real, intent(in) :: x
          integer, intent(out) :: a(ceiling(x))
          integer, intent(out) :: b(ceiling(x) + 5)
          integer, intent(out) :: c(ceiling(x, kind=4))
          integer, intent(out) :: s
          integer, intent(out) :: sb
          integer, intent(out) :: sc
          s = size(a)
          sb = size(b)
          sc = size(c)
          if (s > 0) a(1) = 7
          if (sb > 0) b(1) = 8
          if (sc > 0) c(1) = 9
        end subroutine cout
        """)

    def test_ceiling_extent(self):
        a, b, c, s, sb, sc = self.module.cout(2.5)
        assert s == 3
        assert a.shape == (3,)
        assert a[0] == 7
        assert sb == 8
        assert b.shape == (8,)
        assert b[0] == 8
        assert sc == 3
        assert c.shape == (3,)
        assert c[0] == 9
        a, b, c, s, sb, sc = self.module.cout(2.0)
        assert s == 2
        assert a.shape == (2,)
        assert a[0] == 7
        assert sb == 7
        assert sc == 2
        a, b, c, s, sb, sc = self.module.cout(-0.5)
        assert s == 0
        assert a.shape == (0,)
        assert sb == 5
        assert b.shape == (5,)
        assert b[0] == 8
        assert sc == 0
        assert c.shape == (0,)

    def test_ceiling_negative_extent(self):
        a, b, c, s, sb, sc = self.module.cout(-1.2)
        assert s == 0
        assert a.shape == (0,)
        assert sb == 4
        assert b.shape == (4,)
        assert b[0] == 8
        assert sc == 0
        assert c.shape == (0,)
        a, b, c, s, sb, sc = self.module.cout(-10.0)
        assert s == 0
        assert sb == 0
        assert b.shape == (0,)
        assert sc == 0


def test_ceiling_extent_c_form(tmp_path):
    src = tmp_path / "ceiling.f90"
    src.write_text(textwrap.dedent("""
        subroutine cout(a, b, c, x)
          real, intent(in) :: x
          integer, intent(out) :: a(ceiling(x))
          integer, intent(out) :: b(ceiling(x) + 5)
          integer, intent(out) :: c(ceiling(x, kind=4))
        end subroutine cout
        """))
    routine = crackfortran.crackfortran([str(src)])[0]
    dims = routine["vars"]
    assert dims["a"]["dimension"] == [
        "(((npy_intp)ceil(x))<0?0:((npy_intp)ceil(x)))"
    ]
    assert dims["b"]["dimension"] == [
        "((5 + (npy_intp)ceil(x))<0?0:(5 + (npy_intp)ceil(x)))"
    ]
    assert dims["c"]["dimension"] == dims["a"]["dimension"]
    includes = cfuncs.outneeds["includes0"]
    includes[:] = [item for item in includes if item != "math.h"]
    _, depargs = getargs2(routine)
    capi_maps.depargs = depargs
    got = capi_maps.getarrdims("a", dims["a"])
    assert "ceil(" in got["setdims"]
    assert "math.h" in includes
    includes[:] = [item for item in includes if item != "math.h"]
