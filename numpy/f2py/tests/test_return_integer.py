import numpy.f2py.crackfortran as crackfortran
import numpy.f2py.func2subr as func2subr
import pytest

from numpy import array

from . import util


@pytest.mark.slow
class TestReturnInteger(util.F2PyTest):
    def check_function(self, t, tname):
        assert t(123) == 123
        assert t(123.6) == 123
        assert t("123") == 123
        assert t(-123) == -123
        assert t([123]) == 123
        assert t((123, )) == 123
        assert t(array(123)) == 123
        assert t(array(123, "b")) == 123
        assert t(array(123, "h")) == 123
        assert t(array(123, "i")) == 123
        assert t(array(123, "l")) == 123
        assert t(array(123, "B")) == 123
        assert t(array(123, "f")) == 123
        assert t(array(123, "d")) == 123

        # pytest.raises(ValueError, t, array([123],'S3'))
        pytest.raises(ValueError, t, "abc")

        pytest.raises(IndexError, t, [])
        pytest.raises(IndexError, t, ())

        pytest.raises(TypeError, t, t)
        pytest.raises(TypeError, t, {})

        if tname in ["t8", "s8"]:
            pytest.raises(OverflowError, t, 100000000000000000000000)
            pytest.raises(OverflowError, t, 10000000011111111111111.23)


class TestFReturnInteger(TestReturnInteger):
    sources = [
        util.getpath("tests", "src", "return_integer", "foo77.f"),
        util.getpath("tests", "src", "return_integer", "foo90.f90"),
    ]

    @pytest.mark.parametrize("name",
                             ["t0", "t1", "t2", "t4", "t8", "s0", "s1", "s2", "s4", "s8"])
    def test_all_f77(self, name):
        self.check_function(getattr(self.module, name), name)

    @pytest.mark.parametrize("name",
                             ["t0", "t1", "t2", "t4", "t8", "s0", "s1", "s2", "s4", "s8"])
    def test_all_f90(self, name):
        self.check_function(getattr(self.module.f90_return_integer, name),
                            name)


def test_integer_array_result_has_interface(tmp_path):
    src = tmp_path / "arr.f90"
    src.write_text(
        "function f(n) result(r)\n"
        "  integer, intent(in) :: n\n"
        "  integer, dimension(3) :: r\n"
        "  r = n\n"
        "end function\n"
    )
    rout = crackfortran.crackfortran([str(src)])[0]
    _rout, wrapper = func2subr.assubr(rout)
    assert "interface" in wrapper
    assert "integer (3)" not in wrapper


@pytest.mark.slow
class TestIntegerArrayResult(util.F2PyTest):
    sources = [util.getpath("tests", "src", "return_integer", "arr.f90")]

    def test_values(self):
        assert list(self.module.f(7)) == [7, 7, 7]
