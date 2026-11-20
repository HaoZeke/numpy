import pytest

from numpy import array

from . import util


@pytest.mark.slow
class TestReturnComplex(util.F2PyTest):
    def check_function(self, t, tname):
        if tname in ["t0", "t8", "s0", "s8"]:
            err = 1e-5
        else:
            err = 0.0
        assert abs(t(234j) - 234.0j) <= err
        assert abs(t(234.6) - 234.6) <= err
        assert abs(t(234) - 234.0) <= err
        assert abs(t(234.6 + 3j) - (234.6 + 3j)) <= err
        # assert abs(t('234')-234.)<=err
        # assert abs(t('234.6')-234.6)<=err
        assert abs(t(-234) + 234.0) <= err
        assert abs(t([234]) - 234.0) <= err
        assert abs(t((234, )) - 234.0) <= err
        assert abs(t(array(234)) - 234.0) <= err
        assert abs(t(array(23 + 4j, "F")) - (23 + 4j)) <= err
        assert abs(t(array([234])) - 234.0) <= err
        assert abs(t(array([[234]])) - 234.0) <= err
        assert abs(t(array([234]).astype("b")) + 22.0) <= err
        assert abs(t(array([234], "h")) - 234.0) <= err
        assert abs(t(array([234], "i")) - 234.0) <= err
        assert abs(t(array([234], "l")) - 234.0) <= err
        assert abs(t(array([234], "q")) - 234.0) <= err
        assert abs(t(array([234], "f")) - 234.0) <= err
        assert abs(t(array([234], "d")) - 234.0) <= err
        assert abs(t(array([234 + 3j], "F")) - (234 + 3j)) <= err
        assert abs(t(array([234], "D")) - 234.0) <= err

        # pytest.raises(TypeError, t, array([234], 'S1'))
        pytest.raises(TypeError, t, "abc")

        pytest.raises(IndexError, t, [])
        pytest.raises(IndexError, t, ())

        pytest.raises(TypeError, t, t)
        pytest.raises(TypeError, t, {})

        try:
            r = t(10**400)
            assert repr(r) in ["(inf+0j)", "(Infinity+0j)"]
        except OverflowError:
            pass


class TestFReturnComplex(TestReturnComplex):
    sources = [
        util.getpath("tests", "src", "return_complex", "foo77.f"),
        util.getpath("tests", "src", "return_complex", "foo90.f90"),
    ]

    @pytest.mark.parametrize("name", ["t0", "t8", "t16", "td", "s0", "s8", "s16", "sd"])
    def test_all_f77(self, name):
        self.check_function(getattr(self.module, name), name)

    @pytest.mark.parametrize("name", ["t0", "t8", "t16", "td", "s0", "s8", "s16", "sd"])
    def test_all_f90(self, name):
        self.check_function(getattr(self.module.f90_return_complex, name),
                            name)


@pytest.mark.slow
class TestF90ReturnComplexStar(util.F2PyTest):
    suffix = ".f90"
    code = """
    module f90_complex_star
      contains
        complex*8 function t8(value)
          complex*8, intent(in) :: value
          t8 = value * (2.0, 0.0)
        end function t8
        complex*16 function t16(value)
          complex*16, intent(in) :: value
          t16 = value * (2.0d0, 0.0d0)
        end function t16
    end module f90_complex_star
    """

    @pytest.mark.parametrize("name", ["t8", "t16"])
    def test_complex_star(self, name):
        fn = getattr(self.module.f90_complex_star, name)
        assert fn(1.5 + 2.5j) == pytest.approx(3.0 + 5.0j)
