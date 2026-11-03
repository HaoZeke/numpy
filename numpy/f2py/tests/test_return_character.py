import platform

import pytest

from numpy import array

from . import util

IS_S390X = platform.machine() == "s390x"


@pytest.mark.slow
class TestReturnCharacter(util.F2PyTest):
    def check_function(self, t, tname):
        if tname in ["t0", "t1", "s0", "s1"]:
            assert t("23") == b"2"
            r = t("ab")
            assert r == b"a"
            r = t(array("ab"))
            assert r == b"a"
            r = t(array(77, "u1"))
            assert r == b"M"
        elif tname in ["ts", "ss"]:
            assert t(23) == b"23"
            assert t("123456789abcdef") == b"123456789a"
        elif tname in ["t5", "s5"]:
            assert t(23) == b"23"
            assert t("ab") == b"ab"
            assert t("123456789abcdef") == b"12345"
        else:
            raise NotImplementedError


class TestFReturnCharacter(TestReturnCharacter):
    sources = [
        util.getpath("tests", "src", "return_character", "foo77.f"),
        util.getpath("tests", "src", "return_character", "foo90.f90"),
    ]

    @pytest.mark.xfail(IS_S390X, reason="callback returns ' '")
    @pytest.mark.parametrize("name", ["t0", "t1", "t5", "s0", "s1", "s5", "ss"])
    def test_all_f77(self, name):
        self.check_function(getattr(self.module, name), name)

    @pytest.mark.xfail(IS_S390X, reason="callback returns ' '")
    @pytest.mark.parametrize("name", ["t0", "t1", "t5", "ts", "s0", "s1", "s5", "ss"])
    def test_all_f90(self, name):
        self.check_function(getattr(self.module.f90_return_char, name), name)


@pytest.mark.slow
class TestFReturnCharacterNoWrap(util.F2PyTest):
    # --no-wrap-functions calls a CHARACTER*n function directly, passing
    # the result buffer and its length ahead of the other arguments
    sources = [util.getpath("tests", "src", "return_character", "foo77.f")]
    options = ["--no-wrap-functions"]
    only = ["t1", "t5"]

    def test_direct_call(self):
        assert self.module.t1(b"a") == b"a"
        assert self.module.t5(b"abcde") == b"abcde"
