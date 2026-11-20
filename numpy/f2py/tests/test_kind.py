import platform
import sys

import pytest

from numpy.f2py.crackfortran import (
    _selected_int_kind_func as selected_int_kind,
    _selected_real_kind_func as selected_real_kind,
)

from . import util

IS_PPC_OR_AIX = platform.machine().lower().startswith("ppc") or platform.system() == 'AIX'

@pytest.mark.slow
class TestKind(util.F2PyTest):
    sources = [util.getpath("tests", "src", "kind", "foo.f90")]

    @pytest.mark.skipif(sys.maxsize < 2 ** 31 + 1,
                        reason="Fails for 32 bit machines")
    def test_int(self):
        """Test `int` kind_func for integers up to 10**40."""
        selectedintkind = self.module.selectedintkind

        for i in range(40):
            assert selectedintkind(i) == selected_int_kind(
                i
            ), f"selectedintkind({i}): expected {selected_int_kind(i)!r} but got {selectedintkind(i)!r}"

    def test_real(self):
        """
        Test (processor-dependent) `real` kind_func for real numbers
        of up to 31 digits precision (extended/quadruple).
        """
        selectedrealkind = self.module.selectedrealkind

        for i in range(32):
            assert selectedrealkind(i) == selected_real_kind(
                i
            ), f"selectedrealkind({i}): expected {selected_real_kind(i)!r} but got {selectedrealkind(i)!r}"

    @pytest.mark.xfail(IS_PPC_OR_AIX,
                       reason="Some PowerPC may not support full IEEE 754 precision")
    def test_quad_precision(self):
        """
        Test kind_func for quadruple precision [`real(16)`] of 32+ digits .
        """
        selectedrealkind = self.module.selectedrealkind

        for i in range(32, 40):
            assert selectedrealkind(i) == selected_real_kind(
                i
            ), f"selectedrealkind({i}): expected {selected_real_kind(i)!r} but got {selectedrealkind(i)!r}"


class TestKindKeywordName(util.F2PyTest):
    sources = [util.getpath("tests", "src", "kind", "keyword_name.f90")]

    @pytest.mark.parametrize("name, arg, expected",
                             [("half_is", 30000, 15000),
                              ("prefix_is", 30000, 15000),
                              ("third_in", 1.0, 1 / 3)])
    def test_kind_named_like_python_keyword(self, name, arg, expected):
        assert getattr(self.module.keyword_funcs, name)(arg) == expected

    @pytest.mark.parametrize("name", ["third_expr", "third_derived"])
    def test_keyword_kind_expressions(self, name):
        assert getattr(self.module, name)(1.0) == 1 / 3

    def test_keyword_char_len(self):
        assert self.module.echo_len(b"abcde") == b"abcde"
