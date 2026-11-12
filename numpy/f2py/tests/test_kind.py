import os
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
    @pytest.mark.xfail(
        "ifx" in os.environ.get("FC", ""),
        reason=(
            "ifx has no integer kind for 19 decimal digits, so "
            "selected_int_kind(19) is -1. f2py's static kind table "
            "uses gfortran's 16. Using ifx's answer in that table "
            "would break gfortran."
        ),
    )
    def test_int(self):
        """Test `int` kind_func for integers up to 10**40."""
        selectedintkind = self.module.selectedintkind

        for i in range(40):
            assert selectedintkind(i) == selected_int_kind(
                i
            ), f"selectedintkind({i}): expected {selected_int_kind(i)!r} but got {selectedintkind(i)!r}"

    @pytest.mark.xfail(
        "ifx" in os.environ.get("FC", ""),
        reason=(
            "ifx's selected_real_kind(16) is 16, IEEE quad. gfortran's "
            "is 10, 80-bit extended precision. f2py's static kind table "
            "follows gfortran."
        ),
    )
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
