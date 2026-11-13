import platform
import sys

import pytest

import numpy as np
from numpy.f2py import _kind
from numpy.f2py.crackfortran import (
    _selected_int_kind_func as selected_int_kind,
    _selected_real_kind_func as selected_real_kind,
)

from . import util

IS_PPC_OR_AIX = platform.machine().lower().startswith("ppc") or platform.system() == 'AIX'

_DTYPE = {
    'float': np.float32,
    'double': np.float64,
    'long_double': np.longdouble,
    'signed_char': np.int8,
    'short': np.int16,
    'int': np.int32,
    'long_long': np.int64,
}
# Widths f2py can wrap. Anything else (flang real kind 2, 128-bit
# integers) is a legal compiler kind that stays unmapped.
_MAPPED_BITS = {
    'integer': {8, 16, 32, 64},
    'real': {32, 64, 128},
}


def _check_kind(typespec, requested, fortran_kind, model_kind):
    assert fortran_kind == model_kind, (
        f"{typespec} selected_*_kind({requested}): "
        f"compiler returned {fortran_kind!r}, f2py model returned {model_kind!r}"
    )
    if fortran_kind < 0:
        return
    traits = _kind.kind_traits(typespec, fortran_kind)
    ctype = _kind.ctype_from_kind(typespec, fortran_kind)
    if traits is None:
        return
    if ctype is None:
        assert traits['storage'] not in _MAPPED_BITS[typespec]
        return
    dtype = np.dtype(_DTYPE[ctype])
    assert dtype.itemsize * 8 == traits['storage']
    # Quad (decimal precision 33) is stored in 16 bytes and mapped to
    # C long double, which may be 80-bit. Require the dtype to cover
    # the request only when the Fortran kind itself fits in that dtype.
    if typespec == 'real' and traits['precision'] <= np.finfo(dtype).precision:
        assert np.finfo(dtype).precision >= requested


def test_kind_fallback_is_gfortran_numbering(monkeypatch):
    monkeypatch.delenv('FC', raising=False)
    _kind.clear_kind_cache()
    assert _kind.selected_int_kind_fallback(19) == 16
    assert _kind.selected_int_kind_fallback(40) == -1
    assert _kind.selected_real_kind_fallback(0) == 4
    assert _kind.selected_real_kind_fallback(15) == 8
    assert _kind.selected_real_kind_fallback(16) == 10
    assert selected_real_kind(0) == 4
    assert selected_int_kind(18) == 8


def test_half_precision_is_not_mapped_to_float():
    table = {
        'int': {},
        'real': {0: 2, 6: 4, 16: 10},
        'traits': {
            'integer': {},
            'real': {
                2: {'storage': 16, 'precision': 3, 'range': 4},
                4: {'storage': 32, 'precision': 6, 'range': 37},
                10: {'storage': 128, 'precision': 18, 'range': 4931},
            },
        },
    }
    _kind.override_kind_table(table)
    try:
        assert _kind.ctype_from_kind('real', 2) is None
        assert _kind.ctype_from_kind('real', '10') == 'long_double'
        assert _kind.ctype_from_kind('real', 4) == 'float'
    finally:
        _kind.override_kind_table(None)


@pytest.mark.slow
class TestKind(util.F2PyTest):
    sources = [util.getpath("tests", "src", "kind", "foo.f90")]

    @pytest.mark.skipif(sys.maxsize < 2 ** 31 + 1,
                        reason="Fails for 32 bit machines")
    def test_int(self):
        """Kind numbers follow the active compiler, and the mapped dtype matches storage."""
        selectedintkind = self.module.selectedintkind

        for i in range(40):
            _check_kind('integer', i, selectedintkind(i), selected_int_kind(i))

    def test_real(self):
        """
        Decimal precision, not gfortran's kind integer.

        flang's ``selected_real_kind(0) == 2`` is IEEE half. ifx has no
        kind 10; ``selected_real_kind(16)`` is kind 16 (binary128).
        """
        selectedrealkind = self.module.selectedrealkind

        for i in range(32):
            _check_kind('real', i, selectedrealkind(i), selected_real_kind(i))

    @pytest.mark.xfail(IS_PPC_OR_AIX,
                       reason="Some PowerPC may not support full IEEE 754 precision")
    def test_quad_precision(self):
        """Kind for 32 or more decimal digits, when the compiler has one."""
        selectedrealkind = self.module.selectedrealkind

        for i in range(32, 40):
            _check_kind('real', i, selectedrealkind(i), selected_real_kind(i))
