"""check() must raise the module error for every scalar it can format.

gh-31584 passed printf fragments from cformat_map to PyErr_Format.
PyUnicode_FromFormat rejects ``%g``, ``%Lg`` and the ``h`` length
modifier, and it treats a ``%`` in the check text as a conversion.
"""

import pytest

import numpy as np

from . import util

_I8 = np.iinfo(np.int8)
_I16 = np.iinfo(np.int16)
_I32 = np.iinfo(np.int32)
_ILONG = np.iinfo(np.int_)
_ILL = np.iinfo(np.longlong)
_F32 = np.finfo(np.float32)
_F64 = np.finfo(np.float64)
_FLD = np.finfo(np.longdouble)

# Meson compiles this only when the real*16 class is set up, not at collection.
_REAL16_PROBE = "subroutine r16(x)\n  real*16 x\nend subroutine r16\n"


def _g(value):
    """C ``%g`` text. Python's ``.6g`` matches it for these IEEE values."""
    return format(float(value), ".6g")


def _long_double_text(value):
    """Match ``%Lg`` for the exponent form this test prints."""
    # %Lg prints 6 significant digits, so precision=5 is the digits after the point.
    return np.format_float_scientific(
        np.longdouble(value), precision=5, unique=False, trim="-"
    )


def _message(routine, check_text, shown):
    if shown is None:
        return f"({check_text}) failed for 1st argument x"
    return f"({check_text}) failed for 1st argument x: {routine}:{shown}"


@pytest.mark.slow
class TestCheckFormat(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "regression", "gh31584_check.f90"),
    ]

    def _err(self):
        return getattr(self.module, f"_{self.module.__name__}_error")

    def _failed(self, routine, value):
        with pytest.raises(self._err()) as caught:
            getattr(self.module, routine)(value)
        return str(caught.value)

    @pytest.mark.parametrize(
        ("routine", "value", "check_text", "shown"),
        [
            ("chk_signed_char", int(_I8.min), "x>0", f"x={int(_I8.min)}"),
            ("chk_short", int(_I16.min), "x>0", f"x={int(_I16.min)}"),
            ("chk_integer2", int(_I16.min), "x>0", f"x={int(_I16.min)}"),
            ("chk_int", int(_I32.min), "x>0", f"x={int(_I32.min)}"),
            ("chk_long", int(_ILONG.min), "x>0", f"x={int(_ILONG.min)}"),
            ("chk_long_long", int(_ILL.max), "x<0", f"x={int(_ILL.max)}"),
            (
                "chk_long_long",
                int(_I32.max) + 1,
                "x<0",
                f"x={int(_I32.max) + 1}",
            ),
            ("chk_byte", int(_I8.min), "x>0", f"x={int(_I8.min)}"),
            ("chk_float", np.float32(_F32.tiny), "x<0", f"x={_g(_F32.tiny)}"),
            ("chk_float", np.float32(_F32.max), "x<0", f"x={_g(_F32.max)}"),
            ("chk_double", np.float64(_F64.tiny), "x<0", f"x={_g(_F64.tiny)}"),
            ("chk_double", np.float64(_F64.max), "x<0", f"x={_g(_F64.max)}"),
            ("chk_double0", 0.0, "x>0", f"x={_g(0.0)}"),
            (
                "chk_long_double",
                np.longdouble(_FLD.tiny),
                "x<0",
                f"x={_long_double_text(_FLD.tiny)}",
            ),
            ("chk_logical", True, "x==0", "x=1"),
            ("chk_logical1", True, "x==0", "x=1"),
            ("chk_logical2", True, "x==0", "x=1"),
            ("chk_logical8", True, "x==0", "x=1"),
            ("chk_mod", int(_I32.max), "x%2==0", f"x={int(_I32.max)}"),
            ("chk_complex_float", np.complex64(_F32.tiny), "x.r<0", None),
            ("chk_complex_double", np.complex128(_F64.tiny), "x.r<0", None),
            (
                "chk_complex_long_double",
                np.clongdouble(_FLD.tiny),
                "x.r<0",
                None,
            ),
        ],
    )
    def test_failed_check_is_module_error(
        self, routine, value, check_text, shown
    ):
        got = self._failed(routine, value)
        assert got == _message(routine, check_text, shown)

    @pytest.mark.parametrize(
        ("routine", "value", "expected"),
        [
            (
                "chk_character",
                b"Z",
                "(x=='a') failed for 1st argument x: chk_character:x='Z'",
            ),
            (
                "chk_character",
                b"%",
                "(x=='a') failed for 1st argument x: chk_character:x='%'",
            ),
            (
                "chk_string",
                b"zzzz",
                (
                    "(x[0]=='a') failed for 1st argument x: "
                    'chk_string:slen(x)=4 x="zzzz"'
                ),
            ),
            (
                "chk_str",
                b"%zzzz",
                (
                    "(x[0]=='a') failed for 1st argument x: "
                    'chk_str:slen(x)=5 x="%zzzz"'
                ),
            ),
            (
                "chk_str_pct",
                b"zzzz",
                (
                    "(x[0]=='%') failed for 1st argument x: "
                    'chk_str_pct:slen(x)=4 x="zzzz"'
                ),
            ),
            (
                "chk_str8",
                b"bcdefghi",
                (
                    "(*x=='a') failed for 1st argument x: "
                    'chk_str8:slen(x)=8 x="bcdefghi"'
                ),
            ),
            (
                "chk_str8",
                b"%bcdefgh",
                (
                    "(*x=='a') failed for 1st argument x: "
                    'chk_str8:slen(x)=8 x="%bcdefgh"'
                ),
            ),
            (
                "chk_strlen",
                b"bcdefghi",
                (
                    "(slen(x)%2==1) failed for 1st argument x: "
                    'chk_strlen:slen(x)=8 x="bcdefghi"'
                ),
            ),
        ],
    )
    def test_failed_string_check(self, routine, value, expected):
        assert self._failed(routine, value) == expected

    def test_check_string_non_utf8(self):
        # A direct PyErr_Format %s replaces the illegal byte. v2.5.4's
        # PyErr_SetString rejects the whole message instead.
        got = self._failed("chk_str8", b"b\xffcdefgh")
        assert got == (
            '(*x==\'a\') failed for 1st argument x: '
            'chk_str8:slen(x)=8 x="b\ufffdcdefgh"'
        )

    def test_second_argument_is_checked_when_the_first_passes(self):
        with pytest.raises(self._err()) as caught:
            self.module.chk_two(int(_I32.max), int(_I32.min))
        assert str(caught.value) == (
            f"(b>0) failed for 2nd argument b: chk_two:b={int(_I32.min)}"
        )

    def test_first_failure_skips_the_later_check(self):
        with pytest.raises(self._err()) as caught:
            self.module.chk_two(int(_I32.min), int(_I32.min))
        assert str(caught.value) == (
            f"(a>0) failed for 1st argument a: chk_two:a={int(_I32.min)}"
        )

    def test_passing_check(self):
        assert self.module.chk_int(int(_I32.max)) is None
        assert self.module.chk_double(-np.float64(_F64.tiny)) is None
        assert self.module.chk_character(b"a") is None
        assert self.module.chk_two(int(_I32.max), int(_I32.max)) is None


@pytest.mark.slow
class TestCheckReal16(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "regression", "gh31584_real16.f90"),
    ]

    def setup_method(self):
        if self.module is None and not util.check_language(
            "fortran", _REAL16_PROBE
        ):
            pytest.skip("Fortran compiler has no real*16")
        super().setup_method()

    def test_real16(self):
        value = np.longdouble(_FLD.tiny)
        err = getattr(self.module, f"_{self.module.__name__}_error")
        with pytest.raises(err) as caught:
            self.module.chk_real16(value)
        shown = _long_double_text(value)
        assert str(caught.value) == (
            f"(x<0) failed for 1st argument x: chk_real16:x={shown}"
        )
