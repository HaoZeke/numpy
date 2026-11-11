"""Fortran OPTIONAL arguments are omitted when ``absent`` is set."""

import re
import textwrap

import pytest

import numpy as np
from numpy.f2py import crackfortran

from . import util

# Values the Fortran routines return. Named so the checks stay readable.
PRESENT_SCALAR = 5.0
ABSENT_ARRAY = -1.0
ARRAY_TERMS = (1.0, 2.0)
ARRAY_SUM = 3.0
EXPLICIT = (1.0, 2.0, 3.0)
FLAG_B = 2
FLAG_BOTH = 3
INOUT_STEP_BASE = 3
INOUT_STEP = 10
OUT_VALUE = 4.0
ABSENT_STR = -5
SAMPLE_TEXT = "ab"
INIT_VALUE = 4.0
CALLBACK_INPUT = 2.0
CALLBACK_WHEN_PRESENT = 3.0


def _attr(var):
    return set(var.get("attrspec") or [])


def _routine(blocks, name):
    if isinstance(blocks, dict):
        blocks = [blocks]
    for block in blocks:
        if block.get("name") == name and block.get("block") in (
                "function", "subroutine"):
            return block
        found = _routine(block.get("body") or [], name)
        if found is not None:
            return found
    return None


class TestAbsentCrack:
    def test_fortran_optional_sets_absent(self, tmp_path):
        src = tmp_path / "opt.f90"
        src.write_text(textwrap.dedent("""\
            subroutine demo(x, y, z)
              real, intent(in), optional :: x
              real, intent(in) :: y
              real :: z
              optional :: z
            end subroutine
            """))
        mod = crackfortran.crackfortran([str(src)])
        routine = _routine(mod, "demo")
        assert routine is not None
        assert "absent" in _attr(routine["vars"]["x"])
        assert "optional" in _attr(routine["vars"]["x"])
        assert "absent" not in _attr(routine["vars"]["y"])
        assert "absent" in _attr(routine["vars"]["z"])

    def test_pyf_optional_does_not_set_absent(self, tmp_path):
        src = tmp_path / "opt.pyf"
        src.write_text(textwrap.dedent("""\
            python module m
            interface
              subroutine demo(x, y, n)
                real, optional, intent(in) :: x
                real, optional, absent, intent(in) :: y
                real, intent(in) :: n = 1
              end subroutine
            end interface
            end python module m
            """))
        mod = crackfortran.crackfortran([str(src)])
        routine = _routine(mod, "demo")
        assert "optional" in _attr(routine["vars"]["x"])
        assert "absent" not in _attr(routine["vars"]["x"])
        assert "absent" in _attr(routine["vars"]["y"])
        # An init expression makes the argument optional and does not
        # mark it absent.
        assert "optional" in _attr(routine["vars"]["n"])
        assert "absent" not in _attr(routine["vars"]["n"])

    def test_f2py_directive_does_not_set_absent(self, tmp_path):
        src = tmp_path / "opt.f90"
        src.write_text(textwrap.dedent("""\
            subroutine demo(x)
              real, intent(in) :: x
              !f2py real, optional, intent(in) :: x
            end subroutine
            """))
        mod = crackfortran.crackfortran([str(src)])
        routine = _routine(mod, "demo")
        assert "optional" in _attr(routine["vars"]["x"])
        assert "absent" not in _attr(routine["vars"]["x"])

    def test_check_does_not_set_absent(self, tmp_path):
        src = tmp_path / "opt.pyf"
        src.write_text(textwrap.dedent("""\
            python module m
            interface
              subroutine demo(x)
                real, intent(in), check(x > 0) :: x = 1
              end subroutine
            end interface
            end python module m
            """))
        mod = crackfortran.crackfortran([str(src)])
        routine = _routine(mod, "demo")
        assert "absent" not in _attr(routine["vars"]["x"])


@pytest.mark.slow
class TestOptionalAbsent(util.F2PyTest):
    sources = [util.getpath("tests", "src", "regression", "gh4013.f90")]

    def test_scalar(self):
        mod = self.module.gh4013
        assert mod.s_scalar() == pytest.approx(1.0)
        assert mod.s_scalar(PRESENT_SCALAR) == pytest.approx(PRESENT_SCALAR)
        assert mod.s_scalar(None) == pytest.approx(1.0)

    def test_assumed_shape(self):
        mod = self.module.gh4013
        assert mod.s_array() == pytest.approx(ABSENT_ARRAY)
        got = mod.s_array(np.array(ARRAY_TERMS, dtype=np.float32))
        assert got == pytest.approx(ARRAY_SUM)
        assert mod.s_array(None) == pytest.approx(ABSENT_ARRAY)

    def test_mixed_presence(self):
        mod = self.module.gh4013
        assert mod.n_present() == 0
        assert mod.n_present(1) == 1
        assert mod.n_present(b=np.array(EXPLICIT, dtype=np.float32)) == FLAG_B
        both = mod.n_present(1, np.array(EXPLICIT, dtype=np.float32))
        assert both == FLAG_BOTH

    def test_inout_scalar(self):
        mod = self.module.gh4013
        assert mod.s_inout() == 0
        value = np.array(INOUT_STEP_BASE, dtype=np.int32)
        flag = mod.s_inout(value)
        assert flag == 1
        assert int(value) == INOUT_STEP_BASE + INOUT_STEP

    def test_intent_out_stays_present(self):
        # A pure intent(out) argument is the return value. f2py allocates
        # it, so present() stays true even though the dummy is optional.
        mod = self.module.gh4013
        value, flag = mod.s_out()
        assert flag == 1
        assert value == pytest.approx(OUT_VALUE)

    def test_string(self):
        mod = self.module.gh4013
        assert mod.s_str() == ABSENT_STR
        assert mod.s_str(None) == ABSENT_STR
        assert mod.s_str(SAMPLE_TEXT) == len(SAMPLE_TEXT)


@pytest.mark.slow
class TestPyfAbsent(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "regression", "gh4013_pyf.pyf"),
        util.getpath("tests", "src", "regression", "gh4013_pyf.f90"),
    ]
    module_name = "gh4013pyf"

    def test_hand_written_absent(self):
        assert self.module.pyf_absent_s() == pytest.approx(1.0)
        got = self.module.pyf_absent_s(PRESENT_SCALAR)
        assert got == pytest.approx(PRESENT_SCALAR)

    def test_optional_only_keeps_default(self):
        # No absent attribute: the omitted value is still passed, and the
        # Fortran dummy is present with the zero default.
        assert self.module.pyf_optional_s() == pytest.approx(0.0)
        got = self.module.pyf_optional_s(PRESENT_SCALAR)
        assert got == pytest.approx(PRESENT_SCALAR)


def test_init_drops_absent(capsys):
    src = util.getpath("tests", "src", "regression", "gh4013_init.f90")
    mod = crackfortran.crackfortran([str(src)])
    routine = _routine(mod, "with_init")
    assert "absent" not in _attr(routine["vars"]["x"])
    assert "=" in routine["vars"]["x"]
    message = capsys.readouterr().out
    assert 'routine "with_init" argument "x"' in message
    assert "init expression" in message


def test_source_callback_drops_absent(capsys):
    src = util.getpath("tests", "src", "regression", "gh4013_cb.f")
    mod = crackfortran.crackfortran([str(src)])
    routine = _routine(mod, "applyopt")
    assert "absent" not in _attr(routine["vars"]["fun"])
    message = capsys.readouterr().out
    assert 'routine "applyopt" argument "fun"' in message
    assert "callback" in message


def test_source_unsupported_kinds_drop_absent(capsys):
    src = util.getpath("tests", "src", "regression", "gh4013_keep.f90")
    mod = crackfortran.crackfortran([str(src)])
    expected = (
        ("use_cb", "fun", "callback"),
        ("use_charr", "c", "character array"),
        ("use_alloc", "x", "allocatable"),
        ("use_ptr", "x", "pointer"),
        ("use_value", "x", "value"),
        ("use_cintent", "x", "intent(c)"),
    )
    message = capsys.readouterr().out
    for name, arg, reason in expected:
        routine = _routine(mod, name)
        assert "absent" not in _attr(routine["vars"][arg])
        assert f'routine "{name}" argument "{arg}"' in message
        assert reason in message


def test_f77_optional_drops_absent(capsys):
    src = util.getpath("tests", "src", "regression", "gh4013_f77.f")
    mod = crackfortran.crackfortran([str(src)])
    routine = _routine(mod, "f77opt")
    assert "absent" not in _attr(routine["vars"]["x"])
    message = capsys.readouterr().out
    assert 'routine "f77opt" argument "x"' in message
    assert "F77 external" in message


def test_pyf_init_drops_absent(capsys, tmp_path):
    src = tmp_path / "init.pyf"
    src.write_text(textwrap.dedent("""\
        python module m
        interface
          subroutine with_init(x)
            real, optional, absent, intent(in) :: x = 4.0
          end subroutine with_init
        end interface
        end python module m
        """))
    mod = crackfortran.crackfortran([str(src)])
    routine = _routine(mod, "with_init")
    assert "absent" not in _attr(routine["vars"]["x"])
    assert "=" in routine["vars"]["x"]
    message = capsys.readouterr().out
    assert 'routine "with_init" argument "x"' in message
    assert "init expression" in message


@pytest.mark.slow
class TestKeepPreviousCall(util.F2PyTest):
    """Unsupported OPTIONAL kinds from Fortran source keep the old call."""

    sources = [util.getpath("tests", "src", "regression", "gh4013_keep.f90")]

    def test_callback_matches_previous_call(self):
        # The base build runs the callback and returns its value.
        assert self.module.use_cb(lambda: 4) == 4

    def test_character_array_matches_previous_call(self):
        # The base build passes a character actual, so present() is true.
        assert self.module.use_charr() == 1

    def test_allocatable_matches_previous_call(self):
        # The base build leaves the allocatable unallocated.
        assert self.module.use_alloc() == 0

    def test_pointer_matches_previous_call(self):
        # The base build leaves the pointer unassociated.
        assert self.module.use_ptr() == 0

    def test_value_matches_previous_call(self):
        # The base build reports the value dummy as not present.
        assert self.module.use_value() == 0

    def test_intent_c_matches_previous_call(self):
        # The base build reports the intent(c) dummy as not present.
        assert self.module.use_cintent() == 0


@pytest.mark.slow
class TestExplicitInitKeepsDefault(util.F2PyTest):
    sources = [util.getpath("tests", "src", "regression", "gh4013_init.f90")]

    def test_omitted_uses_init(self):
        # The directive default is INIT_VALUE. A negative result would
        # mean the actual was absent.
        assert self.module.gh4013init.with_init() == pytest.approx(INIT_VALUE)

    def test_passed_value(self):
        got = self.module.gh4013init.with_init(PRESENT_SCALAR)
        assert got == pytest.approx(PRESENT_SCALAR)


@pytest.mark.slow
class TestOptionalCallbackBuilds(util.F2PyTest):
    sources = [util.getpath("tests", "src", "regression", "gh4013_cb.f")]

    def test_passed_callback_runs(self):
        assert self.module.applyopt(lambda: 4) == 4


@pytest.mark.slow
class TestF77OptionalBuilds(util.F2PyTest):
    sources = [util.getpath("tests", "src", "regression", "gh4013_f77.f")]

    def test_builds_and_runs(self):
        self.module.f77opt(CALLBACK_INPUT)


def test_f77_hand_written_absent_is_rejected():
    src = util.getpath("tests", "src", "regression", "gh4013_f77_hand.f")
    with pytest.raises(Exception, match=r'argument "x".*F77 external'):
        util.build_module([str(src)])


_PYF_UNSUPPORTED = (
    ("cb_hand", "real, external, optional, absent :: x", "callback"),
    (
        "charr_hand",
        "character(len=1), dimension(2), optional, absent, intent(in) :: x",
        "character array",
    ),
    (
        "alloc_hand",
        "real, allocatable, optional, absent, intent(in) :: x(:)",
        "allocatable",
    ),
    (
        "ptr_hand",
        "real, pointer, optional, absent, intent(in) :: x(:)",
        "pointer",
    ),
    (
        "value_hand",
        "integer, value, optional, absent, intent(in) :: x",
        "value",
    ),
    (
        "cintent_hand",
        "real, optional, absent, intent(c) :: x",
        "intent(c)",
    ),
)


@pytest.mark.parametrize("name,decl,reason", _PYF_UNSUPPORTED)
def test_pyf_hand_written_absent_is_rejected(name, decl, reason, tmp_path):
    src = tmp_path / f"{name}.pyf"
    src.write_text(textwrap.dedent(f"""\
        python module m
        interface
          subroutine {name}(x)
            {decl}
          end subroutine {name}
        end interface
        end python module m
        """))
    with pytest.raises(Exception, match=rf'argument "x".*{re.escape(reason)}'):
        util.build_module([str(src)], module_name=name)
