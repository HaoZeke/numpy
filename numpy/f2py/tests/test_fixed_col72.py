import pytest

import numpy as np
from numpy.f2py import crackfortran

from . import util


def _declaration(start, tabbed):
    head = "integer, intent(inout) :: a"
    pad = start - 1 - (6 + len(head))
    gap = " " * pad
    if tabbed:
        return "\t" + head + gap + "(3)"
    return "      " + head + gap + "(3)"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("suffix", "start", "tabbed", "unlimited"),
    [
        (".f", 73, False, False),
        (".f", 80, False, False),
        (".f90", 73, False, False),
        (".f", 70, True, False),
        (".f", 73, True, False),
        (".f", 80, False, True),
    ],
)
def test_fixed_form_column_limit(tmp_path, suffix, start, tabbed, unlimited):
    # (3) is part of the statement only inside the fixed line length.
    line = _declaration(start, tabbed)
    kept = unlimited or start + 2 <= 72
    text = "      subroutine s(a)\n" + line + "\n      a = 9\n      end\n"
    if suffix == ".f":
        text = "! -*- fix -*-\n" + text
    src = tmp_path / f"s{suffix}"
    src.write_text(text)
    flags = []
    if suffix == ".f90":
        flags.append("-ffixed-form")
    if unlimited:
        flags.append("-ffixed-line-length-none")
    options = [f"--f90flags={' '.join(flags)}"] if flags else []
    tag = f"{suffix[1:]}{start}{'t' if tabbed else 's'}{'u' if unlimited else 'd'}"
    mod = util.build_module(
        [str(src)],
        options=options,
        module_name=f"lim{tag}",
    )
    if kept:
        a = np.zeros(3, dtype=np.int32)
        mod.s(a)
        assert np.all(a == 9)
    else:
        a = np.array(0, dtype=np.int32)
        mod.s(a)
        assert int(a) == 9


def _dimension(path):
    crackfortran.reset_global_f2py_vars()
    blocks = crackfortran.crackfortran([str(path)])
    for block in blocks:
        if block.get("name") == "s":
            return block["vars"]["a"].get("dimension")
    raise AssertionError("subroutine s was not read")


@pytest.mark.parametrize(
    ("suffix", "flags", "expect"),
    [
        (".f", "--f90flags=-ffixed-line-length-none", ["3"]),
        (".f", "--f90flags=-ffixed-line-length-80", ["3"]),
        (".f90", "--f90flags=-ffixed-form", None),
    ],
)
def test_second_parse_keeps_fixed_flags(tmp_path, suffix, flags, expect):
    # (3) starts at column 73. A second crackfortran() call is the
    # build parse after a signature parse, and it must see these flags.
    line = _declaration(73, False)
    text = "      subroutine s(a)\n" + line + "\n      a = 9\n      end\n"
    if suffix == ".f":
        text = "! -*- fix -*-\n" + text
    src = tmp_path / f"s{suffix}"
    src.write_text(text)
    crackfortran._pending_fflags = [flags]
    try:
        assert _dimension(src) == expect
        assert _dimension(src) == expect
    finally:
        crackfortran._pending_fflags = None
        crackfortran.reset_global_f2py_vars()
