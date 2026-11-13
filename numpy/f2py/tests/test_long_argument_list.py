import numpy as np
import pytest

from . import util


def _long_function():
    names = [f"a{i:02d}" for i in range(36)]
    header = "function s(" + ", ".join(names) + ") result(y)"
    assert len(header) > 132
    decls = [
        "  integer, intent(in) :: " + ", ".join(names[i:i + 8])
        for i in range(0, len(names), 8)
    ]
    body = "\n".join(
        [header, *decls, "  integer :: y", "  y = a00 + a35", "end function", ""]
    )
    return body


@pytest.mark.slow
class TestLongArgumentList(util.F2PyTest):
    # One statement longer than the default free-form limit. The
    # generated fixed-form wrapper also has to continue that list.
    suffix = ".f90"
    code = _long_function()

    def test_sum_of_ends(self):
        args = list(range(36))
        assert self.module.s(*args) == 0 + 35
        assert self.module.s(*[np.int32(v) for v in args]) == 35
