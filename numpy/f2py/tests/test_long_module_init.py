import pytest

import numpy as np

from . import util


@pytest.mark.slow
@pytest.mark.parametrize("nlen", [56, 63])
def test_long_module_name(nlen):
    stem = "m" * nlen
    names = [stem[:-1] + "x", stem[:-1] + "y"]
    code = "".join(
        f"module {name}\n"
        "contains\n"
        "  subroutine s(n)\n"
        "    integer, intent(inout) :: n\n"
        f"    n = n + {i + 1}\n"
        "  end subroutine\n"
        "end module\n"
        for i, name in enumerate(names)
    )
    mod = util.build_code(
        code, suffix=".f90", module_name=f"longmod{nlen}")
    for i, name in enumerate(names):
        arr = np.array(0, dtype=np.int32)
        getattr(mod, name).s(arr)
        assert int(arr) == i + 1
