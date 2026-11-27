import sys

import pytest

from . import util

CODES = [
    (
        """\
module moda
contains
  integer function foo()
    foo = 1
  end function foo
end module moda
module modb
contains
  integer function bar()
    bar = 2
  end function bar
end module modb
""",
        "moda --- foo()",
        "modb --- bar()",
    ),
    (
        """\
module moda
  integer :: x = 1
contains
  integer function foo()
    foo = 3
  end function foo
end module moda
module modb
  integer :: y = 2
end module modb
""",
        "moda --- x,foo()",
        "modb --- y",
    ),
]


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32",
                    reason="Fails with MinGW64 Gfortran (Issue #9673)")
@pytest.mark.parametrize("code,left,right", CODES)
def test_multimodule_summary_lines(code, left, right):
    mod = util.build_code(code, suffix=".f90")
    lines = [line.strip() for line in mod.__doc__.splitlines()]
    assert left in lines
    assert right in lines
