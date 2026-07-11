import pytest

from . import util


@pytest.mark.slow
@pytest.mark.xfail(
    reason="gh-19157: f2py does not generate an allocatable-aware "
    "interface for an intent(out), allocatable dummy argument; the "
    "wrapper emits a fixed-shape interface and gfortran rejects the "
    "call ('Actual argument ... must be ALLOCATABLE'). The documented "
    "workaround is module-level allocatables (Method 2 in the "
    "Allocatable arrays docs section). Remove this xfail once "
    "allocatable dummy arguments are supported directly.",
    strict=True,
)
def test_gh19157_allocatable_out_argument_unsupported():
    util.build_module(
        [util.getpath("tests", "src", "regression", "gh19157.f90")],
        module_name="gh19157_alloc",
    )
