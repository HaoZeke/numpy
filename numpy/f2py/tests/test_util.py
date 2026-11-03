import pytest

from . import util


@pytest.mark.slow
def test_check_language_rejects_a_broken_snippet():
    # A compiler that exists must still reject a snippet that does not compile.
    if not util.check_language("fortran"):
        pytest.skip("No Fortran compiler available")
    broken = "this is not fortran\n"
    assert util.check_language("fortran", broken) is False
    good = "program p\nend program p\n"
    assert util.check_language("fortran", good) is True
