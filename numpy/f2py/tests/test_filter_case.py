import pytest

from numpy.f2py.tests import util

CODE = """\
module m
contains
  subroutine AlPhA(n)
    integer, intent(in) :: n
  end subroutine AlPhA
  subroutine BeTa(n)
    integer, intent(in) :: n
  end subroutine BeTa
end module m
"""


@pytest.mark.slow
@pytest.mark.parametrize(
    ("tag", "kwargs", "kept", "dropped"),
    [
        ("only", {"only": ["Alpha"]}, "alpha", "beta"),
        ("skip", {"skip": ["Beta"]}, "alpha", "beta"),
    ],
)
def test_filter_ignores_case(tag, kwargs, kept, dropped):
    mod = util.build_code(
        CODE, suffix=".f90", module_name=f"filter_case_{tag}", **kwargs)
    assert hasattr(mod.m, kept)
    assert not hasattr(mod.m, dropped)
