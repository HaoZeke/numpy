import pytest

from . import util


@pytest.mark.slow
class TestLogicalScalarDefault(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine flag_true(out, a)
  integer, intent(out) :: out
  logical, intent(in) :: a
  !f2py logical, optional :: a = .true.
  if (a) then
    out = 1
  else
    out = 0
  end if
end subroutine flag_true

subroutine flag_false(out, a)
  integer, intent(out) :: out
  logical, intent(in) :: a
  !f2py logical, optional :: a = .false.
  if (a) then
    out = 1
  else
    out = 0
  end if
end subroutine flag_false

subroutine flag_ktrue(out, a)
  integer, intent(out) :: out
  logical, intent(in) :: a
  !f2py logical, optional :: a = .true._1
  if (a) then
    out = 1
  else
    out = 0
  end if
end subroutine flag_ktrue

subroutine flag_kfalse(out, a)
  integer, intent(out) :: out
  logical, intent(in) :: a
  !f2py logical, optional :: a = .false._1
  if (a) then
    out = 1
  else
    out = 0
  end if
end subroutine flag_kfalse

subroutine flag_ntrue(out, a)
  integer, intent(out) :: out
  integer, parameter :: lk = 1
  logical, intent(in) :: a
  !f2py logical, optional :: a = .true._lk
  if (a) then
    out = 1
  else
    out = 0
  end if
end subroutine flag_ntrue

subroutine flag_nfalse(out, a)
  integer, intent(out) :: out
  integer, parameter :: lk = 1
  logical, intent(in) :: a
  !f2py logical, optional :: a = .false._lk
  if (a) then
    out = 1
  else
    out = 0
  end if
end subroutine flag_nfalse
"""

    @pytest.mark.parametrize(("name", "expected"), [
        ("flag_true", 1),
        ("flag_false", 0),
        ("flag_ktrue", 1),
        ("flag_kfalse", 0),
        ("flag_ntrue", 1),
        ("flag_nfalse", 0),
    ])
    def test_omitted_logical_scalar_default(self, name, expected):
        assert getattr(self.module, name)() == expected
