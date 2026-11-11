! numpy/f2py/tests/src/regression/gh4013.f90
module gh4013
contains
  real function s_scalar(x)
    real, intent(in), optional :: x
    if (present(x)) then
      s_scalar = x
    else
      s_scalar = 1.0
    end if
  end function
  real function s_array(x)
    real, intent(in), optional :: x(:)
    if (present(x)) then
      s_array = sum(x)
    else
      s_array = -1.0
    end if
  end function
  integer function n_present(a, b)
    integer, intent(in), optional :: a
    real, intent(in), optional :: b(3)
    n_present = 0
    if (present(a)) n_present = n_present + 1
    if (present(b)) n_present = n_present + 2
  end function
  subroutine s_inout(x, flag)
    integer, intent(inout), optional :: x
    integer, intent(out) :: flag
    if (present(x)) then
      flag = 1
      x = x + 10
    else
      flag = 0
    end if
  end subroutine
  subroutine s_out(x, flag)
    real, intent(out), optional :: x
    integer, intent(out) :: flag
    if (present(x)) then
      flag = 1
      x = 4.0
    else
      flag = 0
    end if
  end subroutine
  integer function s_str(x)
    character(len=*), intent(in), optional :: x
    if (present(x)) then
      s_str = len(x)
    else
      s_str = -5
    end if
  end function
end module
