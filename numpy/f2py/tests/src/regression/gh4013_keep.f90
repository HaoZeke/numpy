subroutine use_cb(fun, flag)
  external fun
  integer flag
  optional fun
!f2py intent(out) flag
  if (present(fun)) then
    call fun(flag)
  else
    flag = 0
  end if
end subroutine use_cb

subroutine use_charr(c, flag)
  character(len=1), optional, intent(in) :: c(2)
  integer, intent(out) :: flag
  if (present(c)) then
    flag = 1
  else
    flag = 0
  end if
end subroutine use_charr

subroutine use_alloc(x, flag)
  real, allocatable, optional, intent(in) :: x
  integer, intent(out) :: flag
  if (allocated(x)) then
    flag = 1
  else
    flag = 0
  end if
end subroutine use_alloc

subroutine use_ptr(x, flag)
  real, pointer, optional, intent(in) :: x
  integer, intent(out) :: flag
  if (associated(x)) then
    flag = 1
  else
    flag = 0
  end if
end subroutine use_ptr

subroutine use_value(x, flag)
  integer, optional, value, intent(in) :: x
  integer, intent(out) :: flag
  if (present(x)) then
    flag = 1
  else
    flag = 0
  end if
end subroutine use_value

subroutine use_cintent(x, flag)
  real, optional, intent(in) :: x
!f2py intent(c) x
  integer, intent(out) :: flag
  if (present(x)) then
    flag = 1
  else
    flag = 0
  end if
end subroutine use_cintent
