subroutine primary()
  external callback
  !f2py intent(callback, hide) :: callback
  call callback(1, 2)
end subroutine primary

subroutine secondary()
  external callback
  call callback(3, 4)
end subroutine secondary

subroutine tertiary()
  external callback
  !f2py intent(callback, hide) :: callback
  call callback(5, 6)
end subroutine tertiary

subroutine bound_a(x, n)
  integer :: n
  real :: x(n)
  external cbound
  !f2py intent(callback, hide) :: cbound
  call cbound(x, n)
end subroutine bound_a

subroutine bound_b(y, m)
  integer :: m
  real :: y(m)
  external cbound
  !f2py intent(callback, hide) :: cbound
  call cbound(y, m)
end subroutine bound_b

subroutine name_a(x)
  integer :: x
  external pyfunc
  !f2py intent(callback, hide) :: pyfunc
  call pyfunc(x)
end subroutine name_a

subroutine name_b(y)
  integer :: y
  external pyfunc
  !f2py intent(callback, hide) :: pyfunc
  call pyfunc(y)
end subroutine name_b

subroutine intent_a(x)
  integer, intent(in) :: x
  external cintent
  !f2py intent(callback, hide) :: cintent
  call cintent(x)
end subroutine intent_a

subroutine intent_b(y)
  integer :: y
  external cintent
  !f2py intent(callback, hide) :: cintent
  call cintent(y)
end subroutine intent_b
