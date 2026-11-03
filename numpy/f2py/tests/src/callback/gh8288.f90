subroutine routine()
  !f2py intent(callback, hide) pyfunc
  integer i
  external pyfunc
  i = 1
  call pyfunc(i)
end subroutine routine

subroutine routine2()
  !f2py intent(callback, hide) pyfunc
  integer i
  external pyfunc
  i = 2
  call pyfunc(i)
end subroutine routine2
