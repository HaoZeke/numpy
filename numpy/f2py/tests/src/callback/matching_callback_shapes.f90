subroutine primary()
  implicit none
  double precision values(2, 3)
  external callback
  !f2py intent(callback, hide) :: callback
  !f2py intent(in) :: values
  values = reshape((/ 1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0, 6.0d0 /), shape(values))
  call callback(values)
end subroutine primary

subroutine secondary()
  implicit none
  double precision values(2, 3)
  external callback
  !f2py intent(callback, hide) :: callback
  !f2py intent(in) :: values
  values = reshape((/ 11.0d0, 12.0d0, 13.0d0, 14.0d0, 15.0d0, 16.0d0 /), shape(values))
  call callback(values)
end subroutine secondary
