module f90_return_long_double
  use iso_c_binding, only: c_long_double
  implicit none
contains
  function t16(value)
    real(kind=c_long_double) :: value
    real(kind=c_long_double) :: t16
    t16 = value
  end function t16

  subroutine s16(t16, value)
    real(kind=c_long_double) :: value
    real(kind=c_long_double) :: t16
!f2py intent(out) t16
    t16 = value
  end subroutine s16
end module f90_return_long_double
