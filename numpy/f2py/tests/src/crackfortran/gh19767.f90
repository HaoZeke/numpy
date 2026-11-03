real(kind=8) function mysqrt(x) result(r)
  real(kind=8), intent(in) :: x
  r = x*x
end function

real*8 function real_star8(x) result(r)
  real*8, intent(in) :: x
  r = x
end function

pure real(kind=8) function real_pure(x) result(r)
  real(kind=8), intent(in) :: x
  r = x
end function

integer(kind=8) function int_kind8(x) result(r)
  integer(kind=8), intent(in) :: x
  r = x
end function

complex(kind=8) function cplx_kind8(x) result(r)
  complex(kind=8), intent(in) :: x
  r = x
end function

character(len=5) function char_len5(x) result(r)
  character(len=5), intent(in) :: x
  r = x
end function

module gh19767_mod
  implicit none
contains
  real(kind=8) function mysqrt(x) result(r)
    real(kind=8), intent(in) :: x
    r = x*x
  end function
end module

integer function int_implicit(x) result(r)
  implicit real*8 (a-h, o-z)
  integer, intent(in) :: x
  r = x
end function
