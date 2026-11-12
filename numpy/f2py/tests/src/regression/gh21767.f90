subroutine s(x)
  real(8) :: x
end subroutine s

module m
  real(8) :: v
contains
  subroutine t()
  end subroutine t
end module m

module vmod
  real(8) :: w
end module vmod

subroutine touch_common(x)
  real(8) :: x
  real(8) :: cval
  common /cblock/ cval
  cval = x
end subroutine touch_common
