module param_kind_mod
  use iso_fortran_env, only: real64
  implicit none

  ! Simple KIND-parameterized type (F2018 7.5.3)
  ! The KIND parameter k defaults to double precision.
  type :: RealVec(k)
    integer, kind :: k = kind(0.0d0)
    integer :: length
    real(k) :: x, y, z
  end type RealVec

contains

  subroutine set_realvec(v, vx, vy, vz, vlen)
    type(RealVec(kind(0.0d0))), intent(inout) :: v
    real(kind(0.0d0)), intent(in) :: vx, vy, vz
    integer, intent(in) :: vlen
    v%x = vx
    v%y = vy
    v%z = vz
    v%length = vlen
  end subroutine set_realvec

  function realvec_norm(v) result(d)
    type(RealVec(kind(0.0d0))), intent(in) :: v
    real(kind(0.0d0)) :: d
    d = sqrt(v%x**2 + v%y**2 + v%z**2)
  end function realvec_norm

end module param_kind_mod
