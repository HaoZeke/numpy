module param_kind_nodefault_mod
  implicit none

  ! KIND parameter WITHOUT default value (F2018 7.5.3)
  ! User must specify k= at construction time.
  type :: GenericVec(k)
    integer, kind :: k
    real(k) :: x, y, z
  end type GenericVec

contains

  function generic_norm(v) result(d)
    type(GenericVec(kind(0.0d0))), intent(in) :: v
    real(kind(0.0d0)) :: d
    d = sqrt(v%x**2 + v%y**2 + v%z**2)
  end function generic_norm

end module param_kind_nodefault_mod
