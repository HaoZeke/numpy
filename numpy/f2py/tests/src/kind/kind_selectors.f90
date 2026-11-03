! Arguments whose kinds are given by the kind intrinsics
module kind_selectors
  implicit none
contains
  subroutine twice(x, y, i, j, z, w)
    real(kind=selected_real_kind(15)), intent(in) :: x
    real(kind(1.0d0)), intent(out) :: y
    integer(kind=selected_int_kind(18)), intent(in) :: i
    integer(selected_int_kind(18)), intent(out) :: j
    complex(kind=kind(1.0d0)), intent(in) :: z
    complex(kind(1.0d0)), intent(out) :: w
    y = 2 * x
    j = 2 * i
    w = 2 * z
  end subroutine twice
end module kind_selectors
