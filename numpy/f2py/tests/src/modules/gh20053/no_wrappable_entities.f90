module type_only
  implicit none
  type :: point
    real :: x, y
  end type point
end module type_only

module interface_only
  implicit none
  interface
    subroutine external_sub(n)
      integer, intent(in) :: n
    end subroutine external_sub
  end interface
end module interface_only

subroutine twice(x, y)
  real(8), intent(in) :: x
  real(8), intent(out) :: y
  y = 2 * x
end subroutine twice
