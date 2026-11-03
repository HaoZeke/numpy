module types_only
  implicit none
  type :: point
    real :: x, y
  end type point
end module types_only

module interfaces_only
  implicit none
  abstract interface
    subroutine callback(x)
      real, intent(in) :: x
    end subroutine callback
  end interface
end module interfaces_only

module wrapped
  implicit none
contains
  subroutine double_it(x, y)
    real, intent(in) :: x
    real, intent(out) :: y
    y = 2 * x
  end subroutine double_it
end module wrapped
