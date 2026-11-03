module tbp_with_arrays
  implicit none

  type :: counter
    integer :: n = 0
  contains
    procedure :: add
  end type counter

contains

  subroutine add(self, k)
    class(counter), intent(inout) :: self
    integer, intent(in) :: k
    self%n = self%n + k
  end subroutine add

  ! an array argument makes the module define rank as a macro
  function total(x) result(s)
    integer, intent(in) :: x(:)
    integer :: s
    s = sum(x)
  end function total

end module tbp_with_arrays
