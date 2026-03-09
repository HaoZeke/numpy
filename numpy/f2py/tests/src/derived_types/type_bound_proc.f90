module tbp_mod
  implicit none

  type :: Counter
    integer :: count = 0
  contains
    procedure :: increment => counter_increment
    procedure :: get_count => counter_get_count
    procedure :: reset => counter_reset
  end type Counter

contains

  subroutine counter_increment(self, amount)
    class(Counter), intent(inout) :: self
    integer, intent(in) :: amount
    self%count = self%count + amount
  end subroutine counter_increment

  function counter_get_count(self) result(val)
    class(Counter), intent(in) :: self
    integer :: val
    val = self%count
  end function counter_get_count

  subroutine counter_reset(self)
    class(Counter), intent(inout) :: self
    self%count = 0
  end subroutine counter_reset

end module tbp_mod
