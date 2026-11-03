module proc_pointer_pass
  implicit none

  type :: counter
    integer :: n = 0
    procedure(step_iface), pointer :: step
    procedure(check_iface), pointer, nopass :: check => null()
  end type counter

  type, extends(counter) :: named_counter
    integer :: id = 0
  end type named_counter

  abstract interface
    subroutine step_iface(self, k)
      import :: counter
      class(counter), intent(inout) :: self
      integer, intent(in) :: k
    end subroutine step_iface
    function check_iface(flag, x) result(ok)
      logical, intent(in) :: flag
      double precision, intent(in) :: x
      logical :: ok
    end function check_iface
  end interface

contains

  subroutine add(self, k)
    class(counter), intent(inout) :: self
    integer, intent(in) :: k
    self%n = self%n + k
  end subroutine add

  function positive(flag, x) result(ok)
    logical, intent(in) :: flag
    double precision, intent(in) :: x
    logical :: ok
    ok = flag .and. x > 0
  end function positive

  subroutine arm(c)
    class(counter), intent(inout) :: c
    c%step => add
    c%check => positive
  end subroutine arm

end module proc_pointer_pass
