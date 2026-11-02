module long_double_kind
  implicit none
  integer, parameter :: ep = selected_real_kind(18)
  real(ep), parameter :: step = 1 + 2.0_ep**(-60)
  real(ep), parameter :: steps(2) = [step, 2 * step]
  complex(ep), parameter :: zstep = (step, -1)
contains
  subroutine twice(a)
    integer, parameter :: ek = selected_real_kind(18)
    real(ek), intent(inout) :: a(2)
    a = 2 * a
  end subroutine twice
  function half(x) result(y)
    integer, parameter :: ek = selected_real_kind(18)
    real(ek), intent(in) :: x
    real(ek) :: y
    y = x / 2
  end function half
end module long_double_kind
