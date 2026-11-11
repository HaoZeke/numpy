subroutine relax(x, n, nsweeps)
  ! Smooth the interior of x for up to nsweeps sweeps, returning early
  ! when Python has a pending exception, e.g. after Ctrl-C.
  use iso_c_binding, only: c_int
  implicit none
  integer, intent(in) :: n, nsweeps
  double precision, intent(inout) :: x(n)
  interface
    function f2py_check_signals() bind(c) result(pending)
      import :: c_int
      integer(c_int) :: pending
    end function f2py_check_signals
  end interface
  integer :: k
  do k = 1, nsweeps
    x(2:n-1) = 0.5d0 * (x(1:n-2) + x(3:n))
    if (f2py_check_signals() /= 0) return
  end do
end subroutine relax
