subroutine spin(signum, niter, iters)
  ! Raise signum in the first iteration, then stop at the first
  ! f2py_check_signals call that reports a pending Python exception.
  use iso_c_binding, only: c_int
  implicit none
  integer, intent(in) :: signum, niter
  integer, intent(inout) :: iters(1)
  interface
    function f2py_check_signals() bind(c) result(pending)
      import :: c_int
      integer(c_int) :: pending
    end function f2py_check_signals
    function c_raise(sig) bind(c, name="raise") result(ret)
      import :: c_int
      integer(c_int), value :: sig
      integer(c_int) :: ret
    end function c_raise
  end interface
  integer :: i
  do i = 1, niter
    iters(1) = i
    if (i == 1) then
      if (c_raise(int(signum, c_int)) /= 0) return
    end if
    if (f2py_check_signals() /= 0) return
  end do
end subroutine spin

subroutine spin_threadsafe(signum, niter, iters)
  !f2py threadsafe
  implicit none
  integer, intent(in) :: signum, niter
  integer, intent(inout) :: iters(1)
  call spin(signum, niter, iters)
end subroutine spin_threadsafe

subroutine count_pending(signum, nchecks, npending)
  ! Raise signum once, then count how many of nchecks calls to
  ! f2py_check_signals report a pending Python exception.
  use iso_c_binding, only: c_int
  implicit none
  integer, intent(in) :: signum, nchecks
  integer, intent(inout) :: npending(1)
  interface
    function f2py_check_signals() bind(c) result(pending)
      import :: c_int
      integer(c_int) :: pending
    end function f2py_check_signals
    function c_raise(sig) bind(c, name="raise") result(ret)
      import :: c_int
      integer(c_int), value :: sig
      integer(c_int) :: ret
    end function c_raise
  end interface
  integer :: i
  npending(1) = 0
  if (c_raise(int(signum, c_int)) /= 0) return
  do i = 1, nchecks
    if (f2py_check_signals() /= 0) npending(1) = npending(1) + 1
  end do
end subroutine count_pending
