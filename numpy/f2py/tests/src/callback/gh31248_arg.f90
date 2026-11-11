subroutine solve(n, monitor)
   implicit none
   integer, intent(in) :: n
   external monitor
   integer :: i
   double precision :: x
   x = 1d0
   do i = 1, n
      x = x / 2
      call advance(i, x, monitor)
   end do
end subroutine solve

subroutine advance(step, x, monitor)
   implicit none
   integer, intent(in) :: step
   double precision, intent(in) :: x
   external monitor
   call monitor(step, x)
end subroutine advance
