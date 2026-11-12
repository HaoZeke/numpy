subroutine report(step, x)
   implicit none
   integer, intent(in) :: step
   double precision, intent(in) :: x
   external on_step
   call on_step(step, x)
end subroutine report

subroutine run(n)
   implicit none
   integer, intent(in) :: n
   integer :: i
   double precision :: x
   x = 1d0
   do i = 1, n
      x = x / 2
      call report(i, x)
   end do
end subroutine run
