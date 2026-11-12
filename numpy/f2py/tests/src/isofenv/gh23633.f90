subroutine fib_selected(a, n)
   ! calculate first Fibonacci numbers
   implicit none
   integer :: i, n
   integer, parameter :: dp = selected_real_kind(15, 50)
   real(kind=dp) :: a(n)

   do i = 1, n
   if (i .eq. 1) then
      a(i) = 0.0_dp
   elseif (i .eq. 2) then
      a(i) = 1.0_dp
   else
      a(i) = a(i - 1) + a(i - 2)
   end if
   end do
end subroutine fib_selected

subroutine fib_renamed(a, n)
   ! calculate first Fibonacci numbers
   use iso_fortran_env, only: dp => real64
   implicit none
   integer :: i, n
   real(kind=dp) :: a(n)

   do i = 1, n
   if (i .eq. 1) then
      a(i) = 0.0_dp
   elseif (i .eq. 2) then
      a(i) = 1.0_dp
   else
      a(i) = a(i - 1) + a(i - 2)
   end if
   end do
end subroutine fib_renamed

subroutine fib_intrinsic(a, n)
   ! calculate first Fibonacci numbers
   use iso_fortran_env
   implicit none
   integer :: i, n
   real(kind=real64) :: a(n)

   do i = 1, n
   if (i .eq. 1) then
      a(i) = 0.0_real64
   elseif (i .eq. 2) then
      a(i) = 1.0_real64
   else
      a(i) = a(i - 1) + a(i - 2)
   end if
   end do
end subroutine fib_intrinsic

subroutine fib_selected_out(a, n)
   ! calculate first Fibonacci numbers
   implicit none
   integer :: i
   integer, intent(in) :: n
   integer, parameter :: dp = selected_real_kind(15, 50)
   real(kind=dp), intent(out) :: a(n)

   do i = 1, n
   if (i .eq. 1) then
      a(i) = 0.0_dp
   elseif (i .eq. 2) then
      a(i) = 1.0_dp
   else
      a(i) = a(i - 1) + a(i - 2)
   end if
   end do
end subroutine fib_selected_out

subroutine fib_renamed_out(a, n)
   ! calculate first Fibonacci numbers
   use iso_fortran_env, only: dp => real64
   implicit none
   integer :: i
   integer, intent(in) :: n
   real(kind=dp), intent(out) :: a(n)

   do i = 1, n
   if (i .eq. 1) then
      a(i) = 0.0_dp
   elseif (i .eq. 2) then
      a(i) = 1.0_dp
   else
      a(i) = a(i - 1) + a(i - 2)
   end if
   end do
end subroutine fib_renamed_out
