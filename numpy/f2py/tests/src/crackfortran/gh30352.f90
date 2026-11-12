module gh30352_kinds
   implicit none
   integer, parameter :: rk = kind(1.0d0)
end module gh30352_kinds

module gh30352
   use iso_fortran_env, only: dp => real64, ik => int64
   use iso_c_binding, only: cdp => c_double
   use gh30352_kinds, only: wp => rk
   implicit none
contains
   subroutine test_function(arr, nt)
      integer :: nt
      complex(kind=dp), intent(inout) :: arr(nt)
      arr(:) = (1.0_dp, -1.0_dp)
   end subroutine test_function

   subroutine set_cdp(arr, nt)
      integer :: nt
      complex(kind=cdp), intent(inout) :: arr(nt)
      arr(:) = (1.0_cdp, -1.0_cdp)
   end subroutine set_cdp

   subroutine set_wp(arr, nt)
      integer :: nt
      complex(kind=wp), intent(inout) :: arr(nt)
      arr(:) = (1.0_wp, -1.0_wp)
   end subroutine set_wp

   function add_dp(a, b) result(c)
      real(dp), intent(in) :: a, b
      real(dp) :: c
      c = a + b
   end function add_dp

   function add_ik(a, b) result(c)
      integer(ik), intent(in) :: a, b
      integer(ik) :: c
      c = a + b
   end function add_ik
end module gh30352
