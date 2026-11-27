module constants
  use, intrinsic :: iso_c_binding, only: c_int
  implicit none
  integer, parameter :: constant = 4
  integer :: non_parameter = 5
  integer, parameter :: vector(3) = (/1, 2, 3/)
  real, parameter :: ratio = 1.5
  character(len=3), parameter :: word = 'abc'
  character(len=2), parameter :: names(2) = ['ab', 'cd']
  integer, parameter :: dp = kind(1.0d0)
  real(dp), parameter :: wide = 2.5_dp
  integer(kind=selected_int_kind(9)), parameter :: counted = 6
  integer, parameter, private :: hk = selected_int_kind(9)
  integer(hk), parameter :: sized = 11
  integer(c_int), parameter :: from_c = 7
  logical, parameter :: flag = .true.
  complex, parameter :: zed = (1.0, 2.0)
contains
  subroutine get_constant(out)
    integer, intent(out) :: out
    out = constant
  end subroutine get_constant
end module constants
