module host_kinds
  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  integer, parameter :: nlen = 6, npts = 3

  type :: sample
    real(dp) :: a
    character(len=nlen) :: name
    real(dp) :: pts(npts)
  end type sample

contains

  subroutine fill(s)
    type(sample), intent(inout) :: s
    s%a = 1.0_dp / 3
    s%name = 'filled'
    s%pts = [1, 2, 3] / 3.0_dp
  end subroutine fill

end module host_kinds
