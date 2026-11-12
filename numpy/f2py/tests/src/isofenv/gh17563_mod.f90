module gh17563_mod
  use, intrinsic :: iso_fortran_env, only: wp => real64, ik => int32
  use, intrinsic :: iso_c_binding, only: cdp => c_double
  implicit none
contains
  subroutine rescale(v, factor, n)
    use, intrinsic :: iso_fortran_env, only: int32
    real(wp), intent(inout) :: v(:)
    real(kind=wp), intent(in) :: factor
    integer(ik), intent(out) :: n
    v = v * factor
    n = int(size(v), int32)
  end subroutine rescale
  function total(v) result(s)
    real(wp), intent(in) :: v(:)
    real(wp) :: s
    s = sum(v)
  end function total
  ! gh-30352
  subroutine fill(arr, nt)
    integer, intent(in) :: nt
    complex(kind=wp), intent(inout) :: arr(nt)
    arr(:) = (1.0_wp, -1.0_wp)
  end subroutine fill
  subroutine cscale(z, factor)
    complex(cdp), intent(inout) :: z(:)
    real(cdp), intent(in) :: factor
    z = z * factor
  end subroutine cscale
end module gh17563_mod
