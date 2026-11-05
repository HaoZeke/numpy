module specification_intrinsics_mod
  use, intrinsic :: iso_fortran_env, only: working_real => real64
  implicit none
  integer, parameter :: rounded = 5 / 2 * 2
  integer, parameter :: lower = -5 / 2 * 2
  integer, parameter :: tiny_count = merge(7, 0, tiny(1.0) > 0.0)
  integer, parameter :: dimensions(-1:0) = (/2, 3/)
  character(len=5/2*2), parameter :: label = 'ABCD'
contains
  subroutine constant_bounds(output)
    integer, intent(out) :: output(lower:lower+rounded-1)
    output = 41
  end subroutine constant_bounds

  subroutine intrinsic_bounds(output)
    integer, intent(out) :: output(max(2,tiny_count),min(3,dimensions(0)))
    output = 43
  end subroutine intrinsic_bounds

  subroutine inquiry_bounds(output)
    real(working_real) :: scalar
    integer, intent(out) :: output(ceiling(-log10(tiny(scalar))))
    output = 47
  end subroutine inquiry_bounds

  integer function native_inquiry_extent()
    real(working_real) :: scalar
    native_inquiry_extent = ceiling(-log10(tiny(scalar)))
  end function native_inquiry_extent

  subroutine runtime_bounds(n, output)
    integer, intent(in) :: n
    integer, intent(out) :: output(n/2*2)
    output = 53
  end subroutine runtime_bounds

  subroutine runtime_signed(n, output)
    integer, intent(in) :: n
    integer, intent(out) :: output(-n/2*(-2))
    output = 59
  end subroutine runtime_signed

  subroutine copy_label(output)
    character(len=len(label)), intent(out) :: output
    output = label
  end subroutine copy_label

  subroutine dynamic_allocation(n, total)
    integer, intent(in) :: n
    integer, intent(out) :: total
    integer, allocatable :: work(:)
    allocate(work(-2:n-3))
    work = 7
    total = sum(work) + size(work)
    deallocate(work)
  end subroutine dynamic_allocation
end module specification_intrinsics_mod
