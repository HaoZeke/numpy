module intent_out_type_array_mod
  implicit none

  type :: IndexedVal
    real :: value
    integer :: idx
  end type IndexedVal

contains

  subroutine make_sequence(start, count, results, nresults)
    real, intent(in) :: start
    integer, intent(in) :: count
    type(IndexedVal), allocatable, intent(out) :: results(:)
    integer, intent(out) :: nresults
    integer :: i

    nresults = count
    allocate(results(count))

    do i = 1, count
      results(i)%value = start + real(i - 1)
      results(i)%idx = i
    end do
  end subroutine make_sequence

end module intent_out_type_array_mod
