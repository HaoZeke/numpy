module finalizer_mod
  implicit none

  ! F2018 7.5.6: Final subroutines
  ! Tests that finalization runs when the opaque wrapper deallocates
  ! the Fortran object. Per F2018 7.5.6.3 paragraph 2, deallocation
  ! of an allocatable entity triggers finalization.

  ! Module-level counter to track finalizer invocations
  integer :: finalize_count = 0

  type :: Tracked
    integer :: value
  contains
    ! F2018 7.5.6.1 (R753): FINAL statement
    final :: tracked_finalize
  end type Tracked

contains

  subroutine tracked_finalize(self)
    ! Final subroutine (F2018 C786): one non-optional, nonpointer,
    ! nonallocatable, nonpolymorphic dummy argument of the type
    type(Tracked), intent(inout) :: self
    finalize_count = finalize_count + 1
  end subroutine tracked_finalize

  function get_finalize_count() result(count)
    integer :: count
    count = finalize_count
  end function get_finalize_count

  subroutine reset_finalize_count()
    finalize_count = 0
  end subroutine reset_finalize_count

end module finalizer_mod
