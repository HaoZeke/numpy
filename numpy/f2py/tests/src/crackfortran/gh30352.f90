! Resolving iso_fortran_env named kind parameters (gh-30352).
! Each module exercises a distinct use-statement scoping case; the parser
! must fold the standard kind constants into the kindselector so that
! e.g. real64 resolves to the same numeric kind as an explicit kind=8.
module test_only_rename
   use iso_fortran_env, only: dp => real64
   implicit none
contains
   subroutine s_complex(arr, nt)
      integer :: nt
      complex(kind=dp), intent(inout) :: arr(nt)
   end subroutine s_complex
end module test_only_rename

module test_bare_use
   use iso_fortran_env
   implicit none
contains
   subroutine s_real(x)
      real(real64), intent(inout) :: x
   end subroutine s_real
   subroutine s_int(y)
      integer(int32), intent(inout) :: y
   end subroutine s_int
end module test_bare_use

module test_only_missing
   use iso_fortran_env, only: int32
   implicit none
contains
   subroutine s_unresolved(x)
      ! real64 is not on the only-list, so it must stay unresolved.
      real(real64), intent(inout) :: x
   end subroutine s_unresolved
end module test_only_missing
