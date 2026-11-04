module array_total
  implicit none

contains

  ! An assumed-shape argument, with no class entity in this file.
  function total(x) result(s)
    integer, intent(in) :: x(:)
    integer :: s
    s = sum(x)
  end function total

end module array_total
