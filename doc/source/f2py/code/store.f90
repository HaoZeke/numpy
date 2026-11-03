module store
  implicit none
  real(kind=8), allocatable, dimension(:) :: x
contains
  subroutine fill(n)
    integer, intent(in) :: n
    integer :: i
    if (allocated(x)) deallocate(x)
    allocate(x(n))
    do i = 1, n
      x(i) = i
    end do
  end subroutine fill
  function total()
    real(kind=8) :: total
    total = 0
    if (allocated(x)) total = sum(x)
  end function total
end module store
