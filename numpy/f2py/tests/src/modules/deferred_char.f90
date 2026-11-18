module defchar
  implicit none
  character(len=:), allocatable :: names(:)
  character(len=:), allocatable :: grid(:,:)
contains
  subroutine setup()
    if (.not. allocated(names)) then
      allocate(character(len=4) :: names(2))
      names(1) = 'ab'
      names(2) = 'wxyz'
    end if
    if (.not. allocated(grid)) then
      allocate(character(len=2) :: grid(2, 3))
      grid = 'q'
      grid(1, 1) = 'xy'
    end if
  end subroutine setup
end module defchar
