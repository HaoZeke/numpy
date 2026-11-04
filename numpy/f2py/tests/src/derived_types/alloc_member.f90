module alloc_member_mod
  implicit none

  type :: DataVec
    integer :: tag
    real, allocatable :: values(:)
  end type DataVec

end module alloc_member_mod
