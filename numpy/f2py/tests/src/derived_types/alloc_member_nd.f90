module alloc_member_nd_mod
  implicit none

  ! Multi-dimensional allocatable array members for testing
  ! 2D+ allocatable support (F2003 allocatable components, F2018 9.7.1)

  type :: Matrix
    integer :: rows
    integer :: cols
    double precision, allocatable :: data(:,:)
  end type Matrix

  type :: Tensor3D
    integer :: label
    real, allocatable :: field(:,:,:)
  end type Tensor3D

  type :: MixedAlloc
    ! Type with both 1D and 2D allocatable members
    integer :: tag
    double precision, allocatable :: vector(:)
    double precision, allocatable :: matrix(:,:)
  end type MixedAlloc

end module alloc_member_nd_mod
