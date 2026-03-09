module mixed_types
  implicit none

  ! Non-bind(c) type with mixed scalar members
  type :: Particle
    real :: x, y, z
    real :: mass
    integer :: id
  end type Particle

  ! Type with only integers
  type :: GridIndex
    integer :: i, j, k
  end type GridIndex

  ! Type with doubles
  type :: Vector3D
    double precision :: vx, vy, vz
  end type Vector3D

end module mixed_types
