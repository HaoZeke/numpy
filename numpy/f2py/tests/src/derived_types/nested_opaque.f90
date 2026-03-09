module nested_opaque_types
  implicit none

  type :: Vec2
    real :: x, y
  end type Vec2

  type :: Particle
    type(Vec2) :: pos
    type(Vec2) :: vel
    real :: mass
  end type Particle

end module nested_opaque_types
