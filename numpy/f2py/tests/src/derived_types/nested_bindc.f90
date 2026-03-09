module nested_bindc_types
  use iso_c_binding
  implicit none

  type, bind(c) :: Vec2
    real(c_double) :: x, y
  end type Vec2

  type, bind(c) :: Particle
    type(Vec2) :: pos
    type(Vec2) :: vel
    real(c_double) :: mass
  end type Particle

end module nested_bindc_types
