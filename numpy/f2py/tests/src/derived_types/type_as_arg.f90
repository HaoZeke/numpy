module type_arg_mod
  implicit none

  type :: Particle
    real :: x, y, z
    real :: mass
  end type Particle

contains

  subroutine translate_particle(p, dx, dy, dz)
    type(Particle), intent(inout) :: p
    real, intent(in) :: dx, dy, dz
    p%x = p%x + dx
    p%y = p%y + dy
    p%z = p%z + dz
  end subroutine translate_particle

  function particle_distance(p1, p2) result(dist)
    type(Particle), intent(in) :: p1, p2
    real :: dist
    dist = sqrt((p1%x - p2%x)**2 + (p1%y - p2%y)**2 + (p1%z - p2%z)**2)
  end function particle_distance

  subroutine make_particle(x, y, z, mass, p)
    real, intent(in) :: x, y, z, mass
    type(Particle), intent(out) :: p
    p%x = x
    p%y = y
    p%z = z
    p%mass = mass
  end subroutine make_particle

  function create_particle(x, y, z, mass) result(p)
    real, intent(in) :: x, y, z, mass
    type(Particle) :: p
    p%x = x
    p%y = y
    p%z = z
    p%mass = mass
  end function create_particle

end module type_arg_mod
