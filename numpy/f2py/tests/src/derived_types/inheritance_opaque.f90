module inheritance_mod
  implicit none

  type :: Particle
    real :: mass
    integer :: charge
  end type Particle

  type, extends(Particle) :: ChargedParticle
    real :: spin
  end type ChargedParticle

  type, extends(ChargedParticle) :: Electron
    integer :: shell_num
  end type Electron

contains

  subroutine scale_mass(p, factor)
    type(Particle), intent(inout) :: p
    real, intent(in) :: factor
    p%mass = p%mass * factor
  end subroutine scale_mass

end module inheritance_mod
