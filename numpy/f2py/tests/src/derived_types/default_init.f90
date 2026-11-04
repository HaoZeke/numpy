module default_init_mod
  implicit none

  ! Component default initialization (F2018 7.5.4.1, R739)
  ! Members with declared defaults should use those values
  ! when not explicitly provided in the constructor

  type :: Config
    integer :: max_iterations = 100
    double precision :: tolerance = 1.0d-6
    logical :: verbose = .false.
    double precision :: scale_factor = 1.0d0
  end type Config

end module default_init_mod
