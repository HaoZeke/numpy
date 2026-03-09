module array_opaque_types
  implicit none

  ! Non-bind(c) type with fixed-size array members
  type :: Spectrum
    real :: wavelengths(5)
    real :: intensities(5)
    integer :: npeaks
  end type Spectrum

  type :: Triangle
    real :: vertices(9)  ! 3 vertices x 3 coords, stored flat
  end type Triangle

end module array_opaque_types
