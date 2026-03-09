module array_bindc_types
  use iso_c_binding
  implicit none

  ! bind(c) type with fixed-size array members
  type, bind(c) :: Matrix2x2
    real(c_double) :: data(4)  ! stored flat: 2x2
    real(c_double) :: scale
  end type Matrix2x2

  type, bind(c) :: Vec3
    real(c_float) :: v(3)
  end type Vec3

end module array_bindc_types
