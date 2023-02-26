! gh-23276
module cmplxdat
  implicit none
  integer :: i, j
  real :: x, y
  complex(kind=8), target :: medium_ref_index

  data i, j / 2, 3 /
  data x, y / 1.5, 2.0 /
  data medium_ref_index / (1.d0, 0.d0) /
end module cmplxdat
