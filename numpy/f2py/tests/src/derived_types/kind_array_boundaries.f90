module kind_array_boundaries
  implicit none
  type :: kind_array(k)
    integer, kind :: k = kind(0.0d0)
    real(k) :: plane(2, 3)
    real(k), allocatable :: cube(:, :, :)
  end type kind_array
end module kind_array_boundaries
