module type_assumed_shape_mod
  implicit none

  type :: Vec3
    real :: x, y, z
  end type Vec3

contains

  subroutine sum_vectors(vecs, total_x, total_y, total_z)
    type(Vec3), intent(in) :: vecs(:)
    real, intent(out) :: total_x, total_y, total_z
    integer :: i
    total_x = 0.0
    total_y = 0.0
    total_z = 0.0
    do i = 1, size(vecs)
      total_x = total_x + vecs(i)%x
      total_y = total_y + vecs(i)%y
      total_z = total_z + vecs(i)%z
    end do
  end subroutine sum_vectors

end module type_assumed_shape_mod
