module type_array_arg_mod
  implicit none

  type :: Atom
    real :: x, y, z
    integer :: atomic_num
  end type Atom

contains

  subroutine sum_positions(atoms, n, total_x, total_y, total_z)
    integer, intent(in) :: n
    type(Atom), intent(in) :: atoms(n)
    real, intent(out) :: total_x, total_y, total_z
    integer :: i
    total_x = 0.0
    total_y = 0.0
    total_z = 0.0
    do i = 1, n
      total_x = total_x + atoms(i)%x
      total_y = total_y + atoms(i)%y
      total_z = total_z + atoms(i)%z
    end do
  end subroutine sum_positions

  function count_element(atoms, n, z) result(cnt)
    integer, intent(in) :: n
    type(Atom), intent(in) :: atoms(n)
    integer, intent(in) :: z
    integer :: cnt
    integer :: i
    cnt = 0
    do i = 1, n
      if (atoms(i)%atomic_num == z) cnt = cnt + 1
    end do
  end function count_element

end module type_array_arg_mod
