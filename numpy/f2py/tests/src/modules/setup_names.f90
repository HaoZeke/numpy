module setup_names
  implicit none
  ! names the setup function of the module uses for itself
  integer :: i_f2py = 1
  integer :: f2py_setup_names_def = 2
end module setup_names

subroutine set_common()
  integer :: i_f2py
  common /names/ i_f2py
  i_f2py = 3
end subroutine set_common
