module pointer_init_mod
  implicit none

  ! Test that crackfortran correctly handles pointer component
  ! default initialization with => null() (F2018 7.5.4.6, R741)

  double precision, target :: target_array(3)

  type :: PtrHolder
    integer :: label
    double precision, pointer :: data(:) => null()
  end type PtrHolder

contains

  subroutine setup_target()
    target_array(1) = 100.0d0
    target_array(2) = 200.0d0
    target_array(3) = 300.0d0
  end subroutine setup_target

  subroutine link_data(holder)
    type(PtrHolder), intent(inout) :: holder
    call setup_target()
    holder%data => target_array
  end subroutine link_data

end module pointer_init_mod
