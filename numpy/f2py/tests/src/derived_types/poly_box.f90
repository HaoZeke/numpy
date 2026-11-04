module poly_box
  implicit none

  ! The container and its tag do not need class(*).
  ! store_value, which does, lives in unlimited_poly.

  type :: container
    integer :: stored_tag = 0
    real :: stored_real = 0.0
    integer :: stored_int = 0
  end type container

contains

  function get_tag(c) result(t)
    type(container), intent(in) :: c
    integer :: t
    t = c%stored_tag
  end function get_tag

end module poly_box
