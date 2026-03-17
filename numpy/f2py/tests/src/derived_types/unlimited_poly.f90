! Test CLASS(*) unlimited polymorphism (F2018 7.3.2.3 paragraph 4)
module unlimited_poly_mod
  implicit none

  type :: container
    integer :: stored_tag = 0
    real :: stored_real = 0.0
    integer :: stored_int = 0
  end type container

contains

  subroutine store_value(c, item)
    type(container), intent(inout) :: c
    class(*), intent(in) :: item
    select type (item)
    type is (integer)
      c%stored_tag = 1
      c%stored_int = item
    type is (real)
      c%stored_tag = 2
      c%stored_real = item
    class default
      c%stored_tag = -1
    end select
  end subroutine store_value

  function get_tag(c) result(t)
    type(container), intent(in) :: c
    integer :: t
    t = c%stored_tag
  end function get_tag

end module unlimited_poly_mod
