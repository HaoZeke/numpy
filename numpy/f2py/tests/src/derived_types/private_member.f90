! Test PRIVATE component accessibility (F2018 7.5.4.8)
module private_member_mod
  implicit none

  ! Type with all-private default, one public override
  type :: with_private
    private
    integer :: hidden_x = 0
    real    :: hidden_y = 0.0
    integer, public :: visible_z = 42
  end type with_private

  ! Type with mixed accessibility (no PRIVATE statement,
  ! individual PRIVATE attributes)
  type :: mixed_access
    integer :: pub_a = 1
    integer, private :: priv_b = 2
    real :: pub_c = 3.0
  end type mixed_access

contains

  ! Subroutine that accesses private members (proves they exist)
  subroutine set_hidden(obj, x, y)
    type(with_private), intent(inout) :: obj
    integer, intent(in) :: x
    real, intent(in) :: y
    obj%hidden_x = x
    obj%hidden_y = y
  end subroutine set_hidden

  subroutine get_hidden(obj, x, y)
    type(with_private), intent(in) :: obj
    integer, intent(out) :: x
    real, intent(out) :: y
    x = obj%hidden_x
    y = obj%hidden_y
  end subroutine get_hidden

  function sum_mixed(obj) result(s)
    type(mixed_access), intent(in) :: obj
    real :: s
    s = real(obj%pub_a) + real(obj%priv_b) + obj%pub_c
  end function sum_mixed

end module private_member_mod
