module gh22511_mod
  integer, public, parameter :: my_const = 1234
  real, public, parameter :: my_real = 3.14
  public :: dummy_sub
contains
  subroutine dummy_sub(x)
    integer, intent(in) :: x
  end subroutine dummy_sub
end module gh22511_mod
