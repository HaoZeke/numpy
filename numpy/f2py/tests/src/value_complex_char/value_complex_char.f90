subroutine pass_complex8(z, o)
  complex(8), value :: z
  real(8), intent(out) :: o
  o = dble(z) + 2.0d0 * aimag(z)
end subroutine pass_complex8

subroutine pass_character(c, o)
  character, value :: c
  integer, intent(out) :: o
  o = ichar(c)
end subroutine pass_character

subroutine pass_character_len1(c, o)
  character(len=1), value :: c
  integer, intent(out) :: o
  o = ichar(c)
end subroutine pass_character_len1

subroutine pass_string(c, o)
  character(len=3), intent(in) :: c
  integer, intent(out) :: o
  o = ichar(c(1:1)) + len(c)
end subroutine pass_string

module value_mod
  use iso_c_binding
  implicit none
contains
  subroutine mod_complex_bindc(z, o) bind(c)
    complex(c_double_complex), value :: z
    real(c_double), intent(out) :: o
    o = dble(z) + 2.0d0 * aimag(z)
  end subroutine mod_complex_bindc

  subroutine mod_character_bindc(c, o) bind(c)
    character(kind=c_char), value :: c
    integer(c_int), intent(out) :: o
    o = ichar(c)
  end subroutine mod_character_bindc
end module value_mod
