! Scalar check() coverage. The expressions are C and run in the wrapper.
! A false expression raises the module error.

subroutine chk_signed_char(x)
  use iso_c_binding, only: c_int8_t
  integer(c_int8_t), intent(in) :: x
  !f2py check(x>0) x
end subroutine chk_signed_char

subroutine chk_short(x)
  use iso_c_binding, only: c_short
  integer(c_short), intent(in) :: x
  !f2py check(x>0) x
end subroutine chk_short

subroutine chk_integer2(x)
  integer*2, intent(in) :: x
  !f2py check(x>0) x
end subroutine chk_integer2

subroutine chk_int(x)
  integer, intent(in) :: x
  !f2py check(x>0) x
end subroutine chk_int

subroutine chk_long(x)
  use iso_c_binding, only: c_long
  integer(c_long), intent(in) :: x
  !f2py check(x>0) x
end subroutine chk_long

subroutine chk_long_long(x)
  use iso_c_binding, only: c_long_long
  integer(c_long_long), intent(in) :: x
  !f2py check(x<0) x
end subroutine chk_long_long

subroutine chk_float(x)
  use iso_c_binding, only: c_float
  real(c_float), intent(in) :: x
  !f2py check(x<0) x
end subroutine chk_float

subroutine chk_double(x)
  use iso_c_binding, only: c_double
  real(c_double), intent(in) :: x
  !f2py check(x<0) x
end subroutine chk_double

subroutine chk_double0(x)
  double precision, intent(in) :: x
  !f2py check(x>0) x
end subroutine chk_double0

subroutine chk_long_double(x)
  use iso_c_binding, only: c_long_double
  real(c_long_double), intent(in) :: x
  !f2py check(x<0) x
end subroutine chk_long_double

subroutine chk_complex_float(x)
  use iso_c_binding, only: c_float_complex
  complex(c_float_complex), intent(in) :: x
  !f2py check(x.r<0) x
end subroutine chk_complex_float

subroutine chk_complex_double(x)
  use iso_c_binding, only: c_double_complex
  complex(c_double_complex), intent(in) :: x
  !f2py check(x.r<0) x
end subroutine chk_complex_double

subroutine chk_complex_long_double(x)
  use iso_c_binding, only: c_long_double_complex
  complex(c_long_double_complex), intent(in) :: x
  !f2py check(x.r<0) x
end subroutine chk_complex_long_double

subroutine chk_byte(x)
  !f2py byte, intent(in), check(x>0) :: x
end subroutine chk_byte

subroutine chk_character(x)
  character, intent(in) :: x
  !f2py check(x=='a') x
end subroutine chk_character

subroutine chk_string(x)
  character(len=4), intent(in) :: x
  !f2py check(x[0]=='a') x
end subroutine chk_string

subroutine chk_str(x)
  character(len=5), intent(in) :: x
  !f2py check(x[0]=='a') x
end subroutine chk_str

subroutine chk_str_pct(x)
  character(len=4), intent(in) :: x
  !f2py check(x[0]=='%') x
end subroutine chk_str_pct

subroutine chk_str8(x)
  character*8, intent(in) :: x
  !f2py check(*x=='a') x
end subroutine chk_str8

subroutine chk_strlen(x)
  character*8, intent(in) :: x
  !f2py check(slen(x)%2==1) x
end subroutine chk_strlen

subroutine chk_logical(x)
  logical, intent(in) :: x
  !f2py check(x==0) x
end subroutine chk_logical

subroutine chk_logical1(x)
  logical*1, intent(in) :: x
  !f2py check(x==0) x
end subroutine chk_logical1

subroutine chk_logical2(x)
  logical*2, intent(in) :: x
  !f2py check(x==0) x
end subroutine chk_logical2

subroutine chk_logical8(x)
  logical*8, intent(in) :: x
  !f2py check(x==0) x
end subroutine chk_logical8

subroutine chk_mod(x)
  integer, intent(in) :: x
  !f2py check(x%2==0) x
end subroutine chk_mod

subroutine chk_two(a, b)
  integer, intent(in) :: a
  integer, intent(in) :: b
  !f2py check(a>0) a
  !f2py check(b>0) b
end subroutine chk_two
