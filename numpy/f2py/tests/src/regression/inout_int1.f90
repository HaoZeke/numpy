subroutine inc_int1(a)
  integer*1, intent(inout) :: a
  a = a + 1
end subroutine inc_int1

subroutine inc_c_int8(a)
  use iso_c_binding, only: c_int8_t
  integer(c_int8_t), intent(inout) :: a
  a = a + 1_c_int8_t
end subroutine inc_c_int8
