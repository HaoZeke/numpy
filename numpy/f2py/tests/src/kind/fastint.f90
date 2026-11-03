! The kind of c_int_fast16_t and c_int_fast32_t follows the compiler.
module fastmod
  use, intrinsic :: iso_c_binding, only: c_int_fast16_t, c_int_fast32_t
  implicit none
contains
  function add16(n) result(m)
    integer(c_int_fast16_t), intent(in) :: n
    integer(c_int_fast16_t) :: m
    m = n + 1_c_int_fast16_t
  end function add16
  function add32(n) result(m)
    integer(c_int_fast32_t), intent(in) :: n
    integer(c_int_fast32_t) :: m
    m = n + 1_c_int_fast32_t
  end function add32
end module fastmod
