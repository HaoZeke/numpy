! c_size_t scalars need an unsigned converter of the width of size_t.
module sizemod
  use, intrinsic :: iso_c_binding, only: c_size_t
  implicit none
contains
  function addone(n) result(m)
    integer(c_size_t), intent(in) :: n
    integer(c_size_t) :: m
    m = n + 1_c_size_t
  end function addone
end module sizemod
