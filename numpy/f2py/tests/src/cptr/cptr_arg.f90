module ptrmod
  use iso_c_binding, only: c_ptr, c_funptr, c_intptr_t
  implicit none
contains
  subroutine ptr_bits(p, n)
    type(c_ptr), intent(in) :: p
    integer(c_intptr_t), intent(out) :: n
    n = transfer(p, n)
  end subroutine ptr_bits

  subroutine funptr_bits(p, n)
    type(c_funptr), intent(in) :: p
    integer(c_intptr_t), intent(out) :: n
    n = transfer(p, n)
  end subroutine funptr_bits
end module ptrmod
