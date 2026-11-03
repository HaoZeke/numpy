! Inquiries of the real(16) of gfortran, which LLVM flang lacks on x86-64
module inquiry_quad
  implicit none
  integer, parameter :: qp = 16
  complex(qp) :: zq
  integer, parameter :: kind_q = kind(1.0_qp), digits_q = digits(1.0_qp), &
    precision_q = precision(1.0_qp), radix_q = radix(1.0_qp), &
    range_q = range(1.0_qp), maxexponent_q = maxexponent(1.0_qp), &
    minexponent_q = minexponent(1.0_qp), kind_zq = kind(zq), &
    precision_zq = precision(zq), range_zq = range(zq)
  ! the kinds that only real(16) has
  integer, parameter :: srk_19 = selected_real_kind(19), &
    srk_33 = selected_real_kind(33), srk_19_4932 = selected_real_kind(19, 4932)
  real(qp), parameter :: huge_q = huge(1.0_qp), tiny_q = tiny(1.0_qp), &
    epsilon_q = epsilon(1.0_qp)
contains
  subroutine bits(huge_bits, tiny_bits, epsilon_bits)
    integer(8), intent(out) :: huge_bits(2), tiny_bits(2), epsilon_bits(2)
    huge_bits = transfer(huge_q, huge_bits)
    tiny_bits = transfer(tiny_q, tiny_bits)
    epsilon_bits = transfer(epsilon_q, epsilon_bits)
  end subroutine bits
end module inquiry_quad
