! Inquiries of each integer and real kind that gfortran and LLVM flang
! both have, selected_real_kind(18) being real(10) on x86
module inquiry
  implicit none
  integer, parameter :: ep = selected_real_kind(18)
  integer :: i
  real(8) :: x
  complex :: z4
  complex(8) :: z8
  complex(ep) :: zep
  integer, parameter :: kind_i1 = kind(1_1), &
    digits_i1 = digits(1_1), &
    radix_i1 = radix(1_1), &
    range_i1 = range(1_1), &
    bit_size_i1 = bit_size(1_1)
  integer, parameter :: kind_i2 = kind(1_2), &
    digits_i2 = digits(1_2), &
    radix_i2 = radix(1_2), &
    range_i2 = range(1_2), &
    bit_size_i2 = bit_size(1_2)
  integer, parameter :: kind_i4 = kind(1_4), &
    digits_i4 = digits(1_4), &
    radix_i4 = radix(1_4), &
    range_i4 = range(1_4), &
    bit_size_i4 = bit_size(1_4)
  integer, parameter :: kind_i8 = kind(1_8), &
    digits_i8 = digits(1_8), &
    radix_i8 = radix(1_8), &
    range_i8 = range(1_8), &
    bit_size_i8 = bit_size(1_8)
  integer, parameter :: kind_i16 = kind(1_16), &
    digits_i16 = digits(1_16), &
    radix_i16 = radix(1_16), &
    range_i16 = range(1_16), &
    bit_size_i16 = bit_size(1_16)
  integer, parameter :: huge_i1 = huge(1_1)
  integer, parameter :: huge_i2 = huge(1_2)
  integer, parameter :: huge_i4 = huge(1_4)
  integer(8), parameter :: huge_i8 = huge(1_8)
  ! the halves of huge(1_16), as numpy has no 128-bit integer
  integer(8), parameter :: huge_i16_high = huge(1_16) / 2_16**64, &
    huge_i16_low = mod(huge(1_16), 2_16**63)
  integer, parameter :: kind_r4 = kind(1.0_4), &
    digits_r4 = digits(1.0_4), &
    precision_r4 = precision(1.0_4), &
    radix_r4 = radix(1.0_4), &
    range_r4 = range(1.0_4), &
    maxexponent_r4 = maxexponent(1.0_4), &
    minexponent_r4 = minexponent(1.0_4)
  real(4), parameter :: huge_r4 = huge(1.0_4), &
    tiny_r4 = tiny(1.0_4), &
    epsilon_r4 = epsilon(1.0_4)
  integer, parameter :: kind_r8 = kind(1.0_8), &
    digits_r8 = digits(1.0_8), &
    precision_r8 = precision(1.0_8), &
    radix_r8 = radix(1.0_8), &
    range_r8 = range(1.0_8), &
    maxexponent_r8 = maxexponent(1.0_8), &
    minexponent_r8 = minexponent(1.0_8)
  real(8), parameter :: huge_r8 = huge(1.0_8), &
    tiny_r8 = tiny(1.0_8), &
    epsilon_r8 = epsilon(1.0_8)
  integer, parameter :: kind_rep = kind(1.0_ep), &
    digits_rep = digits(1.0_ep), &
    precision_rep = precision(1.0_ep), &
    radix_rep = radix(1.0_ep), &
    range_rep = range(1.0_ep), &
    maxexponent_rep = maxexponent(1.0_ep), &
    minexponent_rep = minexponent(1.0_ep)
  real(ep), parameter :: huge_rep = huge(1.0_ep), &
    tiny_rep = tiny(1.0_ep), &
    epsilon_rep = epsilon(1.0_ep)
  integer, parameter :: kind_z4 = kind(z4), &
    precision_z4 = precision(z4), &
    range_z4 = range(z4)
  integer, parameter :: kind_z8 = kind(z8), &
    precision_z8 = precision(z8), &
    range_z8 = range(z8)
  integer, parameter :: kind_zep = kind(zep), &
    precision_zep = precision(zep), &
    range_zep = range(zep)
  integer, parameter :: huge_var = huge(i), precision_var = precision(x)
end module inquiry
