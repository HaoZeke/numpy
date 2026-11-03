! Parameters that f2py evaluates as gfortran and LLVM flang do
module parameter_values
  implicit none
  integer, parameter :: dp = 8
  integer, parameter :: k_sum = kind(1.0) + kind(1), k_dp = kind(0.0_dp)
  integer, parameter :: k_mixed = kind(1_8 * 1.0), k_max = kind(max(1, 2_8))
  integer, parameter :: n_single = int(0.29 * 100)
  integer, parameter :: n_rounded = int((2**24 + 1) * 1.0)
  integer, parameter :: n_half = nint(0.49999999999999994)
  integer, parameter :: n_len = len('it''s')
  integer(8), parameter :: n_kinds = 2_8 + 3_8
  real, parameter :: third = 1.0 / 3, big = 1e38 * 3
  real(dp), parameter :: tenth = 0.1_dp, twice = 2 * tenth, wide = 0.1
  character(len=*), parameter :: word = 'it''s' // ' ok'
  logical, parameter :: yes = 3 > 2 .and. .not. 1 /= 1
end module parameter_values
