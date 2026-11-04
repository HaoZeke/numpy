module integer_extents
  implicit none
  integer, parameter :: n_exact = 4 / 2
  integer, parameter :: n_round = 5 / 2 * 2
  integer, parameter :: n_negative = -5 / 2 * 2
  integer, parameter :: n_mixed = 5 / 2.0 * 2
  integer, parameter :: n_from_real = 2.9
  integer, parameter :: n_from_double = 9.9d0
  integer, parameter :: n_text = len('kind(0.0)')
  integer, parameter :: n_power = 2 ** (-2)
  integer(kind=8), parameter :: n_large = ((2_8**62) + 1_8) / 3_8
  character(len=4/2), parameter :: direct_word = 'ab'
  character(len=n_exact), parameter :: word = 'ab'
  character(len=n_round), parameter :: rounded = 'abcd'
  character(len=n_mixed), parameter :: mixed = 'abcde'
  character(len=n_from_real), parameter :: converted = 'xy'
  character(len=n_from_double), parameter :: double_word = 'abcdefghi'
  character(len=n_text), parameter :: text_word = 'abcdefghi'
  integer, parameter :: direct_vector(5/2*2) = (/1, 2, 3, 4/)
  integer, parameter :: vector(n_round) = (/1, 2, 3, 4/)
end module integer_extents
