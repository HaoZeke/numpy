module constants
  implicit none
  integer, parameter :: constant = 4
  integer :: non_parameter = 5
  integer, parameter :: vector(3) = (/1, 2, 3/)
  real, parameter :: ratio = 1.5
  character(len=3), parameter :: word = 'abc'
  character(len=*), parameter :: phrase = 'don''t'
  character(len=2), parameter :: names(2) = (/'ab', 'cd'/)
  integer, parameter :: implied(*) = [4, 6, 8]
  integer, parameter :: memcpy = 3
  integer, parameter :: word_length = 3
end module constants

module constants_word
  implicit none
  integer, parameter :: length = 7
end module constants_word
