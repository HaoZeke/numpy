module constants
  implicit none
  integer, parameter :: n = 3, word_length = 4
  integer, parameter :: scalar = 1234
  real(kind=8), parameter :: wide = 2.5
  complex(kind=8), parameter :: z = (1.5, -2.0)
  logical, parameter :: truth = .true.
  integer(kind=1), parameter :: small(3) = (/1, -2, 3/)
  integer(kind=8), parameter :: large = 1234567890123_8
  integer, parameter :: vector(n) = (/4, 6, 8/)
  integer, parameter :: matrix(2,n) = reshape((/1,2,3,4,5,6/), (/2,n/))
  integer, parameter :: empty(0) = [integer ::]
  integer, parameter :: empty_negative(1-n) = [integer ::]
  character(len=-word_length), parameter :: zero_word = ''
  character(len=word_length), parameter :: word = 'test'
  character(len=word_length), parameter :: words(2) = (/'abcd', 'efgh'/)
  integer, parameter :: memcpy = 19, size_t = 23, complex_double = 37
  integer, parameter :: f2py_copy_index = 41
  integer, parameter :: f2py_constants_def_0_data = 43
  integer :: mutable = 42
contains
  recursive subroutine clobber(seed, checksum)
    integer, intent(in) :: seed
    integer, intent(out) :: checksum
    integer, volatile :: scratch(8192)
    integer :: i
    do i = 1, size(scratch)
      scratch(i) = seed + i
    end do
    checksum = scratch(1) + scratch(8192)
  end subroutine clobber

  subroutine read_mutable(value)
    integer, intent(out) :: value
    value = mutable
  end subroutine read_mutable

  subroutine set_mutable(value)
    integer, intent(in) :: value
    mutable = value
  end subroutine set_mutable
end module constants
