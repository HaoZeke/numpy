module parameter_values
  implicit none
  integer, parameter :: answer = 100
  byte, parameter :: byte_value = -7
  byte, parameter :: byte_array(3) = [1, -2, 3]
  integer(kind=8), parameter :: wide = 1234567890123_8
  real(kind=8), parameter :: real_value = 1.25_8
  complex(kind=8), parameter :: complex_value = (2.0_8, 3.0_8)
  logical, parameter :: truth = .true.
  integer, parameter :: matrix(2,3) = reshape([1,2,3,4,5,6], [2,3])
  integer, parameter :: empty(0) = 0
  character(len=3), parameter :: word = 'abc'
  character(len=3), parameter :: words(2) = ['abc', 'def']
  character(len=0), parameter :: zero_word = ''
  character(len=0), parameter :: empty_words(0) = ''
  integer :: mutable = 0
  integer :: f2py_parameter_0 = 0
  integer :: f2py_initialize_parameters = 0
contains
  integer function compute_sum(offset) result(total)
    integer, intent(in) :: offset
    integer :: values(64), i
    do i = 1,64
      values(i) = i + offset
    end do
    total = sum(values)
  end function compute_sum

  integer function increment_mutable(amount) result(value)
    integer, intent(in) :: amount
    mutable = mutable + amount
    value = mutable
  end function increment_mutable

  integer function read_reserved_names() result(value)
    value = f2py_parameter_0 + f2py_initialize_parameters
  end function read_reserved_names
end module parameter_values

module intrinsic_name_parameters
  implicit none
  integer, parameter :: kind=7, len=8, size=9, reshape=10, max=11
end module intrinsic_name_parameters
