module operator_dispatch_mod
  use iso_c_binding
  implicit none

  type, bind(c) :: small_value
    integer(c_int) :: value
  end type

  type :: wide_value
    integer :: value
  contains
    procedure :: assign_from => wide_method
  end type

  type, bind(c) :: fixed_pair
    integer(c_int) :: value(2)
  end type

  type :: tracked_value
    integer :: value
  contains
    final :: tracked_finalize
  end type

  integer :: finalized = 0

  interface operator(+)
    module procedure small_add, small_positive, small_add_int, int_add_small
    module procedure small_add_long, small_add_real, small_add_double
    module procedure small_add_wide, tracked_add
    module procedure small_array_sum
  end interface
  interface operator(-)
    module procedure small_negative, small_sub_int, int_sub_small
  end interface
  interface operator(*)
    module procedure small_product, small_complex_product
  end interface
  interface operator(==)
    module procedure small_equal, int_equal_small, bool_equal_small
  end interface
  interface operator(<)
    module procedure int_less_small
  end interface
  interface assignment(=)
    module procedure small_assign_int, small_assign_small
    module procedure tracked_assign_small, wide_assign_int, pair_assign_int
  end interface

contains

  function small_add(a, b) result(c)
    type(small_value), intent(in) :: a, b
    type(small_value) :: c
    c%value = a%value + b%value
  end function

  function small_positive(a) result(c)
    type(small_value), intent(in) :: a
    type(small_value) :: c
    c%value = a%value
  end function

  function small_negative(a) result(c)
    type(small_value), intent(in) :: a
    type(small_value) :: c
    c%value = -a%value
  end function

  function small_add_int(a, b) result(c)
    type(small_value), intent(in) :: a
    integer(c_int), intent(in) :: b
    type(small_value) :: c
    c%value = a%value + b
  end function

  function int_add_small(a, b) result(c)
    integer(c_int), intent(in) :: a
    type(small_value), intent(in) :: b
    type(small_value) :: c
    c%value = a + b%value
  end function

  function small_add_long(a, b) result(c)
    type(small_value), intent(in) :: a
    integer(c_long_long), intent(in) :: b
    type(small_value) :: c
    c%value = a%value + int(b) + 1000
  end function

  function small_add_real(a, b) result(c)
    type(small_value), intent(in) :: a
    real(c_float), intent(in) :: b
    type(small_value) :: c
    c%value = a%value + int(b)
  end function

  function small_add_double(a, b) result(c)
    type(small_value), intent(in) :: a
    real(c_double), intent(in) :: b
    type(small_value) :: c
    c%value = a%value + int(b) + 100
  end function

  function small_sub_int(a, b) result(c)
    type(small_value), intent(in) :: a
    integer(c_int), intent(in) :: b
    type(small_value) :: c
    c%value = a%value - b
  end function

  function int_sub_small(a, b) result(c)
    integer(c_int), intent(in) :: a
    type(small_value), intent(in) :: b
    type(small_value) :: c
    c%value = a - b%value
  end function

  function small_add_wide(a, b) result(c)
    type(small_value), intent(in) :: a
    type(wide_value), intent(in) :: b
    type(wide_value) :: c
    c%value = a%value + b%value
  end function

  function small_product(a, b) result(c)
    type(small_value), intent(in) :: a, b
    integer(c_int) :: c
    c = a%value * b%value
  end function

  function small_equal(a, b) result(equal)
    type(small_value), intent(in) :: a, b
    logical :: equal
    equal = a%value == b%value
  end function

  function small_complex_product(a, b) result(c)
    type(small_value), intent(in) :: a
    complex(c_double_complex), intent(in) :: b
    complex(c_double_complex) :: c
    c = real(a%value, c_double) * b
  end function

  function bool_equal_small(a, b) result(equal)
    logical(c_bool), intent(in) :: a
    type(small_value), intent(in) :: b
    logical :: equal
    equal = (a .eqv. (b%value /= 0))
  end function

  function int_equal_small(a, b) result(equal)
    integer(c_int), intent(in) :: a
    type(small_value), intent(in) :: b
    logical :: equal
    equal = a == b%value
  end function

  function int_less_small(a, b) result(less)
    integer(c_int), intent(in) :: a
    type(small_value), intent(in) :: b
    logical :: less
    less = a < b%value
  end function

  function tracked_add(a, b) result(c)
    type(tracked_value), intent(in) :: a, b
    type(tracked_value) :: c
    c%value = a%value + b%value
  end function

  function small_array_sum(a, b) result(total)
    type(small_value), intent(in) :: a(:), b(:)
    integer :: total
    total = sum(a%value) + sum(b%value)
  end function

  subroutine small_assign_int(destination, source)
    type(small_value), intent(out) :: destination
    integer(c_int), intent(in) :: source
    destination%value = source + 10
  end subroutine

  subroutine small_assign_small(destination, source)
    type(small_value), intent(out) :: destination
    type(small_value), intent(in) :: source
    destination%value = source%value + 50
  end subroutine

  subroutine tracked_assign_small(destination, source)
    type(tracked_value), intent(out) :: destination
    type(small_value), intent(in) :: source
    destination%value = 2 * source%value
  end subroutine

  subroutine wide_assign_int(destination, source)
    type(wide_value), intent(out) :: destination
    integer(c_int), intent(in) :: source
    destination%value = 10 * source
  end subroutine

  function wide_method(self, increment) result(total)
    class(wide_value), intent(in) :: self
    integer, intent(in) :: increment
    integer :: total
    total = self%value + increment + 10000
  end function

  subroutine pair_assign_int(destination, source)
    type(fixed_pair), intent(out) :: destination
    integer(c_int), intent(in) :: source
    destination%value = [source, source + 1]
  end subroutine

  subroutine tracked_finalize(value)
    type(tracked_value), intent(inout) :: value
    finalized = finalized + 1
  end subroutine

  function get_finalized() result(count)
    integer :: count
    count = finalized
  end function

end module
