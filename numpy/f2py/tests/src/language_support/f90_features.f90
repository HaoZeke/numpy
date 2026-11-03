module f90_features
  implicit none

  interface add
    module procedure add_int, add_real
  end interface add

  interface operator(.plus.)
    module procedure add_real
  end interface

contains

  integer function add_int(a, b)
    integer, intent(in) :: a, b
    add_int = a + b
  end function add_int

  real(8) function add_real(a, b)
    real(8), intent(in) :: a, b
    add_real = a + b
  end function add_real

  recursive function factorial(n) result(r)
    integer, intent(in) :: n
    integer :: r
    if (n <= 1) then
      r = 1
    else
      r = n * factorial(n - 1)
    end if
  end function factorial

  elemental real(8) function square(x)
    real(8), intent(in) :: x
    square = x * x
  end function square

  pure subroutine shift(x, y)
    real(8), intent(in) :: x
    real(8), intent(out) :: y
    y = x + 1
  end subroutine shift

  subroutine host(x, y)
    real(8), intent(in) :: x
    real(8), intent(out) :: y
    y = triple(x)
  contains
    real(8) function triple(t)
      real(8), intent(in) :: t
      triple = 3 * t
    end function triple
  end subroutine host

  function twice(a, n) result(b)
    integer, intent(in) :: n
    real(8), intent(in) :: a(n)
    real(8) :: b(n)
    b = 2 * a
  end function twice

end module f90_features
