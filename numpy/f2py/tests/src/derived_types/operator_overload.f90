module operator_mod
  implicit none

  type :: Vec2
    real :: x, y
  end type Vec2

  interface operator(+)
    module procedure vec2_add
  end interface

  interface operator(-)
    module procedure vec2_sub
  end interface

  interface operator(==)
    module procedure vec2_eq
  end interface

contains

  function vec2_add(a, b) result(c)
    type(Vec2), intent(in) :: a, b
    type(Vec2) :: c
    c%x = a%x + b%x
    c%y = a%y + b%y
  end function vec2_add

  function vec2_sub(a, b) result(c)
    type(Vec2), intent(in) :: a, b
    type(Vec2) :: c
    c%x = a%x - b%x
    c%y = a%y - b%y
  end function vec2_sub

  function vec2_eq(a, b) result(eq)
    type(Vec2), intent(in) :: a, b
    logical :: eq
    eq = (a%x == b%x) .and. (a%y == b%y)
  end function vec2_eq

end module operator_mod
