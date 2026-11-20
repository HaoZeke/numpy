function f(x) result(y) bind(c, name="AddTwo")
  use iso_c_binding
  integer(c_int), value :: x
  integer(c_int) :: y
  y = x + 2
end function

function g(x) result(y) bind(c, name='MiXeD')
  use iso_c_binding
  integer(c_int), value :: x
  integer(c_int) :: y
  y = x + 3
end function
