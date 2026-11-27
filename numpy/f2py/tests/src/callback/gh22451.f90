function fun1(myfun1)
  implicit none
  real(8) :: fun1
  real(8), external :: myfun1
  fun1 = myfun1(1)
end function fun1

function fun3(b, myfun3)
  implicit none
  real(8) :: b, fun3
  real(8), external :: myfun3, fun2
  fun3 = fun2(b, myfun3)
end function fun3

function fun2(b, myfun2)
  implicit none
  real(8) :: b, fun2
  real(8), external :: myfun2, fun1
  b = fun1(myfun2)
  fun2 = b
end function fun2
