function fun1(myfun1)
    implicit none
    real(8) :: fun1
    real(8),external :: myfun1
    fun1 = myfun1(1)
    return
end function fun1

function fun2(b,myfun2)
   implicit none
   real(8):: b,fun2
   real(8),external:: myfun2,fun1
   b = fun1(myfun2)
   fun2 = b
   return
end function fun2
