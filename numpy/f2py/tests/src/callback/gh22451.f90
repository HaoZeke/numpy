function fun1(myfun1)
    implicit none
    real(8) :: fun1
    real(8), external :: myfun1
    fun1 = myfun1(1)
end function fun1

function fun3(b,myfun3)
    implicit none
    real(8) :: b,fun3
    real(8), external :: myfun3,fun2
    fun3 = fun2(b,myfun3)
end function fun3

function fun2(b,myfun2)
    implicit none
    real(8) :: b,fun2
    real(8), external :: myfun2,fun1
    b = fun1(myfun2)
    fun2 = b
end function fun2

subroutine consume(cb,result)
    implicit none
    integer, external :: cb
    integer, intent(out) :: result
    result = cb(2)
end subroutine consume

subroutine forward_sub(cb,result)
    implicit none
    integer, external :: cb
    integer, intent(out) :: result
    call consume(cb,result)
end subroutine forward_sub

function named_consumer(cb)
    implicit none
    real(8) :: named_consumer
    real(8), external :: cb
    integer :: myfun2
    myfun2 = 3
    named_consumer = cb(myfun2)
end function named_consumer

function named_forward(myfun2)
    implicit none
    real(8) :: named_forward
    real(8), external :: myfun2,named_consumer
    named_forward = named_consumer(myfun2)
end function named_forward

function nested_forward(cb)
    implicit none
    real(8) :: nested_forward
    real(8), external :: cb,fun1
    nested_forward = abs(fun1(cb))
end function nested_forward
