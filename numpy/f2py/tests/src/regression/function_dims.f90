function sum_pow(n, a) result(total)
    integer, intent(in) :: n
    real, intent(in) :: a(*)
    real :: total
    total = sum(a(1:2**n))
end function

function sum_merge(n, a) result(total)
    integer, intent(in) :: n
    real, intent(in) :: a(*)
    real :: total
    total = sum(a(1:merge(n, 1, n > 0)))
end function
