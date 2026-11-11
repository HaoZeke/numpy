subroutine blah(list, n)
    integer, intent(in) :: n
    real, intent(out) :: list(2*n,0:2**n)

    list = 1.0

end subroutine

! gh-7184
subroutine s(n, a)
  integer :: n, a(1:2**n)
end
