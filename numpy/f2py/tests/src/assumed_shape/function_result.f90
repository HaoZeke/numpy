function sum_noresult(x)
  real(8), intent(in) :: x(:)
  real(8) :: sum_noresult
  sum_noresult = sum(x)
end function sum_noresult

function count_noresult(x)
  real(8), intent(in) :: x(:)
  integer(8) :: count_noresult
  count_noresult = count(x > 0)
end function count_noresult
