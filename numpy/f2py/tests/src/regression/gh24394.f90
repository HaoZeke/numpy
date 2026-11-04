subroutine integer4(value, result)
  integer(kind=4), intent(in) :: value
  integer(kind=4), intent(out) :: result
  result = value
end subroutine integer4

subroutine integer8(value, result)
  integer(kind=8), intent(in) :: value
  integer(kind=8), intent(out) :: result
  result = value
end subroutine integer8

subroutine real4(value, result)
  real(kind=4), intent(in) :: value
  real(kind=4), intent(out) :: result
  result = value
end subroutine real4

subroutine real8(value, result)
  real(kind=8), intent(in) :: value
  real(kind=8), intent(out) :: result
  result = value
end subroutine real8
