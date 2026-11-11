subroutine need_char(x)
  character x
end subroutine need_char

subroutine io_char(x)
  character x
  !f2py intent(inout) x
end subroutine io_char

subroutine dim_vec(a)
  real a(4)
end subroutine dim_vec

module m
  real, allocatable :: a(:)
end module m
