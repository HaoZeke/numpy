subroutine gh16357(fun, a, b)
  external fun
  integer, intent(in) :: a, b
  call fun(a, b)
end subroutine gh16357
