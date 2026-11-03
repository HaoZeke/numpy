subroutine call_cb(cb, s)
  implicit none
  external cb
  character(len=*), intent(inout) :: s
  call cb(s)
end subroutine call_cb

subroutine call_string_function(cb, r)
  implicit none
  character(len=5), external :: cb
  character(len=5), intent(out) :: r
  r = cb()
end subroutine call_string_function

function hello() result(s)
  implicit none
  character(len=5) :: s
  s = 'hello'
end function hello
