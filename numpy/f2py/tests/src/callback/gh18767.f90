subroutine call_cb(cb, s)
  implicit none
  external cb
  character(len=*), intent(inout) :: s
  call cb(s)
end subroutine call_cb
