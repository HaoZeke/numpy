subroutine explicit_subroutine_callback(callback, value, answer)
  integer(8), intent(in) :: value
  integer(8), intent(out) :: answer
  interface
    subroutine callback(value, output)
      integer(8), intent(in) :: value
      integer(8), intent(out) :: output
    end subroutine callback
  end interface
  call callback(value, answer)
end subroutine explicit_subroutine_callback
