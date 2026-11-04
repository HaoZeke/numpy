function saved_top_function(callback, value) result(answer)
  integer(8), intent(in) :: value
  integer(8) :: answer
  interface
    function callback(value) result(callback_result)
      integer(8), intent(in) :: value
      integer(8) :: callback_result
    end function callback
  end interface
  answer = callback(value)
end function saved_top_function

subroutine saved_top_subroutine(callback, value, answer)
  integer(8), intent(in) :: value
  integer(8), intent(out) :: answer
  interface
    function callback(value) result(callback_result)
      integer(8), intent(in) :: value
      integer(8) :: callback_result
    end function callback
  end interface
  answer = callback(value)
end subroutine saved_top_subroutine

module saved_signature_module
  implicit none
contains
  function saved_child_function(callback, value) result(answer)
    integer(8), intent(in) :: value
    integer(8) :: answer
    interface
      function callback(value) result(callback_result)
        integer(8), intent(in) :: value
        integer(8) :: callback_result
      end function callback
    end interface
    answer = callback(value)
  end function saved_child_function
end module saved_signature_module
