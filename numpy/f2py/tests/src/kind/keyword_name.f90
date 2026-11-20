module keyword_kinds
  integer, parameter :: is = 2, in = 8
end module keyword_kinds

module keyword_funcs
  use keyword_kinds
contains
  function half_is(x) result(y)
    integer(is), intent(in) :: x
    integer(is) :: y
    y = x / 2
  end function half_is

  function third_in(x) result(y)
    real(in), intent(in) :: x
    real(in) :: y
    y = x / 3
  end function third_in

  integer(is) function prefix_is(x)
    integer(is), intent(in) :: x
    prefix_is = x / 2
  end function prefix_is
end module keyword_funcs

subroutine third_expr(x, y)
  integer, parameter :: is = 4
  real( kind = is * 2 ), intent(in) :: x
  real( is * 2 ), intent(out) :: y
  y = x / 3.0_8
end subroutine third_expr

subroutine third_derived(x, y)
  integer, parameter :: in = 15
  integer, parameter :: is = selected_real_kind(in)
  real( is ), intent(in) :: x
  real( kind = is ), intent(out) :: y
  y = x / 3.0_is
end subroutine third_derived

subroutine echo_len(s, out)
  integer, parameter :: in = 5
  character(len=in), intent(in) :: s
  character(len=in), intent(out) :: out
  out = s
end subroutine echo_len
