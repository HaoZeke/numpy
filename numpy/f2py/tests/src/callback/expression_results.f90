module callback_expression_results
  use iso_c_binding, only: c_double
  implicit none
contains
  real(c_double) function call_named_kind(hook) result(answer)
    real(c_double), external :: hook
    answer = hook(1.25_c_double)
  end function call_named_kind

  real(kind=8) function call_numeric_kind(hook) result(answer)
    real(kind=8), external :: hook
    answer = hook(1.25_8)
  end function call_numeric_kind

  real(c_double) function call_expression(hook, value) result(answer)
    real(c_double), external :: hook
    real(c_double), intent(in) :: value
    answer = hook(value + 1.25_c_double)
  end function call_expression
end module callback_expression_results
