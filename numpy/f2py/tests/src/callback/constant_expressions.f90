module constant_callback_expressions
  use iso_c_binding, only: c_double, c_double_complex
  implicit none
contains
  real(c_double) function call_sum(callback) result(answer)
    real(c_double), external :: callback
    answer = callback(1.25_c_double + 2.5_c_double)
  end function call_sum

  double precision function call_double_literal(callback) result(answer)
    double precision, external :: callback
    answer = callback(1.25d0)
  end function call_double_literal

  complex(c_double_complex) function call_complex_literal(callback) result(answer)
    complex(c_double_complex), external :: callback
    answer = callback((1.25_c_double, -2.5_c_double))
  end function call_complex_literal
end module constant_callback_expressions
