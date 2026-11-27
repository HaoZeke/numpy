subroutine call_complex(z, f, r)
  implicit none
  complex, intent(in) :: z
  external f
  complex :: f
  complex, intent(out) :: r
  r = f(z)
end subroutine call_complex

subroutine call_double_complex(z, f, r)
  implicit none
  double complex, intent(in) :: z
  external f
  double complex :: f
  double complex, intent(out) :: r
  r = f(z)
end subroutine call_double_complex
