! LLVM flang real(2), stored as C _Float16 and NumPy float16.
module halfmod
  implicit none
  real(2) :: x = 1.5_2
contains
  subroutine twice(a)
    real(2), intent(inout) :: a(2)
    a = 2 * a
  end subroutine twice
  function half_of(v) result(y)
    real(2), intent(in) :: v
    real(2) :: y
    y = v / 2
  end function half_of
  subroutine check_positive(v)
    real(2), intent(in) :: v
    !f2py check(v>0) v
  end subroutine check_positive
end module halfmod
