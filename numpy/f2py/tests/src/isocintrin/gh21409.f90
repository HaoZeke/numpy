module ptrwidth
  use iso_c_binding, only: c_size_t, c_intptr_t, c_ptrdiff_t
  implicit none
contains
  subroutine add_size_t_arr(a, b, c)
    integer(c_size_t), intent(in) :: a(3), b(3)
    integer(c_size_t), intent(out) :: c(3)
    c = a + b
  end subroutine add_size_t_arr
  subroutine add_intptr_arr(a, b, c)
    integer(c_intptr_t), intent(in) :: a(3), b(3)
    integer(c_intptr_t), intent(out) :: c(3)
    c = a + b
  end subroutine add_intptr_arr
  subroutine add_ptrdiff_arr(a, b, c)
    integer(c_ptrdiff_t), intent(in) :: a(3), b(3)
    integer(c_ptrdiff_t), intent(out) :: c(3)
    c = a + b
  end subroutine add_ptrdiff_arr
  subroutine add_size_t(a, b, c)
    integer(c_size_t), intent(in) :: a, b
    integer(c_size_t), intent(out) :: c
    c = a + b
  end subroutine add_size_t
  subroutine bump_ptrdiff(n, a)
    integer, intent(in) :: n
    integer(c_ptrdiff_t), intent(inout) :: a(n)
    a = a + 1
  end subroutine bump_ptrdiff
  function intptr_sum(a, b) result(c)
    integer(c_intptr_t), intent(in) :: a, b
    integer(c_intptr_t) :: c
    c = a + b
  end function intptr_sum
end module ptrwidth
