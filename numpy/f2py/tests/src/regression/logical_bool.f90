module logical_bool
  use iso_c_binding, only: c_intptr_t, c_loc
  implicit none
contains
  subroutine count_l1(a, n, cnt, addr)
    integer, intent(in) :: n
    logical(kind=1), intent(in), target :: a(n)
    integer, intent(out) :: cnt
    integer(c_intptr_t), intent(out) :: addr
    cnt = count(a)
    addr = transfer(c_loc(a), addr)
  end subroutine count_l1

  subroutine count_lstar1(a, n, cnt, addr)
    integer, intent(in) :: n
    logical*1, intent(in), target :: a(n)
    integer, intent(out) :: cnt
    integer(c_intptr_t), intent(out) :: addr
    cnt = count(a)
    addr = transfer(c_loc(a), addr)
  end subroutine count_lstar1

  subroutine count_l4(a, n, cnt, addr)
    integer, intent(in) :: n
    logical, intent(in), target :: a(n)
    integer, intent(out) :: cnt
    integer(c_intptr_t), intent(out) :: addr
    cnt = count(a)
    addr = transfer(c_loc(a), addr)
  end subroutine count_l4

  subroutine negate_l1(a, n)
    integer, intent(in) :: n
    logical(kind=1), intent(inout) :: a(n)
    a = .not. a
  end subroutine negate_l1

  subroutine negate_l1_inout(a, n)
    integer, intent(in) :: n
    logical(kind=1) :: a(n)
    !f2py intent(in,out) :: a
    a = .not. a
  end subroutine negate_l1_inout
end module logical_bool
