module routine_args
  implicit none

  type :: account
    integer :: id = 0
    real(kind(0.0d0)) :: balance = 0.0d0
    character(len=8) :: owner = ''
  end type account

contains

  subroutine deposit(acc, amounts, n, who, fee)
    type(account), intent(inout) :: acc
    integer, intent(in) :: n
    real(kind(0.0d0)), intent(inout) :: amounts(n)
    character(len=*), intent(in) :: who
    real(kind(0.0d0)), intent(in), optional :: fee
    acc%balance = acc%balance + sum(amounts)
    if (present(fee)) acc%balance = acc%balance - fee
    acc%owner = who
    amounts = 0.0d0
  end subroutine deposit

  function total(accounts, weights) result(s)
    type(account), intent(in) :: accounts(:)
    real(kind(0.0d0)), intent(in) :: weights(size(accounts))
    real(kind(0.0d0)) :: s
    s = sum(accounts%balance * weights)
  end function total

  subroutine open_account(id, acc)
    integer, intent(in) :: id
    type(account), intent(out) :: acc
    acc%id = id
    acc%balance = 1.0d0
  end subroutine open_account

  subroutine apply(acc, rule)
    type(account), intent(inout) :: acc
    external rule
    real(kind(0.0d0)) :: rule, b
    b = acc%balance
    acc%balance = rule(b)
  end subroutine apply

  function interest(acc, rate) result(gain)
    type(account), intent(in) :: acc
    real(kind(0.0d0)), intent(in) :: rate
    real(kind(0.0d0)) :: gain
    gain = acc%balance * rate
  end function interest

end module routine_args
