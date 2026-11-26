module calc
  use kinds
  implicit none
contains
  integer function wider(x)
    real(kind=wp), intent(in) :: x
    if (x > 1.0_wp) then
      wider = 1
    else
      wider = 0
    end if
  end function wider

  integer function widerep(x)
    real(kind=ep), intent(in) :: x
    if (x > 1.0_ep) then
      widerep = 1
    else
      widerep = 0
    end if
  end function widerep
end module
