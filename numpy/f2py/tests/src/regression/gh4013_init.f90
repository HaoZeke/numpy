module gh4013init
contains
  real function with_init(x)
    real, intent(in), optional :: x
    !f2py real, intent(in), optional :: x = 4.0
    if (present(x)) then
      with_init = x
    else
      with_init = -1.0
    end if
  end function with_init
end module gh4013init
