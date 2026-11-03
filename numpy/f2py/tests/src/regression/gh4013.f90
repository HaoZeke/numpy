module gh4013
contains
  real function foo(x)
    real, intent(in), optional :: x
    if (present(x)) then
      foo = x
    else
      foo = 1
    end if
  end function foo

  integer function pair(x, y)
    integer, intent(in), optional :: x, y
    pair = 0
    if (present(x)) pair = pair + 100 + x
    if (present(y)) pair = pair + 10 + y
  end function pair

  complex function complex_value(x)
    complex, intent(in), optional :: x
    complex_value = (-1, -2)
    if (present(x)) complex_value = x
  end function complex_value

  integer function logical_value(x)
    logical, intent(in), optional :: x
    logical_value = -1
    if (present(x)) then
      logical_value = 0
      if (x) logical_value = 1
    end if
  end function logical_value

  integer function text_length(x)
    character(len=*), intent(in), optional :: x
    text_length = -1
    if (present(x)) text_length = len_trim(x)
  end function text_length

  integer function array_size(x)
    real, intent(in), optional :: x(:)
    array_size = -1
    if (present(x)) array_size = size(x)
  end function array_size

  integer function character_array_size(x)
    character(len=4), intent(in), optional :: x(:)
    character_array_size = -1
    if (present(x)) character_array_size = size(x)
  end function character_array_size

  real function fixed_sum(x)
    real, intent(in), optional :: x(2)
    fixed_sum = -1
    if (present(x)) fixed_sum = sum(x)
  end function fixed_sum

  subroutine array_inout(x, was_present)
    real, intent(inout), optional :: x(:)
    integer, intent(out) :: was_present
    was_present = 0
    if (present(x)) then
      was_present = 1
      x = 2 * x
    end if
  end subroutine array_inout

  subroutine value_probe(x, res)
    integer, intent(in), optional, value :: x
    integer, intent(out) :: res
    res = -1
    if (present(x)) res = x
  end subroutine value_probe

  integer function value_function(x)
    integer, intent(in), optional, value :: x
    value_function = -1
    if (present(x)) value_function = x
  end function value_function

  integer function mixed_value(base, x)
    integer, intent(in), value :: base
    integer, intent(in), optional, value :: x
    mixed_value = base
    if (present(x)) mixed_value = base + x
  end function mixed_value

  real function defaulted(x)
    real, intent(in), optional :: x
    !f2py real :: x = 3
    defaulted = -1
    if (present(x)) defaulted = x
  end function defaulted

  integer function value_defaulted(x)
    integer, intent(in), optional, value :: x
    !f2py integer :: x = 3
    value_defaulted = -1
    if (present(x)) value_defaulted = x
  end function value_defaulted
end module gh4013

real function external_optional(x)
  real, intent(in), optional :: x
  external_optional = -1
  if (present(x)) external_optional = x
end function external_optional

subroutine external_value(x, res)
  integer, intent(in), optional, value :: x
  integer, intent(out) :: res
  res = -1
  if (present(x)) res = x
end subroutine external_value

real function external_defaulted(x)
  real, intent(in), optional :: x
  !f2py real :: x = 3
  external_defaulted = -1
  if (present(x)) external_defaulted = x
end function external_defaulted

subroutine external_value_defaulted(x, res)
  integer, intent(in), optional, value :: x
  !f2py integer :: x = 3
  integer, intent(out) :: res
  res = -1
  if (present(x)) res = x
end subroutine external_value_defaulted
