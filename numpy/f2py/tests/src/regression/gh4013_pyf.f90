real function pyf_absent_s(x)
  real, intent(in), optional :: x
  if (present(x)) then
    pyf_absent_s = x
  else
    pyf_absent_s = 1.0
  end if
end function

real function pyf_optional_s(x)
  real, intent(in), optional :: x
  if (present(x)) then
    pyf_optional_s = x
  else
    pyf_optional_s = 1.0
  end if
end function
