function fixed_letters(value) result(output)
  implicit none
  character, intent(in) :: value(3)
  character :: output(3)
  output = value
end function fixed_letters

function fixed_strings(value) result(output)
  implicit none
  character(len=3), intent(in) :: value(3)
  character(len=3) :: output(3)
  output = value
end function fixed_strings

function letter_matrix(value) result(output)
  implicit none
  character, intent(in) :: value(2, 3)
  character :: output(2, 3)
  output = value
end function letter_matrix

function numeric_array(value) result(output)
  implicit none
  real(kind=8), intent(in) :: value(3)
  real(kind=8) :: output(3)
  output = value
end function numeric_array

module module_array_results
  implicit none
contains
  function module_letters(value) result(output)
    character, intent(in) :: value(3)
    character :: output(3)
    output = value
  end function module_letters
end module module_array_results
