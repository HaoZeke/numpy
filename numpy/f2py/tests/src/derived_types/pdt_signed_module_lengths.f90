module pdt_signed_module_lengths
  use iso_c_binding, only: c_double
  implicit none
  type :: signed_cell(n)
    integer, len :: n = 0
    real(c_double) :: value
    real(c_double) :: values(n)
  end type signed_cell
  type(signed_cell(-1)) :: minus_one
  type(signed_cell(-3)) :: minus_three
  type(signed_cell(:)), allocatable :: optional
contains
  subroutine reset_values()
    minus_one%value = 11
    minus_three%value = 13
    if (allocated(optional)) deallocate(optional)
  end subroutine reset_values

  integer function native_length(which)
    integer, intent(in) :: which
    if (which == 1) then
      native_length = minus_one%n
    else
      native_length = minus_three%n
    end if
  end function native_length

  real(c_double) function native_value(which)
    integer, intent(in) :: which
    if (which == 1) then
      native_value = minus_one%value
    else
      native_value = minus_three%value
    end if
  end function native_value

  subroutine allocate_optional(length)
    integer, intent(in) :: length
    if (allocated(optional)) deallocate(optional)
    allocate(signed_cell(length) :: optional)
    optional%value = 17
  end subroutine allocate_optional
end module pdt_signed_module_lengths
