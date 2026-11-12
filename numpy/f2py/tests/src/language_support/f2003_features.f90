module f2003_features
  use iso_c_binding, only: c_int
  implicit none

  enum, bind(c)
    enumerator :: red = 1, green, blue
  end enum

  integer, protected :: counter = 7
  integer(c_int), bind(c, name="f2py_shared_count") :: shared_count = 5

contains

  integer function last_colour()
    last_colour = blue
  end function last_colour

  subroutine bump()
    counter = counter + 1
  end subroutine bump

  subroutine copy_value(x, y)
    real(8), intent(in), asynchronous :: x
    real(8), intent(out), volatile :: y
    y = x
  end subroutine copy_value

end module f2003_features
