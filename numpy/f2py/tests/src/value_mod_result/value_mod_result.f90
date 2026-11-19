module valmod
  implicit none
contains
  function echo(z) result(r)
    complex(kind=8), value :: z
    complex(kind=8) :: r
    r = z
  end function echo

  function scale(x) result(r)
    real(kind=8), value :: x
    real(kind=8) :: r
    r = x
  end function scale
end module valmod
