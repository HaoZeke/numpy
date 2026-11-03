complex(kind=8) function cplx_kind8(x)
  complex(kind=8), intent(in) :: x
  cplx_kind8 = x
end function

integer(kind=8) function int_kind8()
  int_kind8 = huge(int_kind8)
end function
