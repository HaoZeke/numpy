module param_len_mod
  implicit none

  ! LEN-parameterized type (F2018 7.5.3)
  ! n is a runtime length parameter that determines array size.
  type :: FlexVec(n)
    integer, len :: n
    real(kind(0.0d0)) :: scale
    real(kind(0.0d0)) :: data(n)
  end type FlexVec

end module param_len_mod
