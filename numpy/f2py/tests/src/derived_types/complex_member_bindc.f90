module complex_member_bindc_mod
  use iso_c_binding
  implicit none

  type, bind(c) :: impedance
    real(c_double) :: resistance
    complex(c_double_complex) :: reactance
  end type impedance

  type, bind(c) :: signal_sample
    complex(c_float_complex) :: value
    real(c_float) :: timestamp
  end type signal_sample

contains

  subroutine scale_impedance(z, factor)
    type(impedance), intent(inout) :: z
    real(c_double), intent(in) :: factor
    z%resistance = z%resistance * factor
    z%reactance = z%reactance * factor
  end subroutine

end module complex_member_bindc_mod
