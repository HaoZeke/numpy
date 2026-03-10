module complex_member_opaque_mod
  implicit none

  type :: wavefunction
    double complex :: amplitude
    double precision :: phase
  end type wavefunction

contains

  subroutine set_amplitude(wf, amp)
    type(wavefunction), intent(inout) :: wf
    double complex, intent(in) :: amp
    wf%amplitude = amp
  end subroutine

end module complex_member_opaque_mod
