module generic_tbp_mod
  implicit none

  ! Generic type-bound procedures (F2018 7.5.5, R751)
  ! GENERIC :: generic-name => specific1, specific2

  type :: Scaler
    double precision :: value
  contains
    procedure :: scale_int => scaler_scale_int
    procedure :: scale_real => scaler_scale_real
    ! F2018 R751: generic binding maps one name to multiple specifics
    generic :: scale => scale_int, scale_real
  end type Scaler

contains

  subroutine scaler_scale_int(self, factor)
    ! Scale value by an integer factor
    class(Scaler), intent(inout) :: self
    integer, intent(in) :: factor
    self%value = self%value * dble(factor)
  end subroutine scaler_scale_int

  subroutine scaler_scale_real(self, factor)
    ! Scale value by a real factor
    class(Scaler), intent(inout) :: self
    double precision, intent(in) :: factor
    self%value = self%value * factor
  end subroutine scaler_scale_real

end module generic_tbp_mod
