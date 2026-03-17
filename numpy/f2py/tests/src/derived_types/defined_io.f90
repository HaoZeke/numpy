! Test defined I/O as __str__ (F2018 12.6.4.8)
module defined_io_mod
  implicit none

  type :: formatted_point
    real :: x = 0.0
    real :: y = 0.0
    real :: z = 0.0
  contains
    procedure :: write_point
    generic :: write(formatted) => write_point
  end type formatted_point

contains

  subroutine write_point(dtv, unit, iotype, v_list, iostat, iomsg)
    class(formatted_point), intent(in) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: v_list(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    write(unit, '(A,F6.2,A,F6.2,A,F6.2,A)', iostat=iostat) &
      'point(', dtv%x, ', ', dtv%y, ', ', dtv%z, ')'
  end subroutine write_point

  subroutine set_point(p, x, y, z)
    type(formatted_point), intent(inout) :: p
    real, intent(in) :: x, y, z
    p%x = x
    p%y = y
    p%z = z
  end subroutine set_point

end module defined_io_mod
