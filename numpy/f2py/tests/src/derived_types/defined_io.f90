! Test defined I/O as __str__ and from_string (F2018 12.6.4.8)
module defined_io_mod
  implicit none

  type :: formatted_point
    real :: x = 0.0
    real :: y = 0.0
    real :: z = 0.0
  contains
    procedure :: write_point
    generic :: write(formatted) => write_point
    procedure :: read_point
    generic :: read(formatted) => read_point
  end type formatted_point

  ! a record as long as asked for
  type :: long_record
    integer :: n = 0
  contains
    procedure :: write_long
    generic :: write(formatted) => write_long
  end type long_record

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

  subroutine read_point(dtv, unit, iotype, v_list, iostat, iomsg)
    class(formatted_point), intent(inout) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: v_list(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    read(unit, *, iostat=iostat) dtv%x, dtv%y, dtv%z
  end subroutine read_point

  subroutine write_long(dtv, unit, iotype, v_list, iostat, iomsg)
    class(long_record), intent(in) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: v_list(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    write(unit, '(A)', iostat=iostat) repeat('x', dtv%n)
  end subroutine write_long

  subroutine set_point(p, x, y, z)
    type(formatted_point), intent(inout) :: p
    real, intent(in) :: x, y, z
    p%x = x
    p%y = y
    p%z = z
  end subroutine set_point

end module defined_io_mod
