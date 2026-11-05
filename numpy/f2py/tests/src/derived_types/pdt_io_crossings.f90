module pdt_io_primary
  implicit none
  private
  type, public :: io_cell(n)
    integer, len :: n = 2
    integer :: value = 0
    integer :: values(n)
  contains
    procedure :: write_cell
    generic :: write(formatted) => write_cell
    procedure :: read_cell
    generic :: read(formatted) => read_cell
  end type io_cell
  type(io_cell(3)), public :: shared
  type(io_cell(2)), public, protected :: locked
  public :: reset_values
contains
  subroutine write_cell(dtv, unit, iotype, v_list, iostat, iomsg)
    class(io_cell(*)), intent(in) :: dtv
    integer, intent(in) :: unit, v_list(:)
    character(*), intent(in) :: iotype
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    write(unit, '(i0,1x,i0)', iostat=iostat, iomsg=iomsg) dtv%n, dtv%value
  end subroutine write_cell

  subroutine read_cell(dtv, unit, iotype, v_list, iostat, iomsg)
    class(io_cell(*)), intent(inout) :: dtv
    integer, intent(in) :: unit, v_list(:)
    character(*), intent(in) :: iotype
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    integer :: stated_length, value
    read(unit, *, iostat=iostat, iomsg=iomsg) stated_length, value
    if (iostat /= 0) return
    if (stated_length /= dtv%n) then
      iostat = 1
      iomsg = 'LEN mismatch'
      return
    end if
    dtv%value = value
    dtv%values = value
  end subroutine read_cell

  subroutine reset_values()
    shared%value = 31
    shared%values = 31
    locked%value = 19
    locked%values = 19
  end subroutine reset_values
end module pdt_io_primary

module pdt_io_other
  implicit none
  private
  type, public :: io_cell(n)
    integer, len :: n = 1
    integer :: value = 0
  contains
    procedure :: write_cell
    generic :: write(formatted) => write_cell
    procedure :: read_cell
    generic :: read(formatted) => read_cell
  end type io_cell
contains
  subroutine write_cell(dtv, unit, iotype, v_list, iostat, iomsg)
    class(io_cell(*)), intent(in) :: dtv
    integer, intent(in) :: unit, v_list(:)
    character(*), intent(in) :: iotype
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    write(unit, '(i0,1x,i0)', iostat=iostat, iomsg=iomsg) dtv%n, dtv%value + 1000
  end subroutine write_cell

  subroutine read_cell(dtv, unit, iotype, v_list, iostat, iomsg)
    class(io_cell(*)), intent(inout) :: dtv
    integer, intent(in) :: unit, v_list(:)
    character(*), intent(in) :: iotype
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    integer :: stated_length, value
    read(unit, *, iostat=iostat, iomsg=iomsg) stated_length, value
    if (iostat /= 0) return
    if (stated_length /= dtv%n) then
      iostat = 1
      iomsg = 'LEN mismatch'
      return
    end if
    dtv%value = value - 1000
  end subroutine read_cell
end module pdt_io_other
