module pdt_procedure_values
  use iso_c_binding, only: c_double
  implicit none
  private
  type, public :: proc_cell(n)
    integer, len :: n = 2
    integer :: value = 0
    procedure(compute_interface), pointer, nopass :: compute => null()
    procedure(bump_interface), pointer :: bump => null()
  end type proc_cell
  abstract interface
    real(c_double) function compute_interface(x)
      import c_double
      real(c_double), intent(in) :: x
    end function compute_interface
    subroutine bump_interface(self, amount)
      import proc_cell
      class(proc_cell(*)), intent(inout) :: self
      integer, intent(in) :: amount
    end subroutine bump_interface
  end interface
  type(proc_cell(3)), public :: shared
  type(proc_cell(2)), public, protected :: locked
  public :: configure, reset_values, clear_compute
contains
  real(c_double) function twice(x)
    real(c_double), intent(in) :: x
    twice = 2 * x
  end function twice

  subroutine add_value(self, amount)
    class(proc_cell(*)), intent(inout) :: self
    integer, intent(in) :: amount
    self%value = self%value + amount
  end subroutine add_value

  subroutine configure(obj)
    type(proc_cell(*)), intent(inout) :: obj
    obj%compute => twice
    obj%bump => add_value
  end subroutine configure

  subroutine clear_compute(obj)
    type(proc_cell(*)), intent(inout) :: obj
    nullify(obj%compute)
  end subroutine clear_compute

  subroutine reset_values()
    shared%value = 7
    shared%compute => twice
    shared%bump => add_value
    locked%value = 9
    locked%compute => twice
    locked%bump => add_value
  end subroutine reset_values
end module pdt_procedure_values

module pdt_procedure_kinds
  use iso_c_binding, only: c_float, c_double
  implicit none
  private
  integer, parameter, public :: k_float = c_float, k_double = c_double
  type, public :: kind_proc(k)
    integer, kind :: k = c_double
    real(k) :: value = 0
    procedure(compute_interface), pointer, nopass :: compute => null()
  end type kind_proc
  abstract interface
    real(c_double) function compute_interface(x)
      import c_double
      real(c_double), intent(in) :: x
    end function compute_interface
  end interface
  public :: configure_float, configure_double
contains
  real(c_double) function add_float(x)
    real(c_double), intent(in) :: x
    add_float = x + 4
  end function add_float

  real(c_double) function add_double(x)
    real(c_double), intent(in) :: x
    add_double = x + 8
  end function add_double

  subroutine configure_float(obj)
    type(kind_proc(c_float)), intent(inout) :: obj
    obj%compute => add_float
  end subroutine configure_float

  subroutine configure_double(obj)
    type(kind_proc(c_double)), intent(inout) :: obj
    obj%compute => add_double
  end subroutine configure_double
end module pdt_procedure_kinds
