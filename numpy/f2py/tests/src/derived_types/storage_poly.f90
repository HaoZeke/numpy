module storage_poly
  implicit none

  ! class() allocatable and pointer dummies (F2018 15.5.2.5).
  ! The nonpolymorphic routines live in storage_args so a compiler
  ! that cannot lower class can still build them.

  type :: item
    integer :: n = 0
  end type item

  type, extends(item) :: tagged
    integer :: tag = 0
  end type tagged

contains

  function kind_of(x) result(k)
    class(item), allocatable, intent(in) :: x
    integer :: k
    k = 0
    if (allocated(x)) then
      select type (x)
      type is (item)
        k = 1
      type is (tagged)
        k = 2
      end select
    end if
  end function kind_of

  subroutine retag(x)
    class(item), pointer, intent(in) :: x
    select type (x)
    type is (tagged)
      x%tag = x%tag + 1
    end select
  end subroutine retag

end module storage_poly
