module initial_state
  implicit none

  type :: inner
    integer :: count = 7
    double precision :: weight
    double precision, pointer :: link(:)
  end type inner

  type :: holder
    integer :: id
    type(inner) :: one
    type(inner) :: many(2)
    double precision :: samples(3)
    double precision, pointer :: view(:)
  end type holder

contains

  subroutine inspect(h, counts, total, linked)
    type(holder), intent(in) :: h
    integer, intent(out) :: counts(3)
    double precision, intent(out) :: total
    integer, intent(out) :: linked
    integer :: i
    counts = [h%one%count, h%many(1)%count, h%many(2)%count]
    total = h%one%weight + sum(h%many%weight) + sum(h%samples)
    linked = 0
    if (associated(h%view)) linked = linked + 1
    if (associated(h%one%link)) linked = linked + 1
    do i = 1, size(h%many)
      if (associated(h%many(i)%link)) linked = linked + 1
    end do
  end subroutine inspect

end module initial_state
