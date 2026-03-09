module char_member_mod
  implicit none

  type :: Person
    character(len=32) :: name
    integer :: age
  end type Person

end module char_member_mod
