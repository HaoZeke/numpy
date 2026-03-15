module deferred_char_mod
  implicit none

  ! Deferred-length allocatable character members
  ! F2018 7.4.4.2 paragraph 3: length type parameter is deferred
  ! when the colon is used with the ALLOCATABLE attribute

  type :: NamedItem
    integer :: item_id
    character(:), allocatable :: label
  end type NamedItem

  type :: Document
    character(:), allocatable :: title
    character(:), allocatable :: author
    integer :: page_count
  end type Document

end module deferred_char_mod
