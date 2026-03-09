module array_of_types_opaque
  implicit none

  type :: Atom
    real :: x, y, z
    integer :: atomic_num
  end type Atom

  type :: Molecule
    type(Atom) :: atoms(3)
    integer :: natoms
  end type Molecule

end module array_of_types_opaque
