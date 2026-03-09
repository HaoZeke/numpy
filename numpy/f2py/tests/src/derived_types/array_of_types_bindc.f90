module array_of_types_bindc
  use iso_c_binding
  implicit none

  type, bind(c) :: Atom
    real(c_double) :: x, y, z
    integer(c_int) :: atomic_num
  end type Atom

  type, bind(c) :: Molecule
    type(Atom) :: atoms(3)
    integer(c_int) :: natoms
  end type Molecule

end module array_of_types_bindc
