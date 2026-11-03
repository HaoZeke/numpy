module long_repr
  implicit none

  ! a repr longer than any fixed buffer of a few hundred bytes
  type :: record
    double precision :: v01 = -1.25d-300, v02 = -1.25d-300, v03 = -1.25d-300, v04 = -1.25d-300
    double precision :: v05 = -1.25d-300, v06 = -1.25d-300, v07 = -1.25d-300, v08 = -1.25d-300
    double precision :: v09 = -1.25d-300, v10 = -1.25d-300, v11 = -1.25d-300, v12 = -1.25d-300
    double precision :: v13 = -1.25d-300, v14 = -1.25d-300, v15 = -1.25d-300, v16 = -1.25d-300
    double precision :: v17 = -1.25d-300, v18 = -1.25d-300, v19 = -1.25d-300, v20 = -1.25d-300
    double precision :: v21 = -1.25d-300, v22 = -1.25d-300, v23 = -1.25d-300, v24 = -1.25d-300
  end type record

end module long_repr
