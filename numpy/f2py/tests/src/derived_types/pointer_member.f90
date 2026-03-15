module pointer_member_mod
  implicit none

  ! Pointer components for testing read-only access from Python
  ! F2018 7.5.4.6 "Pointer components"

  ! Target data that pointer members will reference
  ! F2018 8.5.17: TARGET attribute
  double precision, target :: shared_data(5)
  double precision, target :: shared_matrix(3,3)

  type :: DataView
    integer :: label
    double precision, pointer :: view(:)
  end type DataView

  type :: MatrixView
    integer :: tag
    double precision, pointer :: grid(:,:)
  end type MatrixView

contains

  subroutine init_shared_data()
    shared_data(1) = 10.0d0
    shared_data(2) = 20.0d0
    shared_data(3) = 30.0d0
    shared_data(4) = 40.0d0
    shared_data(5) = 50.0d0
  end subroutine init_shared_data

  subroutine init_shared_matrix()
    ! Fill shared_matrix(row,col) = row*10 + col
    integer :: row_idx, col_idx
    do col_idx = 1, 3
      do row_idx = 1, 3
        shared_matrix(row_idx, col_idx) = row_idx * 10 + col_idx
      end do
    end do
  end subroutine init_shared_matrix

  subroutine attach_view(dv)
    ! Associate the pointer with the module-level target array
    ! F2018 10.2.2: Pointer association via pointer assignment
    type(DataView), intent(inout) :: dv
    call init_shared_data()
    dv%view => shared_data
  end subroutine attach_view

  subroutine detach_view(dv)
    ! Nullify the pointer (F2018 10.2.2.3)
    type(DataView), intent(inout) :: dv
    nullify(dv%view)
  end subroutine detach_view

  subroutine attach_matrix_view(mv)
    ! Associate with the 2D target
    type(MatrixView), intent(inout) :: mv
    call init_shared_matrix()
    mv%grid => shared_matrix
  end subroutine attach_matrix_view

end module pointer_member_mod
