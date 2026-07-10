      subroutine get_fact(k, j_size, j_array)
!f2py integer, intent(in) :: k, j_size
      integer, dimension(j_size) :: j_array
!f2py intent(in) :: j_array
      integer :: nine
      nine = (k + j_array(2))/100
      end