      subroutine get_fact(k, j_size, j_array, nine)
      integer, intent(in) :: k, j_size
      integer, dimension(j_size), intent(in) :: j_array
      integer, intent(out) :: nine
      nine = (k + j_array(2))/100
      end
