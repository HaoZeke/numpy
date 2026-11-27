      subroutine see_in(x, flag)
        character(len=3) :: x(2)
        integer :: flag
        flag = 0
        if (x(1) == 'a') flag = flag + 1
        if (x(1) == 'a  ') flag = flag + 2
        if (x(2) == 'b ') flag = flag + 4
      end subroutine see_in

      subroutine see_inout(x, flag)
        character(len=3) :: x(2)
        integer :: flag
        flag = 0
        if (x(1) == 'a') flag = flag + 1
        if (x(1) == 'a  ') flag = flag + 2
        if (x(2) == 'b ') flag = flag + 4
        x(1) = 'q'
      end subroutine see_inout

      subroutine see_out(x, flag)
        character(len=3) :: x(2)
        integer :: flag
        flag = 0
        x(1) = 'q'
        x(2) = 'zz'
      end subroutine see_out
