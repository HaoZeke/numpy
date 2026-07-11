      subroutine chararg(c, n)
      character*1 c
      integer n
Cf2py intent(out) n
      if (c .eq. 'A') then
         n = 1
      else
         n = 2
      endif
      end
