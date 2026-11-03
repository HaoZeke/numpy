C     Every CHARACTER dummy takes a hidden length argument, including a
C     CHARACTER without a length selector, so len(s) below is only right
C     if the wrapper also passes the hidden length of c.
      subroutine charlens(c, s, n)
      character c
      character*(*) s
      integer n
cf2py intent(out) n
      n = len(s)
      if (c .ne. 'A') n = -n
      end

      subroutine chararrlens(c, s, n)
      character c(2)
      character*(*) s
      integer n
cf2py intent(out) n
      n = len(s)
      if (c(2) .ne. 'B') n = -n
      end

      function charlenfunc(c, s)
      character charlenfunc
      character c
      character*(*) s
      charlenfunc = char(len(s))
      if (c .ne. 'A') charlenfunc = c
      end
