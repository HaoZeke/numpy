c     gh-24284: typed and untyped externals passed down, then called
      subroutine diva(divaf, divao, n)
      implicit none
      external divaf, divao
      real*8 divaf
      integer n
      call divam(divaf, divao, n)
      end

      subroutine divam(f, o, n)
      implicit none
      external f, o
      real*8 f
      integer n
      real*8 y
      y = f(n)
      call o(n, y)
      end
