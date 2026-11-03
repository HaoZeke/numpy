      subroutine setn(m)
      integer m, n
      common /blk/ n
      n = m
      end

      subroutine getn(m)
      integer m, n
      common /blk/ n
      m = n
      end
