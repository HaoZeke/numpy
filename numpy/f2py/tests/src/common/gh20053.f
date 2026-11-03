      subroutine set_hidden(a)
      real*8 a, b
      common /hidden/ b
      b = a
      end

      subroutine get_hidden(a)
      real*8 a, b
      common /hidden/ b
      a = b
      end
