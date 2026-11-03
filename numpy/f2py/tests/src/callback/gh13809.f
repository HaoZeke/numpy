C     The caller passes a hidden length for every CHARACTER dummy of the
C     call-back, after the other arguments and in argument order.
      subroutine string_array_arg(fun, a, s, n)
      external fun
      integer fun
      character*3 a(2)
      character*(*) s
      integer n
      n = fun(a, s)
      end

      subroutine character_arg(fun, s, flag)
      external fun
      character*(*) s
      character flag
      call fun(flag, s)
      end

      subroutine string_inout_arg(fun, s)
      external fun
      character*(*) s
      call fun(s)
      end

      subroutine string_function(fun, s)
      external fun
      character*5 fun
      character*5 s
      s = fun()
      end
