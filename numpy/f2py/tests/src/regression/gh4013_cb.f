      subroutine applyopt(fun, a)
      integer a
cf2py intent(out) a
      external fun
      optional fun
      if (present(fun)) then
        call fun(a)
      else
        a = 7
      endif
      end
