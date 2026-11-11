      SUBROUTINE STRING_OUT(FUN,S)
      EXTERNAL FUN
      CHARACTER*5 S
Cf2py intent(out) s
      CALL FUN(S)
      END
