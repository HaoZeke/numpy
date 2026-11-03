C FILE: ARGORDER.F
      SUBROUTINE TALLY(N, A, LIMIT, M, TOTAL)
C
C     COUNT AND ADD UP THE ELEMENTS OF A THAT ARE GREATER THAN LIMIT
C
      INTEGER N, M, I
      REAL*8 A(N), LIMIT, TOTAL
Cf2py intent(out) m, total
      M = 0
      TOTAL = 0D0
      DO I=1,N
         IF (A(I).GT.LIMIT) THEN
            M = M + 1
            TOTAL = TOTAL + A(I)
         ENDIF
      ENDDO
      END

      SUBROUTINE TALLY2(N, A, LIMIT, M, TOTAL)
C
C     THE SAME AS TALLY, BUT N KEEPS ITS PLACE IN THE ARGUMENT LIST
C
      INTEGER N, M
      REAL*8 A(N), LIMIT, TOTAL
Cf2py required n
Cf2py intent(out) m, total
      CALL TALLY(N, A, LIMIT, M, TOTAL)
      END
C END OF FILE ARGORDER.F
