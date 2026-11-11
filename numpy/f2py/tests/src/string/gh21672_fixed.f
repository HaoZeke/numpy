C gh-21672: character name(10)*8 is 10 elements of length 8 (fixed form)
      subroutine gh21672_fixed_entity(inp)
      character inp(10)*8
      character gamma(10)*8
      common /gh21672ent/ gamma
Cf2py intent(in) inp
      integer i
      do 10 i = 1, 10
         gamma(i) = inp(i)
   10 continue
      end
      subroutine gh21672_fixed_star(inp)
      character*8 inp(10)
      character*8 gamma(10)
      common /gh21672star/ gamma
Cf2py intent(in) inp
      integer i
      do 20 i = 1, 10
         gamma(i) = inp(i)
   20 continue
      end
      subroutine gh21672_fixed_entity_out(gamma)
      character gamma(10)*8
Cf2py intent(out) gamma
      integer i
      do 30 i = 1, 10
         gamma(i) = 'abcdefgh'
   30 continue
      end
