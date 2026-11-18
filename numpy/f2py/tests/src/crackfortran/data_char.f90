module data_char
  implicit none
  character(len=8) :: comma_a, comma_b
  character(len=8) :: slash_sq, slash_dq
  character(len=12) :: apostrophe
  integer :: i, j, k
  data comma_a, comma_b /'a,b', 'c,d'/
  data slash_sq /'a/b'/
  data slash_dq /"a/b"/
  data apostrophe /'it''s, ok'/
  data i, j /1, 2/, k /3/
end module data_char
