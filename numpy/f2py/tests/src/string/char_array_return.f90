function r1_explicit(o) result(c)
  character(len=4), intent(in) :: o(2)
  character(len=4) :: c(2)
  c = o
end function

function r1_assumed(o) result(c)
  character(len=*), intent(in) :: o(2)
  character(len=len(o)) :: c(2)
  c = o
end function

function r2_explicit(o) result(c)
  character(len=3), intent(in) :: o(2, 3)
  character(len=3) :: c(2, 3)
  c = o
end function

function r2_assumed(o) result(c)
  character(len=*), intent(in) :: o(2, 3)
  character(len=len(o)) :: c(2, 3)
  c = o
end function
