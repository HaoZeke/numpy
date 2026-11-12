subroutine get_pid(pid)
   use iso_c_binding, only: c_int
   implicit none
   integer(c_int), intent(out) :: pid
   interface
      function getpid() bind(c, name="getpid")
         import :: c_int
         integer(c_int) :: getpid
      end function getpid
   end interface
   pid = getpid()
end subroutine get_pid
