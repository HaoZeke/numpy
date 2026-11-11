module alloc_args
  implicit none
contains
  subroutine mod_out(n, x)
    integer, intent(in) :: n
    real(8), allocatable, intent(out) :: x(:)
    integer :: i
    allocate(x(n))
    x = [(real(i, 8), i = 1, n)]
  end subroutine mod_out

  subroutine threadsafe_out(n, x)
    !f2py threadsafe
    integer, intent(in) :: n
    complex(8), allocatable, intent(out) :: x(:,:,:)
    allocate(x(n, n + 1, 2))
    x = (1d0, -1d0)
  end subroutine threadsafe_out

  subroutine append(x, n)
    !f2py intent(in,out) x
    integer, allocatable, intent(inout) :: x(:)
    integer, intent(in) :: n
    integer :: old(size(x))
    old = x
    deallocate(x)
    allocate(x(size(old) + n))
    x = -1
    x(1:size(old)) = old
  end subroutine append

  function mod_count(n, x) result(k)
    integer, intent(in) :: n
    integer, allocatable, intent(out) :: x(:)
    integer :: k
    allocate(x(n))
    x = 7
    k = n
  end function mod_count

  function mod_range(n) result(r)
    integer, intent(in) :: n
    integer, allocatable :: r(:)
    integer :: i
    allocate(r(n))
    r = [(i, i = 1, n)]
  end function mod_range
end module alloc_args

subroutine alloc_out(n, m, x)
  implicit none
  integer, intent(in) :: n, m
  real(8), allocatable, intent(out) :: x(:,:)
  integer :: i, j
  allocate(x(0:n, 0:m))
  do j = 0, m
    do i = 0, n
      x(i, j) = 10*i + j
    end do
  end do
end subroutine alloc_out

subroutine maybe_out(k, x)
  implicit none
  integer, intent(in) :: k
  integer, allocatable, intent(out) :: x(:)
  if (k >= 0) allocate(x(k))
  if (k > 0) x = k
end subroutine maybe_out

subroutine alloc_in(x, s)
  implicit none
  real(8), allocatable, intent(in) :: x(:,:)
  real(8), intent(out) :: s(3)
  s = -1
  if (allocated(x)) s = [sum(x), real(size(x, 1), 8), real(size(x, 2), 8)]
end subroutine alloc_in

subroutine alloc_inout(x)
  implicit none
  integer, allocatable, intent(inout) :: x(:)
  x = 2*x
end subroutine alloc_inout

subroutine alloc_grow(x)
  implicit none
  integer, allocatable, intent(inout) :: x(:)
  integer :: n
  n = size(x)
  deallocate(x)
  allocate(x(n + 1))
  x = 0
end subroutine alloc_grow

function count_out(n, x) result(k)
  implicit none
  integer, intent(in) :: n
  real, allocatable, intent(out) :: x(:)
  integer :: k
  allocate(x(n))
  x = 0.5
  k = n
end function count_out

function arange(n) result(r)
  implicit none
  integer, intent(in) :: n
  integer, allocatable :: r(:)
  integer :: i
  allocate(r(n))
  r = [(i, i = 0, n - 1)]
end function arange
