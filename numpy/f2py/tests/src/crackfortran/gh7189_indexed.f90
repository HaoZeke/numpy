module minimal
  integer, dimension(2) :: d
  data d(1) / 1 /
  data d(2) / 2 /
end module minimal

module minimal_none
  implicit none
  integer, dimension(2) :: d
  data d(1) / 1 /
  data d(2) / 2 /
end module minimal_none

module repeated
  implicit none
  integer, dimension(2) :: d
  data d(1), d(2) / 2*3 /
end module repeated

module reserved_name
  implicit none
  integer, dimension(2) :: size
  data size(1) / 1 /
  data size(2) / 2 /
end module reserved_name
