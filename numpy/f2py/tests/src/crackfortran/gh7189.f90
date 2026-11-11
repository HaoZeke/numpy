! gh-7189: an array element in DATA is not its own variable.
module elemdata
  integer, dimension(2) :: d
  data d(1) / 1 /
  data d(2) / 2 /
end module elemdata
