module f2py_parameter_storage_0
  implicit none
  integer, parameter :: n = 9
end module f2py_parameter_storage_0

module parameter_signature_consumer
  use f2py_parameter_storage_0, only: n
  implicit none
  integer, parameter :: answer = n
contains
  integer function read_answer() result(value)
    value = answer
  end function read_answer
end module parameter_signature_consumer
