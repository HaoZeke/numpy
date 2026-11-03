module gh28648
  implicit none
contains
  subroutine run_passive_model(input_stream, output_stream, n)
    integer, intent(in) :: n
    double precision, intent(in) :: input_stream(n)
    double precision, intent(out) :: output_stream(n)
    output_stream = 20*input_stream
  end subroutine run_passive_model

#ifdef NEW_MODEL_PARM
  subroutine run_active_model(input_stream, output_stream, n)
    integer, intent(in) :: n
    double precision, intent(in) :: input_stream(n)
    double precision, intent(out) :: output_stream(n)
    output_stream = 10*input_stream
  end subroutine run_active_model
#endif
end module gh28648
