module model
  implicit none

contains

  subroutine run_passive_model(input_stream, output_stream, n)
    integer, intent(in) :: n
    real, intent(in) :: input_stream(n)
    real, intent(out) :: output_stream(n)
    output_stream = 20*input_stream
  end subroutine run_passive_model

#ifdef NEW_MODEL_PARM
  subroutine run_active_model(input_stream, output_stream)
    real, intent(in) :: input_stream
    real, intent(out) :: output_stream
    output_stream = 10*input_stream
  end subroutine run_active_model
#endif

end module model
