module deferred_tbp_mod
  implicit none

  ! Abstract type with two deferred TBPs (F2018 7.5.5, R752)
  type, abstract :: Integrator
    integer :: npoints = 1
  contains
    procedure(integrate_iface), deferred :: integrate
    procedure(get_order_iface), deferred :: get_order
  end type Integrator

  abstract interface
    function integrate_iface(self, a, b) result(res)
      import :: Integrator
      class(Integrator), intent(in) :: self
      double precision, intent(in) :: a, b
      double precision :: res
    end function
    function get_order_iface(self) result(res)
      import :: Integrator
      class(Integrator), intent(in) :: self
      integer :: res
    end function
  end interface

  ! Concrete child implementing both deferred TBPs
  type, extends(Integrator) :: MidpointRule
  contains
    procedure :: integrate => midpoint_integrate
    procedure :: get_order => midpoint_order
  end type MidpointRule

  ! Deeper child overriding integrate, inheriting get_order
  type, extends(MidpointRule) :: CorrectedMidpoint
  contains
    procedure :: integrate => corrected_integrate
  end type CorrectedMidpoint

contains
  function midpoint_integrate(self, a, b) result(res)
    class(MidpointRule), intent(in) :: self
    double precision, intent(in) :: a, b
    double precision :: res
    double precision :: mid
    mid = 0.5d0 * (a + b)
    res = (b - a) * mid * mid  ! integrates x^2
  end function

  function midpoint_order(self) result(res)
    class(MidpointRule), intent(in) :: self
    integer :: res
    res = 2
  end function

  function corrected_integrate(self, a, b) result(res)
    class(CorrectedMidpoint), intent(in) :: self
    double precision, intent(in) :: a, b
    double precision :: res
    double precision :: mid, h
    h = b - a
    mid = 0.5d0 * (a + b)
    ! Midpoint + correction term
    res = h * mid * mid + h**3 / 24.0d0
  end function
end module deferred_tbp_mod
