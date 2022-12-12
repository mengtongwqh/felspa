#ifndef _FELSPA_PDE_TIME_INTEGRATION_H_
#define _FELSPA_PDE_TIME_INTEGRATION_H_

#include <deal.II/base/smartpointer.h>
#include <deal.II/base/subscriptor.h>
#include <felspa/base/exceptions.h>
#include <felspa/base/felspa_config.h>
#include <felspa/base/types.h>

#include <functional>
#include <utility>
#include <vector>

FELSPA_NAMESPACE_OPEN

template <typename NumberType>
class TempoIntegrator;

/* ************************************************** */
/**
 * \enum TempoCategory
 * Enumerate all method categories for temporal integration
 * Namely, Explicit method, Implicit method, and Mixed method.
 */
/* ************************************************** */
enum class TempoCategory
{
  undefined,
  exp,
  imp,
  mix
};

std::ostream& operator<<(std::ostream&, TempoCategory);


/* ************************************************** */
/**
 * \enum TempoMethod
 * Enumeration of all time integration methods.
 */
/* ************************************************** */
enum class TempoMethod
{
  undefined,
  rktvd1,
  rktvd2,
  rktvd3,
};

std::ostream& operator<<(std::ostream&, TempoMethod);


/* ************************************************** */
/**
 * \class TempoIntegrationSelector
 * This construct allows the \c TempoIntegrator to
 * pick the appropriate integrator in \c TempoSchemeType
 * and generate relevant code.
 * The advantage is that if a category is not implemented
 * in \c TempoSchemeType, this error will be detected
 */
/* ************************************************** */
template <typename TempoSchemeType, TempoCategory category>
struct TempoIntegrationSelector
{
  template <typename SimulatorType, typename TimeSizeType>
  static void integrate(SimulatorType& simulator, TimeSizeType time_step,
                        TimeSizeType current_time)
  {
    UNUSED_VARIABLE(simulator);
    UNUSED_VARIABLE(time_step);
    UNUSED_VARIABLE(current_time);
    THROW(ExcInternalErr());
  }
};


template <typename TempoSchemeType>
struct TempoIntegrationSelector<TempoSchemeType, TempoCategory::exp>
{
  template <typename SimulatorType, typename TimeSizeType>
  static void integrate(SimulatorType& simulator, TimeSizeType time_step,
                        TimeSizeType current_time)
  {
    TempoSchemeType::integrate_explicit(simulator, time_step, current_time);
  }
};


template <typename TempoSchemeType>
struct TempoIntegrationSelector<TempoSchemeType, TempoCategory::imp>
{
  template <typename SimulatorType, typename TimeSizeType>
  static void integrate(SimulatorType& simulator, TimeSizeType time_step,
                        TimeSizeType current_time)
  {
    TempoSchemeType::integrate_implicit(simulator, time_step, current_time);
  }
};


template <typename TempoSchemeType>
struct TempoIntegrationSelector<TempoSchemeType, TempoCategory::mix>
{
  template <typename SimulatorType, typename TimeSizeType>
  static void integrate(SimulatorType& simulator, TimeSizeType time_step,
                        TimeSizeType current_time)
  {
    TempoSchemeType::integrate_mixed(simulator, time_step, current_time);
  }
};


/* ************************************************** */
/**
 * \class TempoControl
 * Control parameters to be passed to
 */
/* ************************************************** */
template <typename NumberType>
class TempoControl : public dealii::Subscriptor
{
  friend class TempoIntegrator<NumberType>;

 public:
  using value_type = NumberType;

  /**
   * Default constructor.
   */
  TempoControl() = default;


  /**
   * Constructor.
   */
  TempoControl(TempoMethod method, TempoCategory category, value_type cfl_min_,
               value_type cfl_max_);


  /**
   * Virtual destructor.
   */
  virtual ~TempoControl() = default;


  /**
   * Release memory occupied by the vectors
   */
  void clear();


  /**
   * Return a pair indicating the temporal method name and its category.
   */
  std::pair<TempoMethod, TempoCategory> query_method() const;


  /**
   * Set the CFL number
   */
  void set_cfl(value_type cfl_min_, value_type cfl_max_);


  /**
   * Set automatic adjustment
   */
  void set_auto_adjust(bool flag) { time_step_auto_adjust = flag; }


  /**
   * Define the method used for temporal integration.
   */
  void use_method(TempoMethod method, TempoCategory category);


  /**
   * Return a pair of lower and upper limit of CFL number.
   */
  std::pair<value_type, value_type> get_cfl() const;


  /**
   * Return the status of time step auto adjustment
   */
  bool defined_auto_adjust() const { return time_step_auto_adjust; }


  /**
   * Test if CFL is properly initialized
   */
  bool is_initialized() const;


  /**
   * Scale the CFL lower and upper limit by a constant coefficient.
   */
  void scale_cfl(value_type coeff);


 protected:
  /**
   * Flag if time step auto adjustment should be enabled.
   */
  bool time_step_auto_adjust = true;


  /**
   * Time integration scheme.
   */
  TempoMethod method = TempoMethod::undefined;


  /**
   * Implicit, explicit or mixed.
   */
  TempoCategory category = TempoCategory::undefined;


  /**
   * CFL upper bound.
   */
  value_type cfl_min = -1.0;


  /**
   * CFL lower bound.
   */
  value_type cfl_max = -1.0;

  /**
   * Number of substeps taken for major time step.
   */
  std::vector<types::SizeType> n_substeps;


  /**
   * Time step size of each time step.
   */
  std::vector<value_type> time_steps;
};


/* ************************************************** */
/**
 * Workhorse class for executing time integration
 * and optimizing time step according to CFL-number
 * if a CFL estimator is given
 * We want a CFL estimator of the following signature:
 * cfl = CFL_estimater(current_time)
 *
 * We need three interfaces from the simulator side:
 * \c explicit/implicit_time_derivative
 * \c update_solution which update both \c solution and \c solution_time
 * \c set_time_temporal_passive_members
 */
/* ************************************************** */
template <typename NumberType>
class TempoIntegrator : public dealii::Subscriptor
{
  /**
   * Grant friendship to \c TempoControl so that it can call
   * some protected methods. e.g. \c set_cfl() to update CFL limits
   */
  friend class TempoControl<NumberType>;

 public:
  /** Integral type for time step counts */
  using size_type = types::SizeType;

  /** Float type for time step sizes */
  using value_type = NumberType;

  /** CFL estimator function call signature */
  using cfl_estimate_helper_type = std::function<value_type(value_type)>;


  /** \name Basic Object Behavior */
  //@{
  /**
   * Constructor
   */
  TempoIntegrator();

  /**
   * Constructor taking a \c TempoControl
   */
  TempoIntegrator(const std::shared_ptr<TempoControl<value_type>>&);

  /**
   * Copy constructor
   */
  TempoIntegrator(const TempoIntegrator<NumberType>&);

  /**
   * Copy assignment
   */
  TempoIntegrator<NumberType>& operator=(const TempoIntegrator<NumberType>&);

  /**
   * Destructor
   */
  virtual ~TempoIntegrator() = default;
  //@}


  /**
   * \name Getting/Setting Object Properties
   */
  //@{
  /**
   * obtain current cumulate time
   */
  value_type get_time() const { return time; }

  /**
   * Obtain current step count
   */
  size_type get_step_count() const { return step_count; }

  /**
   * Attach control structure and populate control parameters.
   * Also, add this object to the signal slots of the \c TempoControl
   * object to receive updates whenever there is a change in control paramters.
   */
  void attach_control(const std::shared_ptr<TempoControl<value_type>>& ctrl);

  /**
   * Setting the CFL limits in the \c TempoControl object.
   * This is only valid if the \c TempoControl object is generated by
   * \c TempoIntegrator internally, or an exception will be thrown.
   */
  void set_cfl(value_type cfl_min, value_type cfl_max) const;

  /**
   * Perform initialization on the TempoIntegrator.
   * This include zeroing out the #time, #step_count and #sub_step count.
   * Also clear the time step vector in the control structure.
   */
  void initialize();
  //@}


  /// \name Interface for Advancing Temporal Integration
  //@{
  /**
   * Simply advance the solution by given time_step
   */
  template <typename TempoSchemeType, TempoCategory category,
            typename SimulatorType>
  value_type advance_time_step(SimulatorType& simulator, value_type time_step);

  /**
   * If CFL limit is already set and CFL estimate is present,
   * then auto adjust time step to respect the CFL limit by taking multiple
   * time steps if necessary
   * If CFL estimate is not set, throw an error
   * If CFL estimate is defaulted, simply advance the solution by the time
   * step specified
   * @param[in]     time_derivative_fcn Function pointer to evolution function
   * @param[in,out] solution            The solution to be advanced
   * @param[in]     time_step           Size of time step
   * @param[in]     cfl_estimate        CFL estimation from the solver
   */
  template <typename TempoSchemeType, TempoCategory category,
            typename SimulatorType>
  std::vector<value_type> advance_time_step(
    SimulatorType& simulator, value_type time_step,
    const cfl_estimate_helper_type& cfl_estimator);
  //@}


  /// \name Exceptions
  //@{
  /**
   * Make sure that CFL upper limit > CFL lower limit
   */
  DECL_EXCEPT_2(ExcIllegalCflLimits,
                "Maximum CFL = " << arg1 << "is not greater than Minimum CFL = "
                                 << arg2,
                value_type, value_type);

  /**
   * If CFL limits are not set, time step auto-adjustment is point-less '
   */
  DECL_EXCEPT_2(ExcCflNotSet,
                "CFL is not properly set for time-step auto adjustment. "
                  << "The CFL-min = " << arg1 << ", The CFL-max = " << arg2,
                value_type, value_type);

  /**
   * Simulator is not synchronized after time step
   */
  DECL_EXCEPT_0(
    ExcSimulatorNotSync,
    "Simulator fails synchronization check after time step completed.");
  //@}


 protected:
  /**
   * Adjust time step so as to
   * respect CFL upper/lower limits
   */
  value_type adjust_time_step_to_cfl_limit(value_type time_step_given,
                                           value_type max_velo_diam) const;


  /**
   * Cumulate time up to the current time
   */
  value_type time;


  /**
   * Cumulate major time steps taken
   */
  size_type step_count;


  /**
   * Cumulate major + minor time steps taken
   */
  size_type substep_count;


  /**
   * Pointer to control parameters
   */
  std::shared_ptr<TempoControl<NumberType>> ptr_control;
};


/* ************************************************** */
/**
 * TemporalSchemes
 * These class types are to be substituted into the \c TempoSchemeType
 * template parameter in \c TempoIntegrator. We choose to implement
 * as structs and implement integration algorithms as static functions
 * so that instantiations won't happen and therefore improve efficiency.
 *
 * To utilize the time-stepping mechanism, we require the simulator to. \n
 * 1) implement one of the following methods:
 * \c explicit_time_derivative. The discretization is of:
 * \f$ \phi^{n+1} = \Delta{t}^n M^{-1}F(t, \phi^n) + \phi^{n} \f$\n
 * and hence explicit time derivative is defined as
 * \f$ \mathcal{D}^{exp}(t, \phi) = M^{-1}F(t, \phi) \f$
 * \c implicit_time_derivative. The discretization is of:
 * \f$ M\frac{\phi^{n+1} - \phi^n}{\Delta{t}^n} = F(\phi^{n+1}, t^{n+1}) \f$\n
 * hence the implicit time derivative is defined as
 * \f$ \mathcal{D}^{imp}(t,\phi,\Delta{t})
 * 		= I - \Delta{t}\frac{\partial F(\phi,t)}{\partial \phi} \f$
 *
 * \c mixed_time_derivative, which requires both explicit and
 * implicit time derivatives
 *
 * 2) We also require the simulator to implement \c synchronize \n
 * 3) allow the \c TemporalScheme to access the \c solution vector \n
 */
/* ************************************************** */

/* ************************************************** */
/**
 * \class RungeKuttaTVD
 * \tparam VectorType
 * Runge-Kutta total variance diminishing (TVD)
 * family of explicit methods.
 * \TODO in fact, for each Runge-Kutta substep,
 * the solver must go through the system_setup , assemble and solve cycle
 * or we are "over-linearizing" throughout the whole time step
 */
/* ************************************************** */
template <int order>
struct RungeKuttaTVD
{};

/**
 * Specialization for 1st order Runge-Kutta TVD method
 */
template <>
struct RungeKuttaTVD<1>
{
  /** Typedef for time step type */
  using value_type = types::TimeStepType;

  /** Override explicit integrator */
  template <typename SimulatorType>
  static void integrate_explicit(SimulatorType& simulator,
                                 const value_type time_step,
                                 const value_type current_time);
};


/**
 * Specialization for 2nd order Runge-Kutta TVD method
 */
template <>
struct RungeKuttaTVD<2>
{
  /** Typedef for time step type */
  using value_type = types::TimeStepType;

  /** Override explicit integrator */
  template <typename SimulatorType>
  static void integrate_explicit(SimulatorType& simulator,
                                 const value_type time_step,
                                 const value_type current_time);
};


/**
 * Specialization for 3rd order Runge-Kutta TVD method.
 * \note For this time integration scheme to work properly,
 * all temporal passive members of the simulator
 * must be backward synchronizable.
 */
template <>
struct RungeKuttaTVD<3>
{
  /** Typedef for time step type */
  using value_type = types::TimeStepType;

  /** Override explicit integrator */
  template <typename SimulatorType>
  static void integrate_explicit(SimulatorType& simulator,
                                 const value_type time_step,
                                 const value_type current_time);
};

FELSPA_NAMESPACE_CLOSE

/* -------- Template Implementations ----------*/
#include "src/time_integration.implement.h"
/* -------------------------------------------*/

#endif  //_FELSPA_PDE_TIME_INTEGRATION_H_
