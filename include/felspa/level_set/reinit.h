#ifndef _FELSPA_LEVEL_SET_REINIT_H_
#define _FELSPA_LEVEL_SET_REINIT_H_

#include <felspa/base/function.h>
#include <felspa/base/io.h>
#include <felspa/pde/hamilton_jacobi.h>

#include <memory>

FELSPA_NAMESPACE_OPEN

/* -------------------------------------------*/
namespace dg
/* -------------------------------------------*/
{
template <int dim, typename NumberType>
class AdvectSimulator;

template <int dim, typename NumberType>
class ReinitControl;

template <int dim, typename NumberType>
class ReinitSimulatorLDG;

template <int dim, typename NumberType = types::DoubleType>
using ReinitSimulator = ReinitSimulatorLDG<dim, NumberType>;

/* ************************************************** */
/**
 * The reinitialization operator. Defined as:
 * \f[ H_{reinit}(\phi) = sgn(\phi_0)(|\nabla{\phi}| - 1) \f]
 * where \f$sgn(\phi_0)\f$ is the sign function.
 */
/* ************************************************** */
template <int dim, typename NumberType = types::DoubleType>
class ReinitOperator : public HJOperator<dim, NumberType>
{
 public:
  using value_type = NumberType;
  using base_type = HJOperator<dim, NumberType>;
  using vector_type = typename dg::HJTypedefs<dim, NumberType>::vector_type;


  /** Constructor */
  ReinitOperator() = default;


  /**
   * This function will initialize the smoothed signum function values.
   * These values will later be used for computing numerical Hamilton-Jacobi
   * reinitialization operator. This function also cache the smoothed sign
   * function values for use in later computation.
   */
  void initialize(const HJSimulator<dim, value_type>& hj_simulator) override;


  /**
   * @brief Test if the operator is properly initialized
   */
  bool is_initialized() const override;


  /**
   * @pre The \c dealii::FEValues must be reinitialized to the current cell
   * @return std::vector<value_type>  a vector containing the HJ value at
   * quadrature points of this cell
   */
  void cell_values(const dealii::FEValuesBase<dim>& feval,
                   std::vector<value_type>& hj_values) const override;


  /**
   * @brief Wave velocities at quadrature points of the cell
   * @pre \c dealii::FEValues must be reinited to the current cell
   * @return std::vector<dealii::Tensor<1, dim, value_type>>
   */
  void cell_velocities(const dealii::FEValuesBase<dim>& feval,
                       std::vector<dealii::Tensor<1, dim, value_type>>&
                         hj_velocities) const override;


  /**
   * Export the initialized signum values to file.
   * \todo also consider export the gradients.
   */
  void export_signum_values(ExportFile&,
                            const dealii::DoFHandler<dim>& dofh) const;


 private:
  /**
   * @brief Helper function to compute smooth signum function
   */
  value_type smooth_signum(
    value_type phi0,
    const dealii::Tensor<1, dim, value_type>& left_grad,
    const dealii::Tensor<1, dim, value_type>& right_grad) const;

  /**
   * @brief Helper function to compute smooth signum function
   */
  value_type smooth_signum(value_type phi0, value_type grad_phi0_norm) const;

  /**
   * Compute numerical Hamilton-Jacobi
   * for each entry of smoothed sign function,
   * left local gradient and right local gradient.
   */
  value_type operator()(const value_type smooth_sign,
                        const dealii::Tensor<1, dim, value_type>& lgrad,
                        const dealii::Tensor<1, dim, value_type>& rgrad) const;

  value_type operator()(
    const value_type smooth_sign,
    const dealii::Tensor<1, dim, value_type>& normal,
    const dealii::Tensor<1, dim, value_type>& left_grad,
    const dealii::Tensor<1, dim, value_type>& right_grad) const;


  /**
   * The characteristic speed of the reinitialization operator. Computed as
   * \f$ |sgn_\epsilon(\phi_0)| \f$, which is the abstract value of the
   * smoothed signum function.
   */
  dealii::Tensor<1, dim, value_type> characteristic_velocity(
    value_type signum,
    const dealii::Tensor<1, dim, value_type>& lgrad,
    const dealii::Tensor<1, dim, value_type>& rgrad) const;


  /**
   * Cached (smoothed) sign values of the initial level set.
   */
  // std::map<dealii::TriaIterator<dealii::CellAccessor<dim>>,
  //          std::vector<value_type>>
  //   cached_signum_values;
  const vector_type* ptr_initial_solution_vector = nullptr;


  /**
   * @brief Characteristic cell diameter of the mesh.
   * Taken as the minimum diameter of the mesh.
   */
  value_type cell_diameter;
};


/* ************************************************** */
/**
 * Solving the level set reinitialization problem.
 */
/* ************************************************** */

template <int dim, typename NumberType = types::DoubleType>
class ReinitSimulatorLDG : public HJSimulator<dim, NumberType>
{
 public:
  struct AdditionalControl;

  using value_type = NumberType;
  using base_type = HJSimulator<dim, NumberType>;
  using size_type =
    typename HJSimulator<dim, NumberType>::vector_type::size_type;
  using typename base_type::time_step_type;
  using typename base_type::vector_type;
  using fe_simulator_type = typename base_type::base_type;
  using control_type = ReinitControl<dim, value_type>;


  /**
   * Constructor
   */
  ReinitSimulatorLDG(Mesh<dim, value_type>& triag, unsigned int fe_degree,
                     const std::string& label = "LDGReinit");


  /**
   * Use the same \c FETypes, \c DofHandler,
   * \c LinearSystem and \c Solution as in \c AdvectionSimulator.
   * We have this because in the case of level set, the reinit solver
   * and the advection can share the same FE configuration.
   */
  explicit ReinitSimulatorLDG(const fe_simulator_type&);


  /**
   * Attach the control parameters to the ReinitSimulatorLDG.
   */
  void attach_control(const std::shared_ptr<ReinitControl<dim, value_type>>&);


  /**
   * Initialize the solution vector by a ScalarFunction.
   */
  void initialize(const ScalarFunction<dim, value_type>& initial_condition,
                  bool use_independent_solution = false);


  /**
   * Directly work on the \c SolutionVector of another simulator.
   */
  void initialize(
    const TimedSolutionVector<vector_type, time_step_type>& initial_condition);


  /**
   * Import \c advance_time definition from base class (HJ-simulator).
   */
  using base_type::advance_time;


  /**
   * Compute a time step from the CFL estimator and take the time step.
   * Return the size of the time step.
   */
  time_step_type advance_time();


  /**
   * Run iteration till convergence.
   */
  size_type run_iteration();

  /**
   * Get const reference to the initial solution vector.
   */
  const vector_type& get_initial_solution_vector() const;

  /**
   *  Apply WENO limiter to high curvature cellls
   */
  void apply_curvature_limiting();


  /** \name Exceptions */
  //@{
  DECL_EXCEPT_2(ExcUnconverged,
                "Solution has not reached expected convergence criteria "
                  << arg1 << '/' << arg2
                  << " after all iterations completed.\n",
                value_type, value_type);

  DECL_EXCEPT_1(ExcNotEnoughStepsForWidth,
                "Maximum iteration reached but the reinitialization has not "
                "reached the desired width "
                  << arg1 << '\n',
                value_type);

  DECL_EXCEPT_0(ExcNoCellsInBand, "There are no cells in the interface band");
  //@}


 private:
  value_type gradient_deviation(const vector_type& ls_values,
                                bool global_solve) const;


  size_type run_iteration_global(AdditionalControl&);


  size_type run_iteration_fixed_width(AdditionalControl&);


  /**
   * make a copy of the initial solution
   */
  vector_type initial_solution_vector;
};


/* ************************************************** */
/**
 * Additional control parameters for reinit simulator.
 */
/* ************************************************** */
template <int dim, typename NumberType>
struct ReinitSimulatorLDG<dim, NumberType>::AdditionalControl
{
  using value_type = NumberType;
  using size_type = typename ReinitSimulatorLDG<dim, NumberType>::size_type;
  /**
   * Maximum number of iterations (or, pseudo time steps)
   * the simulator will execute for.
   * Beyond which the solver will abort.
   */
  size_type max_iter = 1000;


  /**
   * If we are computing a global solve, iteration will stop after the
   * tolerance is reached.
   */
  value_type tolerance = 1.0e-4;


  /**
   * Maximum width of the band. Computed as band_width_coeff * minimum cell
   * diameter of the whole domain.
   */
  value_type band_width_coeff = 4.0;


  /**
   * \c true if reinit computation will be carried out over the whole domain,
   * \c false if reinit is carried out only in a band of width
   * \c band_width_coeff *  \c min_cell_diameter. If we are in the latter
   * case, \c tolerance will be ignored and the simulator will not check for
   * convergence.
   */
  bool global_solve = false;
};


/* ************************************************** */
/**
 * Control parameters for \c ReinitSimulatorLDG
 */
/* ************************************************** */
template <int dim, typename NumberType = types::DoubleType>
class ReinitControl : public HJControl<dim, NumberType>
{
 public:
  /**
   * @brief Construct a new Reinit Control object
   */
  ReinitControl(TempoMethod method = TempoMethod::rktvd1,
                TempoCategory category = TempoCategory::exp)
    : HJControl<dim, NumberType>(method, category)
  {
    this->ptr_tempo->set_cfl(0.1, 0.2);
  }


  /**
   * When the control is attached to ReinitSimulatorLDG,
   * these data will be copied into the same struct in the
   * ReinitSimulatorLDG.
   */
  typename ReinitSimulatorLDG<dim, NumberType>::AdditionalControl
    additional_control;
};


#ifdef FELSPA_BETTER_REINIT

/* ************************************************** */
/**
 * @brief Interface preserving reinitialization solver
 * @tparam dim
 * @tparam NumberType
 */
/* ************************************************** */
template <int dim, typename NumberType = types::DoubleType>
class IPReinitSimulator
  : FESimulator<dim, FEDGType<dim>, DGLinearSystem<dim, NumberType>,
                TempoIntegrator<NumberType>>
{
 public:
  using value_type = NumberType;

  using AdditionalControl = ReinitSimulator<dim, NumberType>::AdditionalControl;

  using base_type =
    FESimulator<dim, FEDGType<dim>, DGLinearSystem<dim, NumberType>,
                TempoIntegrator<NumberType>>;

  IPReinitSimulator(Mesh<dim, value_type>& triag, unsigned int fe_degree,
                    const std::string& label = "IPReinit");


  void attach_control(const std::shared_ptr<ReinitControl<dim, value_type>>&);

  void initialize(
    const TimedSolutionVector<vector_type, time_step_type>& initial_condition);


  dealii::Tensor<1, dim, NumberType> hj_face_operator(
    const dealii::Tensor<1, dim, NumberType>& normal,
    const dealii::Tensor<1, dim, NumberType>& grad_int,
    const dealii::Tensor<1, dim, NumberType>& grad_ext,
    NumberType max_penalty_coeff) const;

  NumberType hj_cell_operator(
    const NumberType phi0,
    const dealii::Tensor<1, dim, NumberType>& grad_phi0,
    const dealii::Tensor<1, dim, NumberType>& grad) const;


  /**
   * @brief Wrapper for all operations needed to reinit the level set
   * @return size_type
   */
  size_type run_iteration();

 private:
  static NumberType smooth_signum(const NumberType phi0, const NumberType eta);

  static NumberType smooth_signum(
    const NumberType phi0,
    const dealii::Tensor<1, dim, NumberType>& grad_phi0,
    const NumberType eta);
};


#endif  // FELSPA_BETTER_REINIT

}  // namespace dg

FELSPA_NAMESPACE_CLOSE
#endif  // FELSPA_LEVEL_SET_REINIT_H_
