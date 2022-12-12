#ifndef _FELSPA_PDE_DIFFUSION_H_
#define _FELSPA_PDE_DIFFUSION_H_

#include <felspa/base/felspa_config.h>
#include <felspa/pde/ldg.h>
#include <felspa/pde/pde_base.h>
#include <felspa/pde/time_integration.h>

#include <memory>

FELSPA_NAMESPACE_OPEN

/* -------------------------------------------*/
namespace dg
/* -------------------------------------------*/
{
  /* ************************************************** */
  /**
   * Control for the diffusion simulator
   */
  /* ************************************************** */
  template <int dim, typename NumberType = types::DoubleType>
  struct DiffusionControl : public SimulatorControlBase<NumberType>
  {
    /**
     * Constructor
     */
    DiffusionControl(TempoMethod method = TempoMethod::rktvd3,
                     TempoCategory category = TempoCategory::exp)
      : SimulatorControlBase<NumberType>(),
        ptr_tempo(std::make_shared<TempoControl<NumberType>>(
          method, category, 0.5, 0.9)),
        ptr_ldg(std::make_shared<LDGControl<dim, NumberType>>())
    {}

    std::shared_ptr<TempoControl<NumberType>> ptr_tempo;

    std::shared_ptr<LDGControl<dim, NumberType>> ptr_ldg;
  };


  /* ************************************************** */
  /**
   * Simulator to model diffusion using LDG Scheme
   * and explicit time-stepping
   */
  /* ************************************************** */
  template <int dim, typename NumberType = types::DoubleType>
  class DiffusionSimulator : public FESimulator<dim,
                                                FEDGType<dim>,
                                                DGLinearSystem<dim, NumberType>,
                                                TempoIntegrator<NumberType>>
  {
   public:
    using base_type = FESimulator<dim,
                                  FEDGType<dim>,
                                  DGLinearSystem<dim, NumberType>,
                                  TempoIntegrator<NumberType>>;
    using value_type = NumberType;
    using typename base_type::fe_type;
    using typename base_type::time_step_type;
    using typename base_type::vector_type;
    using control_type = DiffusionControl<dim, value_type>;
    using initial_condition_type = ScalarFunction<dim, value_type>;
    using gradient_vector_type =
      typename LDGGradientLinearSystem<dim, value_type>::vector_type;


    /** \name Basic object behavior */
    //@{
    /**
     * Constructor, with default control parameters
     */
    DiffusionSimulator(Mesh<dim, value_type>& mesh,
                       unsigned int fe_degree,
                       const std::string& label = "Diffusion");

    /**
     * Attach the control parameters to the simulator
     */
    void attach_control(
      const std::shared_ptr<DiffusionControl<dim, value_type>>&);

    /**
     * Initialize the simulator
     */
    void initialize(const initial_condition_type& initial_condition,
                    bool use_independent_solution = false);
    //@}


    /** \name Advancing Time */
    //@{
    using base_type::advance_time;

    /**
     * Compute time step based on CFL estimate.
     */
    time_step_type advance_time(time_step_type time_step,
                                bool compute_single_cycle) override;

    vector_type explicit_time_derivative(
      time_step_type current_time, const vector_type& soln_prev_step) override;
    //@}


    void export_solution(ExportFile& file) const override;


   protected:
    /** \name Simulator Helpers */
    //@{
    void do_initialize(bool use_independent_solution);

    bool is_initialized() const;

    void upon_mesh_update() override;

    void allocate_assemble_system();

    void assemble_mass_matrix();

    void assemble_gradient_mass_matrix();

    void assemble_gradient_rhs();

    void assemble_rhs();

    void solve_linear_system(vector_type& soln, const vector_type& rhs);
    //@}


    /** \name Temporal Helpers */
    //@{
    time_step_type estimate_max_time_step() const;
    //@}


    /**
     * Collection of boundary conditions
     */
    BCBookKeeper<dim, value_type> bcs;

    /**
     * Finite element object
     */
    dealii::FESystem<dim> grad_fe;

    /**
     * DoFHandler object for FE System
     */
    dealii::DoFHandler<dim> grad_dof_handler;

    /**
     * Pointer to a linear system to solve
     */
    std::shared_ptr<LDGGradientLinearSystem<dim, value_type>>
      ptr_grad_linear_system;

    /** Intermediate gradient solution */
    gradient_vector_type solution_gradient;

    /** Pointer to the initial condition function */
    const initial_condition_type* ptr_initial_condition;

    /**
     * Control parameters for diffusion simulator
     */
    std::shared_ptr<control_type> ptr_control;
  };

}  // namespace dg

FELSPA_NAMESPACE_CLOSE

#endif  // _FELSPA_PDE_DIFFUSION_H_ //
