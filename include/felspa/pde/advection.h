#ifndef _FELSPA_PDE_ADVECTION_H_
#define _FELSPA_PDE_ADVECTION_H_

#include <deal.II/base/smartpointer.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/distributed/solution_transfer.h>
#include <felspa/base/felspa_config.h>
#include <felspa/base/function.h>
#include <felspa/base/numerics.h>
#include <felspa/fe/sync_iterators.h>
#include <felspa/linear_algebra/linear_system.h>
#include <felspa/linear_algebra/system_assembler.h>
#include <felspa/mesh/mesh_refine.h>
#include <felspa/pde/linear_systems.h>
#include <felspa/pde/pde_tools.h>
#include <felspa/pde/time_integration.h>

#include <functional>

FELSPA_NAMESPACE_OPEN

/* --------------*/
namespace dg
/* --------------*/
{
  /* ************************************************** */
  /**
   * \defgroup AdvectionSolvers Advection Solvers
   * Classes in this modules are dedicated to solve the
   * non-steady advection-type partial differential equations
   * The problems are, in general, of the following form:
   * \f{equation}{
   * \frac{\partial{\phi}}{\partial{t}} +
   * \nabla\cdot\mathbf{F}(\phi) = g(x,t)
   * \f}
   * After casting into the weak form we have
   * \f{equation}{
   * M\frac{\phi^{n+1} - \phi^{n}}{\Delta{t}} = F{\phi}
   * \f}
   */
  /* ************************************************** */

  // forward decalaration
  template <AssemblyFramework framework, typename VeloFcnType>
  class AdvectAssembler;

  template <int dim, typename NumberType>
  class AdvectAssemblerBase;


  /* ************************************************** */
  /**
   * Control parameters for advection simulator
   */
  /* ************************************************** */
  template <typename NumberType>
  struct AdvectControl : public SimulatorControlBase<NumberType>
  {
    /**
     * Constructor. Allocate the TempoControl object
     */
    AdvectControl(TempoMethod method = TempoMethod::rktvd3,
                  TempoCategory category = TempoCategory::exp)
      : SimulatorControlBase<NumberType>(),
        ptr_tempo(std::make_shared<TempoControl<NumberType>>(method, category,
                                                             0.5, 0.9))
    {}

    /**
     * Control parameters for time integrator
     */
    std::shared_ptr<TempoControl<NumberType>> ptr_tempo;

    /**
     * The type of matrix assembler we are using
     */
    AssemblyFramework assembly_framework = AssemblyFramework::workstream;
  };


  /* ************************************************** */
  /**
   * \class AdvectSimulator
   * \brief Solving linear advection equation with a given velocity field
   * with discontinuous Galerkin method as the spatial discretization.
   * \ingroup AdvectionSolvers
   */
  /* ************************************************** */
  template <int dim, typename NumberType = types::DoubleType>
  class AdvectSimulator
    : public FESimulator<dim, FEDGType<dim>, DGLinearSystem<dim, NumberType>,
                         TempoIntegrator<NumberType>>
  {
    friend class FESimulator<dim, FEDGType<dim>,
                             DGLinearSystem<dim, NumberType>>;

   public:
    using fe_simulator_base_type =
      FESimulator<dim, FEDGType<dim>, DGLinearSystem<dim, NumberType>>;

    using base_type =
      FESimulator<dim, FEDGType<dim>, DGLinearSystem<dim, NumberType>,
                  TempoIntegrator<NumberType>>;

    using typename base_type::fe_type;

    using linsys_type = DGLinearSystem<dim, NumberType>;

    using vector_type = typename DGLinearSystem<dim, NumberType>::vector_type;

    using value_type = typename vector_type::value_type;

    using time_step_type = typename TempoIntegrator<NumberType>::value_type;

    using control_type = AdvectControl<NumberType>;

    using source_term_type = ScalarFunction<dim, value_type>;

    using typename base_type::bcs_type;

    using bc_fcn_type = BCFunction<dim, value_type>;

    using periodic_bc_type = PeriodicBCFunction<dim, value_type>;


    /** \name Basic Object Behavior */
    //@{
    /** Constructor, with default control parameters */
    AdvectSimulator(Mesh<dim, value_type>& triag, unsigned int fe_degree,
                    const std::string& label = "Advect");

    /** Constructor, with control parameters passed in */
    AdvectSimulator(Mesh<dim, value_type>& triag, unsigned int fe_degree,
                    const std::shared_ptr<AdvectControl<value_type>>& pcontrol,
                    const std::string& label = "Advect");

    /**
     * Copy constructor. Note that the copy constructor will nullify velocity
     * field, boundary conditions and source term.
     */
    AdvectSimulator(const AdvectSimulator<dim, NumberType>& that);

    /**
     * Copy assignment. Note that the copy assignment will nullify velocity
     * field, boundary conditions and source term.
     */
    AdvectSimulator<dim, NumberType>& operator=(
      const AdvectSimulator<dim, NumberType>&);

    /** Destructor */
    virtual ~AdvectSimulator() = default;
    //@}


    /** \name Initialization */
    //@{
    void attach_control(const std::shared_ptr<AdvectControl<value_type>>&);

    /**
     * Initialize simulator with initial condition
     * defined as a (smart pointer to a) function.
     */
    template <typename VeloFcnType>
    void initialize(const ScalarFunction<dim, value_type>& initial_condition,
                    const std::shared_ptr<VeloFcnType>& pvfield,
                    const std::shared_ptr<source_term_type>& prhs,
                    bool execute_mesh_refine,
                    bool use_independent_solution = false);

    /**
     * Initialize simulator with initial condition
     * defined as a Lvalue reference of a vector.
     */
    template <typename VeloFcnType>
    void initialize(const vector_type& initial_condition,
                    const std::shared_ptr<VeloFcnType>& vfield,
                    const std::shared_ptr<source_term_type>& prhs = nullptr,
                    bool use_independent_solution = false);

    /**
     * Initialize simulator with initial condition
     * defined as a Rvalue reference of a vector.
     */
    template <typename VeloFcnType>
    void initialize(vector_type&& initial_condition,
                    const std::shared_ptr<VeloFcnType>& vfield,
                    const std::shared_ptr<source_term_type>& prhs = nullptr,
                    bool use_independent_solution = false);

    /**
     * \p true if RHS source term,
     * boundary condition and initial condition
     * are all properly initialized through \c initialize()
     * function.
     */
    bool is_initialized() const;

    /**
     * Compute the projected/interpolated values of a function field
     * onto a discrete vector.
     */
    void discretize_function_to_vector(
      const ScalarFunction<dim, value_type>& function_field,
      vector_type& vector) const;

    /**
     * Using the function above, interpolate a
     * initial condition to solution vector.
     */
    void discretize_function_to_solution(
      const ScalarFunction<dim, value_type>& function_field);
    //@}


    /** \name Accessors */
    //@{
    /** Accessing control parameters */
    AdvectControl<NumberType>& control()
    {
      ASSERT(ptr_control, ExcNullPointer());
      return *ptr_control;
    }

    /** Const overload. */
    const AdvectControl<NumberType>& control() const
    {
      ASSERT(ptr_control, ExcNullPointer());
      return *ptr_control;
    }
    //@}


    /** \name Temporal Methods */
    //@{
    /**
     * Advance the simulator for the speicified \c time_step,
     * no questions asked, no stability checks.
     * Recommended usage is to call this function after getting the maximum
     * allowed time step from  \c estimate_max_time_step().
     */
    void advance_time(time_step_type time_step) override;

    /**
     * Forward simulator for given \c time_step.
     * If \c compute_single_cycle is true,
     * then only one time step will be computed.
     */
    time_step_type advance_time(time_step_type time_step,
                                bool compute_single_cycle) override;

    /**
     * Advance the simulator for one time step.
     * The size of the forwarded time step depends upon CFL auto-adjustment
     */
    time_step_type advance_time();

    /**
     * The maximum time step allowed for this time step.
     */
    time_step_type estimate_max_time_step() const override;

    /** Compute explicit temporal derivative */
    vector_type explicit_time_derivative(
      time_step_type current_time, const vector_type& soln_prev_step) override;

    /** CFL scaling as proposed by Chalmer and Kridonova */
    static time_step_type cfl_scaling(const unsigned int fe_degree);

    /** Estimate cellwise maximum velocity over diameter */
    value_type max_velocity_over_diameter(time_step_type current_time) const;
    //@}


    /**
     * Integrate error of the solution.
     * Taking the initial condition vector as the analytical solution
     */
    value_type compute_error(
      const ScalarFunction<dim, value_type>& analytical_soln,
      dealii::VectorTools::NormType norm_type =
        dealii::VectorTools::L2_norm) const;

    /**
     * compute the error w.r.t another vector
     */
    value_type compute_error(const vector_type& soln) const;


   protected:
    /**
     * Declaration of the BatchSolutionTransfer object actually used in
     * this simulator.
     */
    struct AdvectSolutionTransfer;


    /** @name Simulator Helper Functions */
    //@{
    /**
     * (Re-)setup the simulator when upon mesh update.
     * Reset constraints, reallocate linear system,
     * and falsify \c mesh_update_detected flag.
     */
    void upon_mesh_update() override;

    /** Flag mesh cell for refinement. */
    void do_flag_mesh_for_coarsen_and_refine(
      const MeshControl<value_type>& mesh) const override;

    /**
     * Allocate space for linear system and then assemble mass matrix.
     * This will either be called in \c initialize(),
     * or through \c explicit_time_derivative() upon receiving
     * \c mesh_update_detected flag
     */
    void allocate_assemble_system();

    /**
     * Assemble the LHS mass matrix.
     * As a helper function in \c setup_system()
     */
    void assemble_mass_matrix();

    /** RHS assembly */
    void assemble_rhs(const vector_type& soln_prev_step);

    /** Solve system by linear solver */
    void solve_linear_system(vector_type& soln,
                              const vector_type& rhs);
    //@}


    /**
     * Control parameters for advection simulator.
     */
    std::shared_ptr<AdvectControl<NumberType>> ptr_control;


    /**  \name Velocity/Boundary function */
    //@{
    /**
     * \brief Generalized velocity function.
     * Note that in here we are assuming that the flux function is separable.
     * In terms of implementation, one way to do this is by defining it
     * as a virtual function so that the application will inherit this class
     * and define its own boundary condition or wind field. In this class we
     * store a pointer to that function because it is polymorphic.
     */
    std::weak_ptr<TimedPhysicsObject<value_type>> ptr_velocity_field;

    /** RHS function, or source term */
    std::shared_ptr<source_term_type> ptr_source_term = nullptr;

    /** A pointer to the CFL esitimator */
    std::unique_ptr<CFLEstimatorBase<dim, value_type>> ptr_cfl_estimator =
      nullptr;
    //@}


   private:
    /**
     * Extracted out code common to all initialization function.
     * It will:
     * 1) zero out time in the simulator as well as all attached objects;
     * 2) allocate space for linear system and solution vector;
     * 3) update flags
     * After this function call, you might want to set
     * mesh_update_detected to \c false
     */
    template <typename VeloFcnType>
    void do_initialize(const std::shared_ptr<VeloFcnType>& pvfield,
                       const std::shared_ptr<source_term_type>& psource,
                       bool use_independent_solution);

    /**
     * Pointer to matrix assembler.
     */
    std::unique_ptr<AdvectAssemblerBase<dim, value_type>> ptr_rhs_assembler;
  };  // class AdvectDG


  /* ************************************************** */
  /**
   * BatchSolutionTransfer specialized for Advection
   */
  /* ************************************************** */
  template <int dim, typename NumberType>
  struct AdvectSimulator<dim, NumberType>::AdvectSolutionTransfer
    : public SolutionTransferBase
  {
    AdvectSolutionTransfer(
      const std::shared_ptr<const fe_type>& pfe,
      const std::shared_ptr<dealii::DoFHandler<dim>>& pdofh,
      TimedSolutionVector<vector_type, time_step_type>& solution);

    virtual void prepare_for_coarsening_and_refinement() override;

    virtual void interpolate() override;

    std::shared_ptr<const fe_type> ptr_fe;

    std::shared_ptr<dealii::DoFHandler<dim>> ptr_dof_handler;

    dealii::SmartPointer<TimedSolutionVector<vector_type, time_step_type>>
      ptr_soln;

#ifdef FELSPA_HAS_MPI
    dealii::parallel::distributed::SolutionTransfer<dim, vector_type,
                                                    dealii::DoFHandler<dim>>
#else
    dealii::SolutionTransfer<dim, vector_type, dealii::DoFHandler<dim>>
#endif
      soln_transfer;
  };


  /* ************************************************** */
  /**
   * Base class for the advection system assembly
   */
  /* ************************************************** */
  template <int dim, typename NumberType>
  class AdvectAssemblerBase
    : public AssemblerBase<DGLinearSystem<dim, NumberType>>
  {
   public:
    using value_type = NumberType;
    using vector_type = typename AdvectSimulator<dim, value_type>::vector_type;
    using bcs_type = typename AdvectSimulator<dim, value_type>::bcs_type;
    using linsys_type = DGLinearSystem<dim, NumberType>;
    using source_term_type =
      typename AdvectSimulator<dim, value_type>::source_term_type;

    /**
     * Assembly function
     */
    virtual void assemble(
      const dealii::Quadrature<dim>& ptr_quad, const vector_type& solution,
      const bcs_type& bcs,
      const std::shared_ptr<const source_term_type>& ptr_source_term) = 0;

   protected:
    /**
     * Constructor
     */
    AdvectAssemblerBase(linsys_type& linsys)
      : AssemblerBase<DGLinearSystem<dim, NumberType>>(linsys, false)
    {}
  };


  /* ************************************************** */
  /**
   * AdvectAssembler<dim, NumberType, AssemblyFramework>
   */
  /* ************************************************** */


  /* ********************** */
  /**
   * Specialization with
   * \c AdvectAssembler
   * <dim, NumberType,
   * AssemblyFramework::meshworker>
   */
  /* ********************** */
  template <typename VeloFcnType>
  class AdvectAssembler<AssemblyFramework::meshworker, VeloFcnType>
    : public AdvectAssemblerBase<VeloFcnType::dimension,
                                 typename VeloFcnType::value_type>
  {
   public:
    constexpr static int dimension = VeloFcnType::dimension, dim = dimension;
    using base_type = AdvectAssemblerBase<VeloFcnType::dimension,
                                          typename VeloFcnType::value_type>;
    using velocity_fcn_type = VeloFcnType;
    using typename base_type::bcs_type; /**< BC Collection */
    using typename base_type::linsys_type;
    using typename base_type::source_term_type;
    using typename base_type::value_type;
    using typename base_type::vector_type;


    /** Constructor */
    AdvectAssembler(
      linsys_type& linsys,
      const std::weak_ptr<const velocity_fcn_type>& ptr_velocity_field);


    /** assemble the linear matrix */
    void assemble(
      const dealii::Quadrature<dim>& ptr_quad, const vector_type& solution,
      const bcs_type& bcs,
      const std::shared_ptr<const source_term_type>& ptr_source_term) override;


   protected:
    using dof_info_t = dealii::MeshWorker::DoFInfo<dim, dim, value_type>;
    using integration_info_t = dealii::MeshWorker::IntegrationInfo<dim, dim>;
    using local_matrix_t = dealii::FullMatrix<value_type>;
    using local_vector_t = dealii::Vector<value_type>;


    /** \name Assembler for Cell Volume, Cell Face and Domain Boundary */
    //@{
    /** Integrator within the cell volume */
    void assemble_cell(dof_info_t& dinfo, integration_info_t& cinfo,
                       const std::shared_ptr<const source_term_type>& p_source);

    /** Integrator for cell face */
    void assemble_face(dof_info_t& dinfo1, dof_info_t& dinfo2,
                       integration_info_t& cinfo1, integration_info_t& cinfo2);

    /** Integrator for boundary term */
    void assemble_boundary(dof_info_t& dinfo, integration_info_t& cinfo,
                           const bcs_type& bcs);
    //@}


    /**
     * Pointer to velocity field
     */
    const std::weak_ptr<const velocity_fcn_type> ptr_velocity_field;

    /**
     * Simplify retrieving data from \c global_data in \c IntegratorInfo by
     * invoking an identifier string
     */
    FEFunctionSelector<dim, dim, value_type> fe_fcn_selector;
  };  // class AdvectAssembler<dim, NumberType, AssemblyFramework::meshworker>
      // //


  /* ********************** */
  /**
   * Specialization with
   * \c AdvectAssembler
   * <dim, NumberType,
   * AssemblyFramework::workstream>
   * This allows extraction from another simulator
   * the velocity information
   */
  /* ********************** */
  template <typename VeloFcnType>
  class AdvectAssembler<AssemblyFramework::workstream, VeloFcnType>
    : public AdvectAssemblerBase<VeloFcnType::dimension,
                                 typename VeloFcnType::value_type>
  {
   public:
    constexpr static int dim = VeloFcnType::dimension,
                         dimension = VeloFcnType::dimension;
    using velocity_fcn_type = VeloFcnType;
    using value_type = typename VeloFcnType::value_type;
    using base_type = AdvectAssemblerBase<dim, value_type>;
    using typename base_type::bcs_type;
    using typename base_type::linsys_type;
    using typename base_type::source_term_type;
    using typename base_type::vector_type;

    /**
     * Component of local data. Base class definition.
     */
    class ScratchData;

    struct ScratchDataBox;

    /**
     * Data that will be assembled serially into the global system.
     */
    class CopyData;

    struct CopyDataBox;

    /**
     * Constructor.
     */
    AdvectAssembler(linsys_type& linsys,
                    const std::weak_ptr<const VeloFcnType>& pvfield)
      : base_type(linsys), ptr_velocity_field(pvfield)
    {}


    void assemble(
      const dealii::Quadrature<dim>& ptr_quad,
      const vector_type& soln,
      const bcs_type& bcs,
      const std::shared_ptr<const source_term_type>& ptr_rhs) override;


   protected:
    using cell_iterator_type =
      typename dealii::DoFHandler<dim>::active_cell_iterator;
    using face_iterator_type = typename dealii::DoFHandler<dim>::face_iterator;
    using synced_iterators_type = SyncedActiveIterators<dim>;

    /**
     * Compute the local matrix
     */
    void local_assembly(const synced_iterators_type& cell_sync_iter,
                        ScratchDataBox& cell_data, CopyDataBox& copy_data);

    /**
     * Assemble the local matrix into the global matrix
     */
    void copy_local_to_global(const CopyDataBox& data);


    /**
     * \name Assembly helpers for different entities
     */
    //@{
    void cell_assembly(const ScratchData& cell_scratch, CopyData& cell_copy);

    void face_assembly(const ScratchData& scratch_in,
                       const ScratchData& scratch_ex, CopyData& copy_in,
                       CopyData& copy_ex);

    void boundary_assembly(const ScratchData& scratch, CopyData& copy,
                           const bcs_type& bcs);
    //@}

    /**
     * Pointer to the velocity field.
     */
    const std::weak_ptr<const VeloFcnType> ptr_velocity_field;

    const bcs_type* ptr_bcs;
  };

}  // namespace dg

FELSPA_NAMESPACE_CLOSE

/* ------ IMPLEMENTATIONS ------- */
#include "src/advection.implement.h"
/* ------------------------------ */

#endif  // _FELSPA_PDE_ADVECTION_H_
