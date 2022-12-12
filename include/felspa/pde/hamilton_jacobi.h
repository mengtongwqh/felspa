#ifndef _FELSPA_PDE_HAMILTON_JACOBI_H_
#define _FELSPA_PDE_HAMILTON_JACOBI_H_

#include <felspa/base/felspa_config.h>
#include <felspa/pde/linear_systems.h>
#include <felspa/pde/ldg.h>
#include <felspa/pde/pde_tools.h>

#include <vector>

FELSPA_NAMESPACE_OPEN
/* --------------- */
namespace dg
/* --------------- */
{
  // forward declaration
  namespace internal
  {
    template <int dim, typename NumberType>
    class HJAssemblerMeshWorker;

    template <int dim, typename NumberType>
    class HJAssemblerWorkStream;
  }  // namespace internal

  template <int dim, typename NumberType>
  class HJSimulator;

  // Implementation for HJAssembler: MeshWorker or WorkStream
  template <int dim, typename NumberType>
  using HJAssembler = internal::HJAssemblerWorkStream<dim, NumberType>;

  /* ************************************************** */
  /**
   * This is a struct that contains the commonly-used
   * typedefs for an \c HJSimulator
   */
  /* ************************************************** */
  template <int dim, typename NumberType>
  struct HJTypedefs
  {
    /** Type of the solution vector */
    using value_type = NumberType;
    using vector_type = typename DGLinearSystem<dim, value_type>::vector_type;
    using function_type = ScalarFunction<dim, value_type>;
    using gradient_vector_type = dealii::BlockVector<value_type>;
  };


  /* ************************************************** */
  /**
   * Base class for numerical Hamilton-Jacobi operator.
   * A numerical Hamilton-Jacobi \f$ \hat{H} \f$ is an approximation
   * of an exact Hamilton-Jacobi operator \f$ H(\nabla\phi) $\f following form:
   * \f[
   * H(\nabla\phi) \approx \hat{H}(\nabla\phi_l,\nabla\phi_r)
   * \f]
   * and \f$\hat{H}$\f must be monotonous in the following sense
   * to guarantee stability: \f$\hat{H()}$\f
   */
  /* ************************************************** */
  template <int dim, typename NumberType>
  class HJOperator
  {
   public:
    using value_type = NumberType;

    /**
     * Constructing the Hamilton-Jacobi operator
     */
    HJOperator() = default;


    /**
     * Destructor
     */
    virtual ~HJOperator() = default;


    /**
     * Initialize the Hamilton-Jacobi operator by
     * in the Hamilton-Jacobi solver.
     */
    virtual void initialize(
      const HJSimulator<dim, value_type>& hj_simulator) = 0;


    /**
     * Test if the operator is properly initialized.
     */
    virtual bool is_initialized() const { return ptr_hj_simulator != nullptr; }


    /**
     * @brief Evaluate HJ values at quadrature point in a cell
     * @param cell active cell iterator pointing to current cell
     * @return std::vector<value_type> HJ operator values on quadrature points
     */
    virtual void cell_values(const dealii::FEValuesBase<dim>& cell,
                             std::vector<value_type>& hj_values) const = 0;


    /**
     * Estimate the characteristic wave speed of the HJ operator
     * at each quadrature point of the cell.
     */
    virtual void cell_velocities(
      const dealii::FEValuesBase<dim>& feval,
      std::vector<dealii::Tensor<1, dim, value_type>>& hj_velocities) const = 0;


   protected:
    /**
     * @brief extract left and right gradients
     * @pre The \c dealii::FEValues<dim> is initialized to
     * the active cell where gradients are desired to be computed
     */
    void extract_cell_gradients(
      const dealii::FEValuesBase<dim>& feval,
      std::vector<dealii::Tensor<1, dim, value_type>>& lgrad,
      std::vector<dealii::Tensor<1, dim, value_type>>& rgrad) const;


    /**
     * Back pointer to the simulator;
     */
    dealii::SmartPointer<const HJSimulator<dim, value_type>,
                         HJOperator<dim, value_type>>
      ptr_hj_simulator = nullptr;
  };


  /* ************************************************** */
  /**
   * Control parameters for HJ simulator
   */
  /* ************************************************** */
  template <int dim, typename NumberType = types::DoubleType>
  class HJControl : public SimulatorControlBase<NumberType>
  {
   public:
    /**
     *
     */
    HJControl(TempoMethod method = TempoMethod::rktvd3,
              TempoCategory category = TempoCategory::exp)
      : SimulatorControlBase<NumberType>(),
        ptr_tempo(std::make_shared<TempoControl<NumberType>>(method, category,
                                                             0.5, 0.9)),
        ptr_ldg(std::make_shared<LDGControl<dim, NumberType>>())
    {}


    /**
     * Control parameter for time integrator.
     */
    std::shared_ptr<TempoControl<NumberType>> ptr_tempo;


    std::shared_ptr<LDGControl<dim, NumberType>> ptr_ldg;
  };


  /* ************************************************** */
  /**
   * Solver object to compute Hamilton-Jacobi type equations
   */
  /* ************************************************** */
  template <int dim, typename NumberType = types::DoubleType>
  class HJSimulator
    : public FESimulator<dim, FEDGType<dim>, DGLinearSystem<dim, NumberType>,
                         TempoIntegrator<NumberType>>,
      public HJTypedefs<dim, NumberType>
  {
   public:
    using base_type =
      FESimulator<dim, FEDGType<dim>, DGLinearSystem<dim, NumberType>,
                  TempoIntegrator<NumberType>>;
    using value_type = NumberType;
    using vector_type = typename DGLinearSystem<dim, NumberType>::vector_type;
    using typename HJTypedefs<dim, value_type>::gradient_vector_type;
    using time_step_type = typename TempoIntegrator<value_type>::value_type;
    using bc_fcn_type = BCFunction<dim, value_type>;


    /** \name Basic object behaviors */
    //@{
    /** Constructor */
    HJSimulator(Mesh<dim, value_type>& triag, unsigned int fe_degree,
                const std::string& label = "HJ");

    /**
     * Constructor.
     * Construct from any arbitrary FESimulator.
     * Sharing the FE,  DofHandler and mapping
     */
    HJSimulator(const base_type& fe_simulator);

    /** Copy Constructor */
    HJSimulator(const HJSimulator<dim, NumberType>& that);

    /** Copy assignment */
    HJSimulator<dim, NumberType>& operator=(
      const HJSimulator<dim, NumberType>&);

    /** Destructor */
    ~HJSimulator() = default;
    //@}


    /** \name Initialization */
    //@{
    void attach_control(
      const std::shared_ptr<HJControl<dim, value_type>>& pcontrol);

    void initialize(
      const ScalarFunction<dim, value_type>& initial_condition,
      const std::shared_ptr<HJOperator<dim, value_type>>& p_hj = nullptr,
      bool execute_mesh_refine = true,
      bool use_independent_solution = false);

    void initialize(
      const vector_type& initial_condition,
      const std::shared_ptr<HJOperator<dim, value_type>>& p_hj = nullptr,
      bool use_independent_solution = false);

    void initialize(
      const TimedSolutionVector<vector_type, time_step_type>& other_soln,
      const std::shared_ptr<HJOperator<dim, value_type>>& p_hj = nullptr);


    void discretize_function_to_vector(
      const ScalarFunction<dim, value_type>& fcn, vector_type& vect) const;


    void discretize_function_to_solution(
      const ScalarFunction<dim, value_type>& fcn);

    /**
     * Attach the p_hj to HJOperator.
     * Also initialize the HJOperator with the initial condition.
     */
    void init_hj_operator(
      const std::shared_ptr<HJOperator<dim, value_type>>& p_hj = nullptr);

    /**
     * Testing if the simulator is initialized.
     */
    bool is_initialized() const;
    //@}


    /** \name Temporal Methods */
    //@{
    /** Forward simulator for given \c time_step */
    virtual time_step_type advance_time(time_step_type time_step,
                                        bool compute_single_cycle) override;

    using base_type::advance_time;


    /** Compute explicit temporal derivative */
    vector_type explicit_time_derivative(
      time_step_type current_time, const vector_type& soln_prev_step) override;

    /**
     * Compute the value of max(velocity / cell diameter). This value will be
     * used for CFL estimation. Will call \c characteristic_speed function in
     * \class HJOperator
     */
    value_type max_velocity_over_diameter(time_step_type current_time) const;

    /**
     * CFL scaling for Hamilton-Jacobi without artificial diffusion
     */
    static time_step_type cfl_scaling(const unsigned int fe_degree,
                                      const value_type max_velo_diam);
    /**
     * CFL scaling for Hamilton-Jacobi with artificial diffusion
     */
    static time_step_type cfl_scaling(const unsigned int fe_degree,
                                      const value_type viscosity,
                                      const value_type max_velo_diam,
                                      const value_type min_diam);
    //@}


    /** @name Getting Members */
    //@{
    /**
     *  Get the local gradient vector
     */
    const gradient_vector_type& get_local_gradients(LDGFluxEnum flux) const;

    /**
     * Getter for the \c HJOperator
     */
    const HJOperator<dim, value_type>& get_hj() const { return *ptr_hj; }
    //@}


   protected:
    /** \name Simulator helper function */
    //@{
    /**
     * Using the local discontinuous Galerkin (LDG) method to approximate
     * the left/right gradient for each node
     */
    void local_left_right_gradients();

    /**
     * Initialize dof_handler and and allocate space for linear system.
     */
    void upon_mesh_update() override;

    /**
     * Allocate linear system and assemble mass matrix
     */
    void allocate_assemble_system();

    /**
     * Assemble mass matrix on the LHS.
     * Called upon changes of the mesh by post_mesh_update_reallocate().
     */
    void assemble_mass_matrix();

    /**
     * Solving the Hamilton-Jacobi system.
     */
    void solve_linear_system(vector_type& soln,
                             const vector_type& rhs_vector);
    //@}

    /**
     * Control parameters
     */
    std::shared_ptr<HJControl<dim, value_type>> ptr_control;


    /**
     * \name Local gradients
     * Local gradients computed by local Discontinuous Galerkin method.
     * Filled by \c local_left_right_gradients() function.
     */
    //@{
    gradient_vector_type left_local_gradients;
    gradient_vector_type right_local_gradients;
    //@}


    /**
     * Pointer to a numerical Hamilton-Jacobi operator
     */
    std::shared_ptr<HJOperator<dim, value_type>> ptr_hj;


    /**
     * Pointer to CFL estimator
     */
    std::unique_ptr<CFLEstimator<HJSimulator<dim, value_type>>>
      ptr_cfl_estimator = nullptr;


    /**
     * Pointer to the solver to generate artificial diffusion.
     * Such generator are of type \class LDGDiffuionTerm.
     */
    std::shared_ptr<LDGDiffusionTerm<dim, value_type>> ptr_diffusion_generator;


    /**
     * This signal will be called at the end of 
     * \c local_left_right_gradients() to apply limiting to curvature
     */
    boost::signals2::signal<void()>  post_local_gradient_limiting;


   private:
    /**
     * Helper for carrying out initialization.
     */
    void do_initialize(bool use_independent_solution);
  };


  /* ------------------ */
  namespace internal
  /* ------------------ */
  {
    /* ************************************************** */
    /**
     * This class is used for assembling the linear system
     * to solve for Hamilton-Jacobi type problem.
     * We only assemble the RHS vector since the LHS mass matrix
     * has already been assembled in \c local_left_right_gradients().
     */
    /* ***************************************************/
    template <int dim, typename NumberType>
    class HJAssemblerMeshWorker
      : public MeshWorkerAssemblerBase<DGLinearSystem<dim, NumberType>>
    {
     public:
      using base_type =
        MeshWorkerAssemblerBase<DGLinearSystem<dim, NumberType>>;
      using typename base_type::dof_info_t;
      using typename base_type::integration_info_t;
      using typename base_type::local_vector_t;
      using typename base_type::value_type;
      using typename base_type::vector_type;
      using linsys_type = DGLinearSystem<dim, NumberType>;

      /**
       * Constructor
       */
      HJAssemblerMeshWorker(linsys_type&, bool construct_mapping_adhoc = true);


      /**
       * Function executing assembly
       */
      void assemble(const HJOperator<dim, value_type>& hj,
                    const dealii::Quadrature<dim>& quadrature,
                    bool zero_out_rhs = true);

     private:
      class HJLocalIntegrator;
    };


    /* ***************************************************/
    /**
     * @brief
     *
     * @tparam dim
     * @tparam NumberType
     */
    /* ***************************************************/
    template <int dim, typename NumberType>
    class HJAssemblerWorkStream
      : public AssemblerBase<DGLinearSystem<dim, NumberType>>
    {
     public:
      using base_type = AssemblerBase<DGLinearSystem<dim, NumberType>>;
      using linsys_type = DGLinearSystem<dim, NumberType>;
      using value_type = NumberType;
      using active_cell_iterator_type =
        typename dealii::DoFHandler<dim>::active_cell_iterator;


      /**
       * Constructor
       */
      HJAssemblerWorkStream(linsys_type&, bool construct_mapping_adhoc = true);


      /**
       * @brief Assemble (the rhs) of the linear system
       */
      void assemble(const HJOperator<dim, value_type>& hj,
                    const dealii::Quadrature<dim>& quadrature,
                    bool zero_out_rhs = true);


      /**
       * Thread local scratch data
       */
      class ScratchData;


      /**
       *  data to be copied to global matrix
       */
      class CopyData;


     private:
      void local_assembly(const active_cell_iterator_type& cell,
                          ScratchData& scratch, CopyData& copy);


      void copy_local_to_global(const CopyData& copy);
    };
  }  // namespace internal
}  // namespace dg


/* ************************************************* */
/**
 * Specialization of \c VelocityExtractor
 * for \c dg::HJSimulator. Used in CFL estimation.
 */
/* ************************************************* */
template <int dim, typename NumberType>
class VelocityExtractor<dg::HJSimulator<dim, NumberType>, false>
  : public VelocityExtractorBase<dg::HJSimulator<dim, NumberType>>
{
 public:
  constexpr static int dimension = dim;
  using base_type = VelocityExtractorBase<dg::HJSimulator<dim, NumberType>>;
  using typename base_type::simulator_type;
  using typename base_type::tensor_type;
  using value_type = NumberType;

  /**
   * Extract wave speeds from by invoking \c HJOperator
   */
  void extract(const simulator_type& sim,
               const dealii::FEValuesBase<dim>& feval,
               std::vector<tensor_type>& velocities) const override
  {
    ASSERT(feval.n_quadrature_points == velocities.size(),
           ExcSizeMismatch(feval.n_quadrature_points, velocities.size()));
    sim.get_hj().cell_velocities(feval, velocities);
  }
};


FELSPA_NAMESPACE_CLOSE

#endif  // _FELSPA_PDE_HAMILTON_JACOBI_H_ //
