
#ifndef _FELSPA_PDE_STOKES_COMMON_H_
#define _FELSPA_PDE_STOKES_COMMON_H_

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/numerics/solution_transfer.h>
#include <felspa/base/control_parameters.h>
#include <felspa/base/exceptions.h>
#include <felspa/base/felspa_config.h>
#include <felspa/base/function.h>
#include <felspa/base/types.h>
#include <felspa/linear_algebra/linear_operators.h>
#include <felspa/linear_algebra/linear_system.h>
#include <felspa/linear_algebra/system_assembler.h>
#include <felspa/pde/boundary_conditions.h>
#include <felspa/pde/pde_base.h>
#include <felspa/pde/pde_tools.h>
#include <felspa/physics/material_base.h>

#ifdef USE_FELSPA_SPARSEILU
#include <felspa/linear_algebra/ilu.h>
#else
#include <deal.II/lac/sparse_ilu.h>
#endif

#include <set>

FELSPA_NAMESPACE_OPEN

// forward declarations //
namespace ls
{
  template <int dim, typename NumberType>
  class MaterialStack;
}  // namespace ls

template <int dim, typename NumberType>
class StokesLinearSystem;

#if defined(FELSPA_HAS_MPI) && defined(DEAL_II_WITH_MPI)
namespace mpi
{
  namespace trilinos
  {
    template <int dim>
    class StokesLinearSystem;
  }
}  // namespace mpi
#endif  // FELSPA_HAS_MPI && DEAL_II_WITH_MPI


namespace internal
{
  /* ************************************************** */
  /**
   * Thread-local data for preparing local matrix
   * for shared-memory parallel assembly.
   * Used by StokesAssember.  The existence of
   * such object is to prevent unnecessary
   * allocation of memory every time when a new cell
   * is visited.
   */
  /* ************************************************** */
  template <int dim, typename NumberType>
  class StokesAssemblyScratch
  {
   public:
    using value_type = NumberType;
    using cell_iterator_type =
      typename dealii::DoFHandler<dim>::active_cell_iterator;

    /**
     * Constructor.
     */
    StokesAssemblyScratch(
      const dealii::FiniteElement<dim>& fe, const dealii::Mapping<dim>& mapping,
      const dealii::Quadrature<dim>& quadrature,
      const dealii::UpdateFlags update_flags,
      const std::shared_ptr<const MaterialBase<dim, value_type>>& p_material);


    /**
     * Copy constructor.
     */
    StokesAssemblyScratch(const StokesAssemblyScratch<dim, NumberType>& that);


    /**
     * Do reinitialization when we are at new cell.
     */
    void reinit(const cell_iterator_type& cell);

    /**
     * FEValues object.
     * Re-inited for each cell.
     */
    dealii::FEValues<dim> fe_values;


    /**
     * Polymorphic pointer to the material accessor
     */
    std::shared_ptr<MaterialAccessorBase<dim, value_type>>
      ptr_material_accessor;


    /**
     * Auxillary struct to pass kinematics to material.
     */
    PointsField<dim, value_type> pts_field;


    /**
     * \name Cached shape function
     */
    //@{
    /**
     * Symmetric gradient of the velocity field
     */
    std::vector<dealii::SymmetricTensor<2, dim, value_type>> sym_grad_v;

    /**
     * Divergence of velocity shape function
     */
    std::vector<value_type> div_v;

    /**
     * Pressure shape function values
     */
    std::vector<value_type> p;

    /**
     * Velocity shape function values
     */
    std::vector<dealii::Tensor<1, dim, value_type>> v;


    /**
     * velocity gradient
     */
    std::vector<dealii::Tensor<2, dim, value_type>> grad_v;

    //@}


    /**
     * \name Cached function values at quadrature points.
     * Shall we do this in the form of a map?
     */
    //@{
    /** Source term values */
    std::vector<dealii::Tensor<1, dim, value_type>> source;

    /** Visocity */
    std::vector<value_type> viscosity;

    /** Denisty */
    std::vector<value_type> density;
    //@}
  };


  /* ************************************************** */
  /**
   * Copy data for the Stokes Assembler
   */
  /* ************************************************** */
  template <int dim, typename NumberType>
  class StokesAssemblyCopy
  {
   public:
    using value_type = NumberType;
    using size_type = types::DoFIndex;

    /**
     * Constructor
     */
    StokesAssemblyCopy(const dealii::FiniteElement<dim>& fe);


    /**
     * Copy constructor.
     */
    StokesAssemblyCopy(const StokesAssemblyCopy<dim, NumberType>& that);


    /**
     * Copy assignment.
     */
    StokesAssemblyCopy<dim, NumberType>& operator=(
      const StokesAssemblyCopy<dim, NumberType>& that) = default;

    /**
     * \name Local data to be copied into global data
     */
    //@{
    std::vector<size_type> local_dof_indices;

    dealii::FullMatrix<value_type> local_matrix;

    dealii::FullMatrix<value_type> local_preconditioner;

    dealii::Vector<value_type> local_rhs;
    //@}
  };


  /* ************************************************** */
  /**
   * @brief
   *
   * @tparam MatrixType
   */
  /* ************************************************** */
  template <typename MatrixType>
  struct VectorType;

  template <typename NumberType>
  struct VectorType<dealii::BlockSparseMatrix<NumberType>>
  {
    using type = dealii::BlockVector<NumberType>;
  };

#if defined(FELSPA_HAS_MPI) && defined(DEAL_II_WITH_TRILINOS)
  template <>
  struct VectorType<
    ::FELSPA_NAMESPACE::mpi::trilinos::dealii::BlockSparseMatrix>
  {
    using type = ::FELSPA_NAMESPACE::mpi::trilinos::dealii::BlockVector;
  };
#endif  // FELSPA_HAS_MPI && DEAL_II_WITH_TRILINOS


  /* ************************************************** */
  /**
   * @brief Scale the linear matrix so that
   * the matrix is not affected by round off error
   * in the solution process.
   *
   * May, Dave A., and Louis Moresi. “Preconditioned Iterative Methods for
   * Stokes Flow Problems Arising in Computational Geodynamics.” Physics of the
   * Earth and Planetary Interiors 171, no. 1–4 (December 2008): 33–47.
   * https://doi.org/10.1016/j.pepi.2008.07.036.
   */
  /* ************************************************** */
  template <typename MatrixType>
  class StokesScalingOperator
  {
   public:
    using matrix_type = MatrixType;
    using matrix_block_type = typename matrix_type::BlockType;
    using vector_type = typename VectorType<matrix_type>::type;
    using size_type = typename MatrixType::size_type;
    using value_type = typename MatrixType::value_type;

    StokesScalingOperator() = default;

    /**
     * @brief allocate and compute the scaling coefficients
     */
    void initialize(const matrix_type& matrix);

    /**
     * @brief Use the initialized value to scale the matrix
     */
    void apply_to_matrix(matrix_type& matrix) const;

    void apply_to_preconditioner(matrix_block_type& precond_matrix) const;

    void apply_to_vector(vector_type& rhs, int component = -1) const;

    void apply_inverse_to_vector(vector_type& solution,
                                 int component = -1) const;

   protected:
    /**
     * @brief BlockVector in the form of [X1, X1]
     */
    vector_type scaling_coeffs;
  };

}  // namespace internal


/* ************************************************** */
/**
 * \c StokesSimulator specialized for difference linear system
 */
/* ************************************************** */
#if defined(FELSPA_HAS_MPI) && defined(DEAL_II_WITH_TRILINOS)
template <int dim, typename NumberType = types::TrilinosScalar,
          typename LinsysType = mpi::trilinos::StokesLinearSystem<dim>>
#else
template <int dim, typename NumberType = types::DoubleType,
          typename LinsysType = StokesLinearSystem<dim, NumberType>>
#endif  // FELSPA_HAS_MPI && DEAL_II_HAS_TRILINOS //
class StokesSimulator;


/* ************************************************** */
/**
 * @brief StokesAssembler class.
 * Specialized for StokesLinearSystem
 * and mpi::trilinos::StokesLinearSystem
 */
/* ************************************************** */
template <typename LinsysType>
class StokesAssembler;


/* ************************************************** */
/**
 * @brief Solution method for the linnear solver
 * FC - fully coupled
 * SCR - Schur complement reduction
 */
/* ************************************************** */
enum StokesSolutionMethod
{
  FC = 0,
  SCR,
  CompareTest
};

std::string to_string(StokesSolutionMethod);


/* ************************************************** */
/**
 * @brief Control parameters for Stokes solver
 */
/* ************************************************** */
class StokesSolverControl : public dealii::ReductionControl
{
 public:
  using base_type = dealii::ReductionControl;

  StokesSolverControl(const unsigned int n = 100, const double tol = 1.0e-10,
                      const double reduce = 1.0e-6,
                      const bool log_history = false,
                      const bool log_result = true)
    : base_type(n, tol, reduce, log_history, log_result)
  {}

  /**
   * @brief Write statistics to csv file
   */
  void write_statistics(const std::string& filename);

  /**
   * @brief Inner iteration for A^{-1}.
   * Not relevant for Schur complement reduction.
   */

  bool apply_diagonal_scaling = false;

  bool log_cg = false;

  bool log_gmres = false;

  std::vector<double> cg_error;

  std::vector<double> cg_timer;

  std::vector<unsigned int> n_cg_inner_iter;

  std::vector<unsigned int> n_cg_outer_iter;

  std::vector<double> gmres_error;

  std::vector<double> gmres_timer;

  std::vector<unsigned int> n_gmres_iter;

  std::vector<double> soln_diff_l2;

  std::vector<double> soln_diff_linfty;
};


/* ************************************************** */
/**
 * Control parameters for Stokes Flow simulator
 */
/* ************************************************** */
template <typename NumberType>
class StokesControlBase : public SimulatorControlBase<NumberType>
{
 public:
  using value_type = NumberType;


  /**
   * @brief Construct a new StokesControlBase
   */
  StokesControlBase();

  /**
   * @brief Deleted copy constructor
   */
  StokesControlBase(const StokesControlBase<NumberType>&) = delete;

  /**
   * @brief Deleted assignment operator
   */
  StokesControlBase<NumberType>& operator=(
    const StokesControlBase<NumberType>&) = delete;

  /**
   * Set the material parameters that will be exported by the simulator
   */
  void set_material_parameters_to_export(
    const std::set<MaterialParameter>& parameters);

  void write_solver_statistics(const std::string& filename);

  /**
   * Scaling factor for the viscosity
   */
  value_type reference_viscosity = 1.0;

  /**
   * Scaling factor for the length
   */
  value_type reference_length = 1.0;

  /**
   * @brief Solution method for the linear solver.
   */
  StokesSolutionMethod solution_method = FC;

 protected:
  /**
   *  Material paramters that will be exported by StokesSimulator
   */
  std::set<MaterialParameter> material_parameters_to_export;
};


/* ************************************************** */
/**
 * Stokes flow simulator
 * This flow simulator
 */
/* ************************************************** */
template <int dim, typename NumberType, typename LinsysType>
class StokesSimulatorBase
  : public FESimulator<dim, dealii::FESystem<dim>, LinsysType>
{
#if 0
  friend class internal::StokesMaterialParameterExporter<dim, NumberType>;
#endif

 public:
  using base_type = FESimulator<dim, dealii::FESystem<dim>, LinsysType>;

  using typename base_type::fe_type;
  using value_type = NumberType;
  using typename base_type::vector_type;
  using linsys_type = LinsysType;
  using time_step_type = value_type;
  using source_type = TensorFunction<1, dim, value_type>;
  using scalar_fcn_type = dealii::Function<dim, value_type>;
  using bcs_type = BCBookKeeper<dim, value_type>;

  /**
   * Component(s) of solution.
   */
  enum class SolutionComponent : unsigned int
  {
    velocities = 0,
    pressure = 1,
  };

  /**
   * Constructor
   */
  StokesSimulatorBase(Mesh<dim, value_type>& mesh, unsigned int degree_v,
                      unsigned int degree_p, const std::string& label);

  /**
   * Constructor.
   */
  StokesSimulatorBase(Mesh<dim, value_type>& mesh,
                      const dealii::FiniteElement<dim>& fe_v,
                      const dealii::FiniteElement<dim>& fe_p,
                      const std::string& label);


  /** \name Get Info about Simulator */
  //@{
  /**
   * Get the fe_degree with solution component
   */
  unsigned int fe_degree(SolutionComponent comp) const;

  /**
   * Get the fe_degree with solution component
   * represented by underlying integer.
   */
  unsigned int fe_degree(unsigned int component) const override;

  /**
   * @brief Get a modifiable reference to the control parameters
   */
  StokesControlBase<value_type>& control();

  /**
   * @brief Get the control parameters
   */
  const StokesControlBase<value_type>& get_control() const;
  //@}


  /** \name Initialization */
  //@{
  /**
   * Attach the control structure to the simulator.
   */
  void attach_control(const std::shared_ptr<StokesControlBase<value_type>>&);

  /**
   * Initialize the simulator.
   * Set material and gravity model. Depending on the type
   * given to \c MaterialType, an appropriate assembler will be allocated
   * to take over the matrix assembly process.
   */
  template <typename MaterialType>
  void initialize(const std::shared_ptr<MaterialType>& p_material,
                  const std::shared_ptr<TensorFunction<1, dim, value_type>>&
                    p_gravity_model = nullptr,
                  bool use_independent_solution = false);

  /**
   * Test if the simulator is initialized.
   */
  bool is_initialized() const;
  //@}


  /** \name Temporal Updates */
  //@{
  /**
   * Forward simulator for given \c time_step.
   */
  virtual void advance_time(time_step_type time_step) override;

  /**
   * Update the velocity field, but defer temporal info update to
   * \c finalize_time_step(). If the finalize function is not called,
   * further update on time step will be blocked.
   */
  void try_advance_time();

  /**
   * Write the time step into the simulator and make the simulator ready
   * to accept time updates again.
   */
  void finalize_time_step(time_step_type time_step);
  //@}


  /**
   * Flag the cells in the mesh for refinement.
   */
  void do_flag_mesh_for_coarsen_and_refine(
    const MeshControl<value_type>& mesh) const override;


  /**
   * Write solution vector to .vtk file.
   */
  void export_solution(ExportFile& file) const override;


  DECL_EXCEPT_0(
    ExcTimeStepNotFinalized,
    "Previous try_advance_time() on the simulator is not finalized.");


 protected:
  /**
   * Declaration of a solution transfer object used for
   * interpolating solution when mesh refinement is carried out
   */
  struct StokesSolutionTransfer;

  /**
   * Reallocate a vector of the size of the solution vector
   */
  virtual void allocate_solution_vector(vector_type& vect) = 0;


  /** \name Simulator Helper Function */
  //@{
  /**
   * Actions that will be executed immediately when
   * mesh update is detected.
   */
  void upon_mesh_update() override;

  /**
   * Allocate space for the linear system.
   * This will be called whenever mesh update is detected.
   */
  void allocate_system();

  /**
   * Run system assembly with the most up-to-date material parameters.
   * This will be called at each time step.
   */
  void assemble_system();

  /**
   * Compute solution to the linear system
   */
  virtual void solve_linear_system(vector_type& soln, vector_type& rhs);

  /**
   * Solve linear system using RHS of the linear system.
   */
  virtual void solve_linear_system(vector_type& soln);
  //@}


  /**
   * Shared pointer to the materials definition.
   * If the material
   */
  std::weak_ptr<MaterialBase<dim, value_type>> ptr_material;


  /**
   * Source term for the momentum equations
   */
  std::shared_ptr<source_type> ptr_momentum_source = nullptr;


  /**
   * Pointer to control parameters
   */
  std::shared_ptr<StokesControlBase<value_type>> ptr_control = nullptr;


  /**
   * Shared pointer to the matrix assembler which is appropriate
   * for the given material framework.
   */
  std::shared_ptr<StokesAssembler<linsys_type>> ptr_assembler = nullptr;


  /**
   * material parameter exporter
   */
#if 0
  std::unique_ptr<internal::StokesMaterialParameterExporter<dim, NumberType>>
    ptr_material_parameter_exporter = nullptr;
#endif


 private:
  /**
   * If set to \c true, this will block time step update unless
   * \c finalize_time_step() is called.
   */
  bool requires_finalizing_time_step = false;
};


/* ************************************************** */
/**
 * SolutionTransfer class for Stoke simulator.
 * This will take care of transferring solution from
 * one grid to another during mesh refinement/coarsening.
 */
/* ************************************************** */
template <int dim, typename NumberType, typename LinsysType>
struct StokesSimulatorBase<dim, NumberType, LinsysType>::StokesSolutionTransfer
  : public SolutionTransferBase
{
  using value_type = NumberType;
  using vector_type = typename LinsysType::vector_type;

  StokesSolutionTransfer(
    StokesSimulatorBase<dim, NumberType, LinsysType>& stokes_simulator);

  void prepare_for_coarsening_and_refinement() override;

  void interpolate() override;

  dealii::SmartPointer<StokesSimulatorBase<dim, NumberType, LinsysType>>
    ptr_simulator;

  dealii::SmartPointer<const dealii::DoFHandler<dim>> ptr_dof_handler;

  dealii::SmartPointer<TimedSolutionVector<vector_type, time_step_type>>
    ptr_soln;

#ifdef FELSPA_HAS_MPI
  dealii::parallel::distributed::SolutionTransfer<dim, vector_type,
                                                  dealii::DoFHandler<dim>>
    soln_transfer;
#else
  dealii::SolutionTransfer<dim, vector_type, dealii::DoFHandler<dim>>
    soln_transfer;
#endif
};


/* ************************************************** */
/**
 * Velocity extractor for \c StokesSimulator.
 * Will extract cell velocity at quadrature points
 * from the solution vector
 */
/* ************************************************** */
template <int dim, typename NumberType, typename LinsysType>
class VelocityExtractor<StokesSimulator<dim, NumberType, LinsysType>, false>
  : public VelocityExtractorBase<StokesSimulator<dim, NumberType, LinsysType>>
{
 public:
  constexpr static int dimension = dim;
  using value_type = NumberType;
  using simulator_type = StokesSimulator<dim, NumberType, LinsysType>;
  using tensor_type = dealii::Tensor<1, dim, value_type>;

  void extract(const simulator_type& simulator,
               const dealii::FEValuesBase<dim>& feval,
               std::vector<tensor_type>& velocities) const override;
};


/* ************************************************** */
/**
 * Assemble the LHS matrix and RHS vector for Stokes system
 */
/* ************************************************** */
template <typename LinsysType>
class StokesAssemblerBase : public AssemblerBase<LinsysType>
{
 public:
  constexpr static int dim = LinsysType::dimension;
  constexpr static int dimension = LinsysType::dimension;

  using linsys_type = LinsysType;
  using value_type = typename LinsysType::value_type;
  using matrix_type = typename linsys_type::matrix_type;
  using vector_block_type = typename linsys_type::vector_block_type;
  using bcs_type = BCBookKeeper<dim, value_type>;
  using source_type =
    typename StokesSimulator<dim, value_type, LinsysType>::source_type;
  using ScratchData = internal::StokesAssemblyScratch<dim, value_type>;
  using CopyData = internal::StokesAssemblyCopy<dim, value_type>;


  /**
   * Constructor.
   */
  StokesAssemblerBase(linsys_type& linsys, bool construct_mapping_adhoc)
    : AssemblerBase<linsys_type>(linsys, construct_mapping_adhoc)
  {}


  /**
   * Destructor.
   */
  virtual ~StokesAssemblerBase() = default;


  /**
   *  Initialize the assemble with the info needed for assembly.
   */
  void initialize(
    const bcs_type& bcs,
    const std::shared_ptr<const MaterialBase<dim, value_type>>& p_material,
    const std::shared_ptr<source_type>& p_source_term);


  /**
   * Run assembly.
   */
  void assemble(const dealii::Quadrature<dim>& quadrature);


 protected:
  /**
   * Access the preconitioner matrix
   */
  matrix_type& preconditioner_matrix();


  /**
   * Access the preconditioner matrix. \c const overload.
   */
  const matrix_type& preconditioner_matrix() const;


  /**
   * Fill the CopyData object with local results so that \c
   * copy_local_to_global can assemble them to the global system.
   */
  virtual void local_assembly(
    const typename dealii::DoFHandler<dim>::active_cell_iterator& cell,
    ScratchData& scratch, CopyData& data) = 0;


  /**
   * Copy the CopyData into global system.
   */
  void copy_local_to_global(const CopyData& data);


  /**
   * Default update flags for assembly.
   * Did not use \c constexpr \c static since the \c deal.II
   * library did not implement operator| as \c constexpr.
   */
  const dealii::UpdateFlags default_update_flags =
    dealii::update_values | dealii::update_quadrature_points |
    dealii::update_JxW_values | dealii::update_gradients;


  /**
   * The boundary conditions
   */
  dealii::SmartPointer<const bcs_type> ptr_bcs;


  /**
   * Pointer to the material model.
   */
  std::shared_ptr<const MaterialBase<dim, value_type>> ptr_material;


  /**
   * Pointer to source (or, gravity) term.
   */
  std::shared_ptr<const source_type> ptr_momentum_source;
};


namespace internal
{
#if 0
  /* ************************************************** */
  /**
   * A class to export the material parameters of
   * the StokesSimulator.
   */
  /* ************************************************** */
  template <int dim, typename NumberType>
  class StokesMaterialParameterExporter
  {
   public:
    constexpr static int dimension = dim;
    using value_type = NumberType;
    using vector_type = dealii::Vector<value_type>;

    enum UpdateMethod
    {
      projection,
      cell_mean,
    };

    /**
     * @brief Construct a new Stokes Material Parameter Exporter object
     * @param simulator @const reference to stokes simulator
     */
    StokesMaterialParameterExporter(
      const StokesSimulator<dim, value_type>& simulator,
      UpdateMethod = cell_mean);


    /**
     * @brief Inform the object that mesh has been updated.
     * This will be called from the \c upon_mesh_update()
     * function in the \c StokesSimulator
     */
    void mesh_updated() { mesh_update_detected = true; }


    /**
     * Reassemble all material parameter vectors.
     * Resizing of the vectors will also be done
     * if there is a change in mesh config.
     */
    void update_parameters(const dealii::Quadrature<dim>& quadrature);


    /**
     * @brief Write the solution to a file
     */
    void export_parameters() const;


    /**
     * @brief Thread-local data for assembly
     */
    struct ScratchData;


    /**
     * @brief Data that will be assembled into
     * the material parameter vector
     */
    struct CopyData;


   private:
    /**
     * @brief compute the parameter vector by taking cell average.
     */
    void update_by_cell_mean(const dealii::Quadrature<dim>& quadrature);


    /**
     * @brief compute the parameter by projecting into the finite element space
     */
    void update_by_projection(const dealii::Quadrature<dim>& quadrature);


    /**
     * @brief assemble the cell-wise contribution of the material parameter
     */
    void local_assembly(
      const typename dealii::DoFHandler<dim>::active_cell_iterator& cell,
      ScratchData& scratch, CopyData& copy);


    /**
     * @brief assemble the local contribution to the global vector
     */
    void copy_local_to_global(const CopyData& copy);


    /**
     * @brief Material parameters arraged in the form of parameter-vector map.
     */
    std::map<MaterialParameter, dealii::Vector<value_type>> parameter_vectors;


    /**
     * @brief mass matrix for projection
     */
    dealii::SparseMatrix<value_type> mass_matrix;


    /**
     * @brief mass matrix sparisty for projection
     */
    dealii::SparsityPattern sparsity;


    /**
     * @brief RHS for projection
     */
    dealii::Vector<value_type> rhs;


    /**
     * @brief Back pointer to the simulator.
     */
    const StokesSimulator<dim, value_type>* ptr_simulator;


    /**
     * @brief finite element used for assembling parameter vector.
     * We define this to be the scalar (component-wise) space of the velocity.
     */
    const dealii::FiniteElement<dim>* ptr_fe;


    /**
     * @brief  DoFHandler for assemblying parameter.
     */
    dealii::DoFHandler<dim> dof_handler;


    /**
     * @brief  For hanging node constraints
     */
    dealii::AffineConstraints<value_type> constraints;


    /**
     * The current parameter vector we are assemblying for
     */
    MaterialParameter current_parameter;


    /**
     * Will be set to true if mesh update is detected
     */
    bool mesh_update_detected;


    /**
     * @brief Update method that will be used for this
     * updating parameter vector
     */
    UpdateMethod update_method;


    /**
     * @brief exporting pvd file
     */
    mutable PVDCollector<value_type> pvd_collector;
  };


  template <int dim, typename NumberType>
  struct StokesMaterialParameterExporter<dim, NumberType>::ScratchData
  {
    using value_type = NumberType;
    using cell_iterator_type =
      typename dealii::DoFHandler<dim>::active_cell_iterator;

    /**
     * @brief Construct a new Scratch Data object
     */
    ScratchData(
      const dealii::FiniteElement<dim>& fe, const dealii::Mapping<dim>& mapping,
      const dealii::Quadrature<dim>& quadrature,
      const dealii::UpdateFlags update_flags,
      const std::shared_ptr<const MaterialBase<dim, value_type>>& p_material);


    /**
     * @brief Copy constructor
     */
    ScratchData(const ScratchData& that);


    /**
     * @brief Prepare the ScratchData for a new cell.
     */
    void reinit(const cell_iterator_type& cell);


    dealii::FEValues<dim> fe_values;


    std::shared_ptr<MaterialAccessorBase<dim, value_type>>
      ptr_material_accessor;


    PointsField<dim, value_type> pts_field;


    std::vector<value_type> scalar_parameter;
  };


  template <int dim, typename NumberType>
  struct StokesMaterialParameterExporter<dim, NumberType>::CopyData
  {
    using value_type = NumberType;
    using size_type = typename dealii::Vector<value_type>::size_type;


    CopyData(size_type n_local_dofs);


    CopyData(const CopyData& that) = default;


    CopyData& operator=(const CopyData& that) = default;


    std::vector<size_type> local_dof_indices;


    dealii::Vector<value_type> local_rhs;
  };
#endif

}  // namespace internal


FELSPA_NAMESPACE_CLOSE

#include "src/stokes_common.implement.h"

#endif  // _FELSPA_PDE_STOKES_COMMON_H_ //