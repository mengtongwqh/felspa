#ifndef _FELSPA_PDE_STOKES_DECOMPOSED_H_
#define _FELSPA_PDE_STOKES_DECOMPOSED_H_

#include <deal.II/lac/sparse_direct.h>
#include <felspa/linear_algebra/ilu.h>
#include <felspa/pde/linear_systems.h>
#include <felspa/pde/stokes_common.h>

FELSPA_NAMESPACE_OPEN

#define FELSPA_STOKES_USE_CUSTOM_ILU

/* ------------------- */
namespace internal
/* ------------------ */
{
#ifdef FELSPA_STOKES_USE_CUSTOM_ILU
  using ::felspa::SparseILU;
#else
  using dealii::SparseILU;  // use deal.II sparse ILU
#endif  // USE_FELSPA_SPARSEILU //

  template <int dim, typename NumberType>
  struct APreconditioner;

  template <int dim, typename NumberType>
  struct SPreconditioner;

  /* ********************************** */
  /**
   * Type of preconditioner to be used
   * when computing A^{-1} in Schur complement
   */
  /* *********************************** */
  template <typename NumberType>
  struct APreconditioner<2, NumberType>
  {
    using type = SparseILU<NumberType>;
  };

  template <typename NumberType>
  struct APreconditioner<3, NumberType>
  {
    using type = SparseILU<NumberType>;
  };

  template <typename NumberType>
  struct SPreconditioner<2, NumberType>
  {
    using type = SparseILU<NumberType>;
  };

  template <typename NumberType>
  struct SPreconditioner<3, NumberType>
  {
    using type = SparseILU<NumberType>;
  };
}  // namespace internal


/* ************************************************** */
/**
 * @brief Block precondition the Stokes matrix
 */
/* ************************************************** */
template <int dim, typename NumberType>
class BlockSchurPreconditioner
  : public BlockSchurPreconditionerBase<
      typename internal::APreconditioner<dim, NumberType>::type,
      typename internal::SPreconditioner<dim, NumberType>::type,
      dealii::BlockSparseMatrix<NumberType>, dealii::BlockVector<NumberType>>
{
 public:
  using value_type = NumberType;
  using number_type = NumberType;
  using matrix_type = dealii::BlockSparseMatrix<NumberType>;
  using matrix_block_type = typename matrix_type::BlockType;
  using vector_type = dealii::BlockVector<NumberType>;
  using S_preconditioner_type =
    typename internal::SPreconditioner<dim, NumberType>::type;
  using S_preconditioner_control =
    typename S_preconditioner_type::AdditionalData;

  using A_preconditioner_type =
    typename internal::APreconditioner<dim, NumberType>::type;
  using A_preconditioner_control =
    typename A_preconditioner_type::AdditionalData;
  using A_inverse_type = InverseMatrixCG<matrix_type, A_preconditioner_type>;

  using this_type = BlockSchurPreconditioner<dim, number_type>;
  using base_type =
    BlockSchurPreconditionerBase<A_preconditioner_type, S_preconditioner_type,
                                 matrix_type, vector_type>;

  BlockSchurPreconditioner(
    const matrix_type& stokes_matrix,
    const matrix_type& stokes_precond_matrix,
    A_preconditioner_type& A_precond,
    S_preconditioner_type& S_precond,
    const std::shared_ptr<A_preconditioner_control>& sp_A_control,
    const std::shared_ptr<S_preconditioner_control>& sp_S_control);

  void reinitialize() override;

  const matrix_type& get_preconditioner_matrix() const;

 protected:
  const dealii::SmartPointer<const matrix_type, this_type>
    ptr_preconditioner_matrix;

  const dealii::SmartPointer<A_preconditioner_type> ptr_A_preconditioner;

  const dealii::SmartPointer<S_preconditioner_type> ptr_S_preconditioner;

  const std::shared_ptr<A_preconditioner_control> ptr_A_control;

  const std::shared_ptr<S_preconditioner_control> ptr_S_control;
};


/* ************************************************** */
/**
 * Stokes linear system.
 * Apart from the LHS and RHS, we also assemble the
 * mass matrix in pressure FE space. An approximate inverse
 * of the pressure mass matrix will be used as a preconditioner
 * for solving the Schur complement.
 */
/* ************************************************** */
template <int dim, typename NumberType>
class StokesLinearSystem : public MixedVPLinearSystem<dim, NumberType>
{
  friend class StokesAssemblerBase<StokesLinearSystem<dim, NumberType>>;

 public:
  using value_type = NumberType;
  using base_type = MixedVPLinearSystem<dim, NumberType>;
  using APreconditioner =
    typename internal::APreconditioner<dim, NumberType>::type;
  using SPreconditioner =
    typename internal::SPreconditioner<dim, NumberType>::type;
  using typename base_type::matrix_block_type;
  using typename base_type::matrix_type;
  using typename base_type::sparsity_type;
  using typename base_type::vector_block_type;
  using typename base_type::vector_type;
  using APreconditionerControl =
    typename internal::APreconditioner<dim, NumberType>::type::AdditionalData;
  using SPreconditionerControl =
    typename internal::SPreconditioner<dim, NumberType>::type::AdditionalData;


  class DiagonalScalingOp : public internal::StokesScalingOperator<matrix_type>
  {
   public:
    using typename internal::StokesScalingOperator<matrix_type>::size_type;
    void allocate(const matrix_type& matrix);
  };

  /**
   * @brief Constructor
   */
  StokesLinearSystem(const dealii::DoFHandler<dim>& dofh);


  /**
   * @brief Constructor
   */
  StokesLinearSystem(const dealii::DoFHandler<dim>& dofh,
                     const dealii::Mapping<dim>& mapping);


  /**
   * Apply pressure scaling for the linear system matrix
   * and the preconditioner matrix.
   */
  void apply_pressure_scaling(value_type ref_viscosity, value_type ref_length);


  /**
   * Scale the matrix, preconditioner, solution and rhs vector
   */
  void pre_solve_scaling(vector_type& soln, vector_type& rhs,
                         bool scale_matrix);


  /**
   * Scale the solution vector to true solution
   */
  void post_solve_scaling(vector_type& soln);


  /**
   * Setting entries to zero, including preconditioner
   */
  void zero_out(bool zero_lhs, bool zero_rhs, bool zero_preconditioner);


  /**
   * Recount dofs and set proper update flags.
   */
  void upon_mesh_update();


  /**
   * @brief  Count number of dofs per each component/block.
   * The dim + 1 components are dim-dimensional velocity + 1-dimensional
   * pressure.\n
   * The 2 blocks are velocity and pressure block.
   */
  void count_dofs(bool count_component = true,
                  bool count_block = true) override;


  /**
   * @brief Allocate linear system
   */
  void populate_system_from_dofs();


  /**
   * @brief Set up the constraints and system object.
   * This combines \c setup_constraints() and
   * \c populate_system_from_dofs() and utilizes multitasking.
   */
  void setup_constraints_and_system(const BCBookKeeper<dim, NumberType>& bc);


  /**
   * @brief Solve the linear system
   */
  void solve(vector_type& soln, vector_type& rhs,
             dealii::SolverControl& solver_control);


  void solve(vector_type& soln, dealii::SolverControl& solver_control);


  /**
   * @brief Attach control structure for inner and outer preconditioners.
   */
  void set_control_parameters(
    const std::shared_ptr<APreconditionerControl>& ptr_A_precond_control,
    const std::shared_ptr<SPreconditionerControl>& ptr_S_precond_control,
    const StokesSolutionMethod& method);


  /**
   * @brief Getter for the preconditioner matrix
   */
  const matrix_type& get_preconditioner_matrix() const;


 protected:
  /**
   * @brief Solve the system using GMRES block preconditioning
   */
  void solve_gmres(vector_type& soln, vector_type& rhs,
                   dealii::SolverControl& solver_control,
                   bool apply_scaling_to_matrix = true);

  /**
   * @brief Solve the system using CG Schur complement reduction
   */
  void solve_cg(vector_type& soln, vector_type& rhs,
                dealii::SolverControl& solver_control,
                bool apply_scaling_to_matrix = true);

  /**
   * @brief Run both CG and GMRES and compare performance
   */
  void solve_compare_test(vector_type& soln, vector_type& rhs,
                          dealii::SolverControl& solver_control);

  /**
   * @brief Construct the sparsity for preconditioning matrix
   */
  void make_preconditioner_sparsity();


  /**
   * @brief Setup constraints for boundary conditions
   */
  void setup_bc_constraints(const BCBookKeeper<dim, value_type>& bcs) override;


  /**
   * @brief Sparsity for matrix preconditioner
   */
  sparsity_type preconditioner_sparsity;


  /**
   * @brief Outer preconditioner matrix
   */
  matrix_type preconditioner_matrix;


  /**
   * @brief (velocity-space) Inner preconditioner type
   */
  std::unique_ptr<APreconditioner> ptr_A_preconditioner;


  /**
   * @brief (pressure-space) Outer preconditioner type
   */
  std::unique_ptr<SPreconditioner> ptr_S_preconditioner;


  /**
   * Control parameters for the preconditioner of A
   */
  std::shared_ptr<APreconditionerControl> ptr_A_preconditioner_control =
    nullptr;


  /**
   * Control parameters for the preconditioner of the mass matrix
   * in the pressure approximation space.
   */
  std::shared_ptr<SPreconditionerControl> ptr_S_preconditioner_control =
    nullptr;


  /**
   * @brief diagonal scaling operator
   */
  std::unique_ptr<DiagonalScalingOp> ptr_scaling_operator;

  /**
   * @brief Solution method, Fully coupled or decomposed
   */
  const StokesSolutionMethod* ptr_solution_method;
};


/* ************************************************** */
/**
 * @brief Specialization for \c StokesLinearSystem
 */
/* ************************************************** */
template <int dim, typename NumberType>
class StokesSimulator<dim, NumberType, StokesLinearSystem<dim, NumberType>>
  : public StokesSimulatorBase<dim, NumberType,
                               StokesLinearSystem<dim, NumberType>>
{
 public:
  using value_type = NumberType;
  using linsys_type = StokesLinearSystem<dim, NumberType>;
  using base_type = StokesSimulatorBase<dim, NumberType, linsys_type>;
  using typename StokesSimulatorBase<dim, NumberType, linsys_type>::vector_type;
  class Control;

  /**
   * @brief Constructor assuming \c dealii::FE_Q<dim>
   */
  StokesSimulator(Mesh<dim, value_type>& mesh, unsigned int degree_v,
                  unsigned int degree_p,
                  const std::string& label = "StokesSerial");


  StokesSimulator(Mesh<dim, value_type>& mesh,
                  const dealii::FiniteElement<dim>& fe_v,
                  const dealii::FiniteElement<dim>& fe_p,
                  const std::string& label = "StokesSerial");

  /**
   * @brief Set the control parameters.
   */
  void attach_control(const std::shared_ptr<Control>& sp_control);


  /**
   * @brief Get a modifiable reference to the control parameters
   */
  Control& control();


  /**
   * @brief Get the control parameters
   */
  const Control& get_control() const;


 protected:
  /**
   * @brief Allocate the solution vector
   */
  void allocate_solution_vector(vector_type& vect) override;
};


/* ************************************************** */
/**
 * @brief Control parameters
 */
/* ************************************************** */
template <int dim, typename NumberType>
class StokesSimulator<dim, NumberType,
                      StokesLinearSystem<dim, NumberType>>::Control
  : public StokesControlBase<value_type>
{
 public:
  using value_type = NumberType;
  using A_preconditioner_control_type =
    typename internal::APreconditioner<dim, value_type>::type::AdditionalData;
  using S_preconditioner_control_type =
    typename internal::SPreconditioner<dim, value_type>::type::AdditionalData;

  /**
   * @brief Default construction
   */
  Control();

  /**
   * Level-of-fill of SparseILU preconditioner for A.
   */
  std::shared_ptr<A_preconditioner_control_type> ptr_A_preconditioner;

  /**
   * Level-of-fill of SparseILU preconditioner for pressure block.
   */
  std::shared_ptr<S_preconditioner_control_type> ptr_S_preconditioner;

  /**
   * @brief Solution method for the Stokes sytem
   */
  StokesSolutionMethod solution_method;

#ifdef FELSPA_STOKES_USE_CUSTOM_ILU
  /**
   * Set the level of fill object
   */
  void set_level_of_fill(unsigned int A_level_of_fill = 1,
                         unsigned int S_level_of_fill = 0);
#endif  // FELSPA_STOKES_USE_CUSTOM_ILU
};      // class StokesControl


template <int dim, typename NumberType>
class StokesAssembler<StokesLinearSystem<dim, NumberType>>
  : public StokesAssemblerBase<StokesLinearSystem<dim, NumberType>>
{
 public:
  using linsys_type = StokesLinearSystem<dim, NumberType>;
  using base_type = StokesAssemblerBase<linsys_type>;
  using typename base_type::matrix_type;
  using typename base_type::source_type;
  using typename base_type::value_type;
  using typename base_type::vector_block_type;
  using typename base_type::vector_type;
  using active_cell_iterator =
    typename dealii::DoFHandler<dim>::active_cell_iterator;
  using typename base_type::CopyData;
  using typename base_type::ScratchData;

  constexpr static int dimension = dim;

  /**
   * @brief Construct a new Stokes Assembler object
   */
  StokesAssembler(linsys_type& linsys, bool construct_mapping_adhoc = false);

 protected:
  void local_assembly(const active_cell_iterator& cell, ScratchData& scratch,
                      CopyData& copy) override;
};

FELSPA_NAMESPACE_CLOSE
#endif  // _FELSPA_PDE_STOKES_DECOMPOSED_H_ //
