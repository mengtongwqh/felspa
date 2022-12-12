#ifndef _FELSPA_PDE_STOKES_TRILINOS_H_
#define _FELSPA_PDE_STOKES_TRILINOS_H_

#include <deal.II/base/index_set.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <felspa/pde/stokes_common.h>

FELSPA_NAMESPACE_OPEN

#if defined(FELSPA_HAS_MPI) && defined(DEAL_II_WITH_TRILINOS)

namespace mpi
{
  template <int dim, typename APreconditionerType, typename SPreconditioner>
  class BlockSchurPreconditioner;

  namespace trilinos
  {
    template <int dim>
    class StokesLinearSystem;
  }  // namespace trilinos
}  // namespace mpi


/* ************************************************** */
/**
 * @brief Specialization for \c mpi::trilinos::StokesLinearSystem
 */
/* ************************************************** */
template <int dim>
class StokesSimulator<dim, types::TrilinosScalar,
                      mpi::trilinos::StokesLinearSystem<dim>>
  : public StokesSimulatorBase<dim, types::TrilinosScalar,
                               mpi::trilinos::StokesLinearSystem<dim>>
{
 public:
  using linsys_type = mpi::trilinos::StokesLinearSystem<dim>;
  using value_type = types::TrilinosScalar;
  using base_type = StokesSimulatorBase<dim, value_type, linsys_type>;
  using typename base_type::vector_type;

  using APreconditioner = mpi::trilinos::dealii::PreconditionAMG;
  using SPreconditioner = mpi::trilinos::dealii::PreconditionIC;
  using APreconditionerControl = typename APreconditioner::AdditionalData;
  using SPreconditionerControl = typename SPreconditioner::AdditionalData;
  using SolverAdditionalControl = typename linsys_type::SolverAdditionalControl;

  class Control;


  StokesSimulator(Mesh<dim, value_type>& mesh, unsigned int degree_v,
                  unsigned int degree_p,
                  const std::string& label = "StokesParShared",
                  const MPI_Comm& mpi_communicator = MPI_COMM_WORLD);


  StokesSimulator(Mesh<dim, value_type>& mesh,
                  const dealii::FiniteElement<dim>& fe_v,
                  const dealii::FiniteElement<dim>& fe_p,
                  const std::string& label = "StokesParShared",
                  const MPI_Comm& mpi_communicator = MPI_COMM_WORLD);


  /**
   * @brief Get a modifiable reference to the control parameters
   */
  Control& control();


  /**
   * @brief Get the control parameters
   */
  const Control& get_control() const;


  /**
   * @brief Attach an external control object
   * @param sp_control
   */
  void attach_control(const std::shared_ptr<Control>& sp_control);

  void solve_linear_system(vector_type& soln, vector_type& rhs_vector) override;

  void solve_linear_system(vector_type& soln) override;


 protected:
  /**
   * @brief Allocate the solution vector
   */
  void allocate_solution_vector(vector_type& vect) override;


  /**
   * @brief The block preconditioner
   */
  std::shared_ptr<
    mpi::BlockSchurPreconditioner<dim, APreconditioner, SPreconditioner>>
    ptr_block_preconditioner = nullptr;


  /**
   * @brief MPI Communicator
   */
  MPI_Comm mpi_communicator;


 private:
  /**
   * @brief Helper function for construction.
   */
  void do_construct();
};


/* ************************************************** */
/**
 * @brief Stokes control parameters
 */
/* ************************************************** */
template <int dim>
class StokesSimulator<dim, types::TrilinosScalar,
                      mpi::trilinos::StokesLinearSystem<dim>>::Control
  : public StokesControlBase<value_type>
{
 public:
  Control();

  std::shared_ptr<APreconditionerControl> ptr_precond_A;

  std::shared_ptr<SPreconditionerControl> ptr_precond_S;

  std::shared_ptr<SolverAdditionalControl> ptr_solver_additional_control;
};


/* ************************************************** */
/**
 * @brief Assembler
 */
/* ************************************************** */
template <int dim>
class StokesAssembler<mpi::trilinos::StokesLinearSystem<dim>>
  : public StokesAssemblerBase<mpi::trilinos::StokesLinearSystem<dim>>
{
 public:
  constexpr static int dimension = dim;

  using linsys_type = mpi::trilinos::StokesLinearSystem<dim>;
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


  explicit StokesAssembler(linsys_type& linsys,
                           bool construct_mapping_adhoc = false);


 protected:
  void local_assembly(const active_cell_iterator& cell, ScratchData& scratch,
                      CopyData& copy) override;
};
#endif  // (FELSPA_HAS_MPI)&& (DEAL_II_WITH_TRILINOS)


#ifdef FELSPA_HAS_MPI
/* --------------- */
namespace mpi
/* --------------- */
{
  /* ************************************************** */
  /**
   * @brief Declaration for the global BlockSchurPreconditioner.
   *
   * @tparam APreconditionerType
   * @tparam SPreconditionerType
   */
  /* ************************************************** */
  template <int dim, typename APreconditionerType, typename SPreconditionerType>
  class BlockSchurPreconditioner;


#ifdef DEAL_II_WITH_TRILINOS

  /**
   * Specialization of the BlockSchurPreconditioner
   */
  template <int dim, typename SPreconditionerType>
  class BlockSchurPreconditioner<dim, trilinos::dealii::PreconditionAMG,
                                 SPreconditionerType>
    : public BlockSchurPreconditionerBase<
        trilinos::dealii::PreconditionAMG, SPreconditionerType,
        trilinos::dealii::BlockSparseMatrix, trilinos::dealii::BlockVector>
  {
   public:
    using A_preconditioner_type = trilinos::dealii::PreconditionAMG;
    using S_preconditioner_type = SPreconditionerType;
    using matrix_type = trilinos::dealii::BlockSparseMatrix;
    using vector_type = trilinos::dealii::BlockVector;
    using matrix_block_type = typename matrix_type::BlockType;
    using base_type =
      BlockSchurPreconditionerBase<A_preconditioner_type, S_preconditioner_type,
                                   matrix_type, vector_type>;
    using this_type = BlockSchurPreconditioner<dim, A_preconditioner_type,
                                               S_preconditioner_type>;

    using A_preconditioner_control =
      typename A_preconditioner_type::AdditionalData;
    using S_preconditioner_control =
      typename S_preconditioner_type::AdditionalData;

    /**
     * Constructor
     */
    BlockSchurPreconditioner(
      const matrix_type& stokes_matrix,
      const matrix_type& stokes_precond_matrix,
      const ::dealii::DoFHandler<dim>& dof_handler,
      const std::shared_ptr<A_preconditioner_control>& sp_A_control,
      const std::shared_ptr<S_preconditioner_control>& sp_S_control,
      MPI_Comm mpi_communicator = MPI_COMM_WORLD);

    /**
     * @brief Reinitialize the preconditioner.
     */
    void reinitialize() override;


    /**
     * @brief Get const reference to the preconditioner matrix
     */
    const matrix_type& get_preconditioner_matrix() const;


   protected:
    void resize_tmp_vector();


   private:
    /**
     * @brief Stokes preconditioning matrix
     * The inverse of \f[ A & O \\ B & S \f]
     */
    const dealii::SmartPointer<const matrix_type, this_type>
      ptr_preconditioner_matrix;

    /**
     * Pointer to the DoFHandler
     */
    const dealii::SmartPointer<const dealii::DoFHandler<dim>> ptr_dof_handler;

    /**
     * @brief Pointer to preconditioner for inverting A
     */
    const std::unique_ptr<A_preconditioner_type> ptr_A_preconditioner;

    /**
     * @brief  Pointer to preconditioner for inverting
     * (approximate) Schur inverse
     */
    const std::unique_ptr<S_preconditioner_type> ptr_S_preconditioner;

    /**
     * @brief Control parameters for the A-preconditioner.
     */
    const std::shared_ptr<A_preconditioner_control> ptr_control_A;

    /**
     * @brief Control parameters for the S-preconditioner.
     */
    const std::shared_ptr<S_preconditioner_control> ptr_control_S;
  };


  /* --------------- */
  namespace trilinos
  /* --------------- */
  {
    /* ************************************************** */
    /**
     * StokesLinearSystem using the Trilinos
     * block vector and sparse matrix.
     */
    /* ************************************************** */
    template <int dim>
    class StokesLinearSystem
      : public LinearSystemBase<dim, dealii::BlockSparseMatrix,
                                dealii::BlockVector>
    {
      friend class StokesAssemblerBase<StokesLinearSystem<dim>>;

     public:
      using base_type =
        LinearSystemBase<dim, dealii::BlockSparseMatrix, dealii::BlockVector>;
      using vector_type = dealii::BlockVector;
      using matrix_type = dealii::BlockSparseMatrix;
      using vector_block_type = typename vector_type::BlockType;
      using matrix_block_type = typename matrix_type::BlockType;
      using number_type = typename vector_type::value_type;
      using value_type = number_type;
      using size_type = typename vector_type::size_type;
      using constraints_type = ::dealii::AffineConstraints<number_type>;
      using SolverType = ::dealii::SolverGMRES<vector_type>;
      // using SolverType = ::dealii::SolverMinRes<vector_type>;
      using SolverAdditionalControl = typename SolverType::AdditionalData;

      constexpr const static int spacedim = dim;
      constexpr const static int dimension = dim;

      class DiagonalScalingOp
        : public ::FELSPA_NAMESPACE::internal::StokesScalingOperator<
            matrix_type>
      {
       public:
        // void initialize(const matrix_type& matrix);

        void allocate(const std::vector<::dealii::IndexSet>& locally_owned_dofs,
                      MPI_Comm mpi_comm);
      };

      /**
       * @brief Constructor
       */
      explicit StokesLinearSystem(const ::dealii::DoFHandler<dim>& dofh,
                                  MPI_Comm mpi_comm = MPI_COMM_WORLD);

      /**
       * @brief Constructor
       */
      StokesLinearSystem(const ::dealii::DoFHandler<dim>& dofh,
                         const ::dealii::Mapping<dim>& mapping,
                         MPI_Comm mpi_comm = MPI_COMM_WORLD);

      /**
       * Actions to be performed when mesh update is detected.
       * This will set recount the dofs
       */
      void upon_mesh_update();


      /**
       * @brief Count locally owned/relevant dofs
       */
      void count_dofs();


      /**
       * zero out the linear system so as to be ready for assembly
       */
      void zero_out(bool zero_lhs, bool zero_rhs, bool zero_preconditioner);

      /**
       * Get preconditioner matrix
       */
      const matrix_type& get_preconditioner_matrix() const;

      /**
       * Apply pressure scaling to system matrix
       * and preconditioner matrix.
       */
      void apply_pressure_scaling(value_type ref_viscosity,
                                  value_type ref_length);

      /**
       * Attach the preconditioner object by pointing
       * storing a polymorphic pointer.
       * All control of the preconditioner
       */
      void set_block_preconditioner(
        const std::shared_ptr<
          MatrixPreconditionerBase<matrix_type, vector_type>>& preconditioner);

      /**
       * Set the mpi communicator object
       */
      void set_mpi_communicator(const MPI_Comm& mpi_communicator);

      /**
       * @brief Additional control parameters for SolverGMRES
       */
      void set_additional_control(
        const std::shared_ptr<SolverAdditionalControl>& ptr_additional_control);

      /**
       * Allocate the sparsematrix/rhs and the
       * setup the constraints.
       */
      void setup_constraints_and_system(
        const BCBookKeeper<dim, value_type>& bcs);

      /**
       * @brief Inform the linear system that
       * the constraints need to be udpated.
       * Called upon every mesh update.
       */
      void flag_constraints_for_update();

      /**
       * @brief Return the status of the update constraints flag.
       */
      bool constraints_are_updated() const;

      /**
       * @brief Obtain the \c IndexSet for each block.
       */
      const std::vector<::dealii::IndexSet>& get_owned_dofs_per_block() const;

      /**
       * @brief Solve the linear system
       */
      void solve(vector_type& soln, vector_type& rhs,
                 ::dealii::SolverControl& solver_control);


      void solve(vector_type& soln, ::dealii::SolverControl& solver_control);


     protected:
      /**
       * Setup constraints from hanging nodes and
       * boundary conditions.
       */
      void setup_constraints(const BCBookKeeper<dim, value_type>& bcs);

      /**
       * Allocate linear system and the preconditioner matrix
       */
      void make_sparsity_pattern();

      /**
       * Precoditioner matrix
       */
      matrix_type preconditioner_matrix;

      /**
       * Number of dofs per velocity and pressure block.
       */
      std::vector<size_type> ndofs_per_block;

      /**
       * Locally owned dof indices
       * per velocity and pressure block
       */
      std::vector<::dealii::IndexSet> owned_dofs_per_block;

      /**
       * Locally relevant dof indices
       * per velocity and pressure block
       */
      std::vector<::dealii::IndexSet> relevant_dofs_per_block;

      /**
       * Locally owned dof indices
       */
      ::dealii::IndexSet locally_owned_dofs;

      /**
       * Locally relevant dof indices
       */
      ::dealii::IndexSet locally_relevant_dofs;

      /**
       * Pointer to the block precondioner.
       */
      std::shared_ptr<MatrixPreconditionerBase<matrix_type, vector_type>>
        ptr_block_preconditioner = nullptr;


      /**
       * @brief Additional control parameters for solver
       */
      std::shared_ptr<SolverAdditionalControl> ptr_additional_control = nullptr;


      /**
       * @brief diagonal scaling operator
       */
      std::unique_ptr<DiagonalScalingOp> ptr_scaling_operator = nullptr;


      /**
       * Update constraints flag.
       */
      bool constraints_updated = false;

      /**
       * If flagged, the preconditioner will be reinitialized
       * in the \c solve() function.
       */
      bool preconditioner_requires_reinit = true;

      /**
       * @brief  MPI Communicator
       */
      MPI_Comm mpi_communicator;
    };

  }     // namespace trilinos
#endif  // DEAL_II_WITH_TRILINOS
}  // namespace mpi
#endif  // FELSPA_HAS_MPI

FELSPA_NAMESPACE_CLOSE

#include "src/stokes_trilinos.implement.h"

#endif  // _FELSPA_PDE_STOKES_TRILINOS_H_
