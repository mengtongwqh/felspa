#ifndef _FELSPA_LINEAR_ALGEBRA_LINEAR_SYSTEM_H_
#define _FELSPA_LINEAR_ALGEBRA_LINEAR_SYSTEM_H_

#include <deal.II/base/subscriptor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>
#include <felspa/base/exceptions.h>
#include <felspa/base/felspa_config.h>
#include <felspa/base/io.h>
#include <felspa/base/types.h>

#include <fstream>
#include <memory>

FELSPA_NAMESPACE_OPEN


#ifdef FELSPA_HAS_MPI
/**
 * Importing definitions from dealii::TrilinosWrappers
 * and dealii::PETScWrappers namespaces
 * into our mpi namespace.
 */
namespace mpi
{
#ifdef DEAL_II_WITH_TRILINOS
  namespace trilinos
  {
    namespace dealii
    {
      using namespace ::dealii::TrilinosWrappers;
      using namespace ::dealii::TrilinosWrappers::MPI;
    }   // namespace dealii
  }     // namespace trilinos
#endif  // DEAL_II_WITH_TRILINOS

#ifdef DEAL_II_WITH_PETSC
  namespace petsc
  {
    namespace dealii
    {
      using namespace ::dealii::PETScWrappers;
    }   // namespace dealii
  }     // namespace petsc
#endif  // DEAL_II_WITH_PETSC //
}  // namespace mpi
#endif  // FELSPA_HAS_MPI


// forward declaration for AssemblerBase
template <typename Linsys>
class AssemblerBase;


/* ************************************************** */
/**
 * Block count is different for * matrix and rhs vector.
 * @ingroup exceptions
 */
/* ************************************************** */
DECL_EXCEPT_2(ExcBlockMismatch,
              "Matrix and RHS vector has non-matching block counts: "
                << arg1 << " vs " << arg2,
              unsigned int, unsigned int);


/* ************************************************** */
/**
 * \class LinearSystemBase
 * \tparam dim, SparsityType, MatrixType, VectorType
 * This is a very general blueprint for a linear system.
 * It outlines the basic members needed to describe
 * the linear system as well as the basic operations
 * that can be performed, including setting up/freeing memory,
 * and solving the linear system. The assembly of the system
 * are taken care of by classes derived from \c AssemblerBase object
 */
/* ************************************************** */
template <int dim, typename MatrixType, typename VectorType>
class LinearSystemBase : public dealii::Subscriptor
{
  /**
   * Grant access to all assembler type classes
   */
  template <typename LinsysType>
  friend class AssemblerBase;

  /**
   * FESimulator will have access to members of this object.
   * Especially for access to constraints object.
   */
  template <int spacedim, typename FEType, typename LinsysType,
            typename TempoIntegrator>
  friend class FESimulator;

 public:
  static_assert(std::is_same<typename MatrixType::value_type,
                             typename VectorType::value_type>::value,
                "matrix and vector must contain the same number type.");

  /**
   * Type of RHS or solution vector
   */
  using vector_type = VectorType;

  /**
   * Type of LHS matrix
   */
  using matrix_type = MatrixType;

  /**
   * Type of Floating Point Number used
   */
  using value_type = typename VectorType::value_type;

  /**
   * Size type for the linear system
   */
  using size_type = typename VectorType::size_type;

  /**
   * Type of the constraint object
   */
  using constraints_type = dealii::AffineConstraints<value_type>;


  constexpr const static int spacedim = dim;
  constexpr const static int dimension = dim;


  /* ------------------------------- */
  /** \name Query Object Information */
  /* ------------------------------- */
  //@{
  /**
   * Get constraints
   */
  const constraints_type& get_constraints() const { return constraints; }

  /**
   * Get a \c const reference to \c dof_handler
   */
  const dealii::DoFHandler<dim>& get_dof_handler() const;

  /**
   * Get mapping object
   */
  const dealii::Mapping<dim>& get_mapping() const;

  /**
   * get matrix
   */
  const matrix_type& get_matrix() const { return matrix; }

  /**
   * get rhs
   */
  const vector_type& get_rhs() const { return rhs; }


  /**
   * return the number of unknowns in the linear system
   */
  size_type size() const;


  /**
   * Test if the linear system is empty.
   */
  bool empty() const;


  /**
   * \brief test if the linear system has been allocated.
   * \return \p true if the system is not populated
   */
  bool is_populated() const;


  /**
   * Worst case O(N) operation
   * Only to be used in debug mode for internal checks
   */
  bool is_all_zero() const;
  //@}


  /* ----------------------------------- */
  /** \name Linear System Setup/Solution */
  /* ----------------------------------- */
  //@{
  /**
   * Free all memories held by matrix and rhs.
   * However the sparsity pattern is left untouched.
   */
  void clear();


  /**
   * Zero out all entries in the LHS matrix and RHS vector
   * But leave the sparsity pattern alone.
   */
  void zero_out(bool zero_lhs, bool zero_rhs);


  /**
   * \brief The function to solve the linear system.
   * We leave this as a pure virtual function so
   * that different simulator can inherit this class
   * and decide upon a suitable linear solver/preconditioner
   * to generate solution to the system
   */
  // void solve(VectorType& solution, dealii::SolverControl& control) const;


  /**
   * Solve the linear system with an alternative RHS.
   */
  // virtual void solve(VectorType& solution, const VectorType& rhs,
  //                    dealii::SolverControl&) const = 0;
  //@}


 protected:
  /* ----------------------------- */
  /** \name Basic Object Behaviors */
  /* ----------------------------- */
  //@{
  /**
   * Constructor
   */
  explicit LinearSystemBase(const dealii::DoFHandler<dim>& dof_handler);

  /**
   * Constructor
   */
  LinearSystemBase(const dealii::DoFHandler<dim>& dof_handler,
                   const dealii::Mapping<dim>& mapping);

  /**
   * Destructor
   */
  virtual ~LinearSystemBase() = default;
//@}


/**
 * pointer to a const \p DoFHandler object.
 * Useful for allocating the space for linear system
 */
#ifdef DEBUG
  dealii::SmartPointer<const dealii::DoFHandler<dim>>
#else
  const dealii::DoFHandler<dim>*
#endif
    ptr_dof_handler;


  /**
   * pointer to a const \p Mapping object
   */
#ifdef DEBUG
  dealii::SmartPointer<const dealii::Mapping<dim>>
#else
  const dealii::Mapping<dim>*
#endif
    ptr_mapping;


  /**
   * Linear constraint object
   */
  constraints_type constraints;


  /**
   * Left Hand Side (LHS) matrix
   */
  MatrixType matrix;


  /**
   * Right Hand Side (RHS) vector
   */
  VectorType rhs;

  /**
   * signal if the linear system has been allocated
   */
  bool populated;
};


/* ************************************************** */
/**
 * \class LinearSystem
 * \tparam dim
 * Linear system class using the ordinary \p deal.II
 * \c SparseMatrix and \c Vector. Designed to be
 * incorporated into scalar FE simulators.
 */
/* ************************************************** */
template <int dim, typename NumberType = types::DoubleType>
class LinearSystem
  : public LinearSystemBase<dim, dealii::SparseMatrix<NumberType>,
                            dealii::Vector<NumberType>>
{
 public:
  using value_type = NumberType;
  using base_type = LinearSystemBase<dim, dealii::SparseMatrix<NumberType>,
                                     dealii::Vector<NumberType>>;
  using sparsity_type = dealii::SparsityPattern;
  using matrix_type = dealii::SparseMatrix<NumberType>;
  using vector_type = dealii::Vector<NumberType>;
  using typename base_type::constraints_type;

  constexpr const static int spacedim = dim;
  constexpr const static int dimension = dim;

  explicit LinearSystem(const dealii::DoFHandler<dim>&);

  LinearSystem(const dealii::DoFHandler<dim>&, const dealii::Mapping<dim>&);

  void clear();

  void print_sparsity(ExportFile&) const;

  const sparsity_type& get_sparsity_pattern() const;


 protected:
  dealii::SparsityPattern sparsity_pattern;
};  // class LinearSystem<dim>


/* ************************************************** */
/**
 * \class BlockLinearSystem
 * \tparam dim
 * Linear system class using the \p deal.II
 * \c BlockSparseMatrix and \c BlockVector.
 * Designed to be incorporated into vector FE simulators.
 */
/* ************************************************** */
template <int dim, typename NumberType = types::DoubleType>
class BlockLinearSystem
  : public LinearSystemBase<dim, dealii::BlockSparseMatrix<NumberType>,
                            dealii::BlockVector<NumberType>>
{
 public:
  using base_type = LinearSystemBase<dim, dealii::BlockSparseMatrix<NumberType>,
                                     dealii::BlockVector<NumberType>>;
  using sparsity_type = dealii::BlockSparsityPattern;
  using matrix_type = dealii::BlockSparseMatrix<NumberType>;
  using vector_type = dealii::BlockVector<NumberType>;
  using matrix_block_type = typename matrix_type::BlockType;
  using vector_block_type = typename vector_type::BlockType;
  using value_type = NumberType;
  using size_type = typename vector_type::size_type;
  using typename base_type::constraints_type;

  constexpr const static int spacedim = dim;
  constexpr const static int dimension = dim;

  /* ---------------------------- */
  /** \name Basic Object Handling */
  /* ---------------------------- */
  //@{
  /**
   * Constructor
   */
  explicit BlockLinearSystem(const dealii::DoFHandler<dim>&);

  /**
   * Constructor
   */
  BlockLinearSystem(const dealii::DoFHandler<dim>& dofh,
                    const dealii::Mapping<dim>& mapping);
  //@}


  /**
   * release all memory held
   */
  void clear();


  /**
   *  Obtain reference to sparsity pattern
   */
  const sparsity_type& get_sparsity_pattern() const;


  /**
   * @brief  Print sparsity pattern to file
   */
  void print_sparsity(ExportFile&) const;


  /**
   * Count how many dofs are there per block/component.
   * Some simulator may want to a block structure that is different
   * from the default, in which case the user can override this method
   * and provide customized way to compute block/component count.
   */
  virtual void count_dofs(bool compute_ndofs_component = true,
                          bool compute_ndofs_block = true);


  /* --------------------------------------------- */
  /** \name Get ndof count for component and block */
  /* --------------------------------------------- */
  //@{
  const std::vector<size_type>& get_component_ndofs() const;

  size_type get_component_ndofs(unsigned int component) const;

  const std::vector<size_type>& get_block_ndofs() const;

  size_type get_block_ndofs(unsigned int block) const;
  //@}


 protected:
  /**
   * Sparsity pattern.
   */
  dealii::BlockSparsityPattern sparsity_pattern;

  /**
   * Number of dofs per component
   */
  std::vector<size_type> ndofs_per_component;

  /**
   * Number of dofs per block
   */
  std::vector<size_type> ndofs_per_block;
};  // class BlockLinearSystem<dim>


FELSPA_NAMESPACE_CLOSE

/* ------- IMPLEMENTATIONS ------- */
#include "src/linear_system.implement.h"
/* ------------------------------- */

#endif  //_FELSPA_LINEAR_ALGEBRA_LINEAR_SYSTEM_H_
