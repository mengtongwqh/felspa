#ifndef _FELSPA_LINEAR_ALGEBRA_LINEAR_OPERATOR_H_
#define _FELSPA_LINEAR_ALGEBRA_LINEAR_OPERATOR_H_

#include <deal.II/base/index_set.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <felspa/base/exceptions.h>
#include <felspa/base/felspa_config.h>
#include <felspa/base/log.h>
#include <felspa/base/types.h>

FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/**
 * Linear operator for computing matrix inverse.
 * This is basically a simplistic reimplementation of
 * deal.II inverse_operator
 */
/* ************************************************** */
template <typename MatrixType, typename PreconditionerType>
class InverseMatrixCG : public dealii::Subscriptor
{
 public:
  using this_type = InverseMatrixCG<MatrixType, PreconditionerType>;
  using matrix_type = MatrixType;
  using value_type = typename matrix_type::value_type;

  /**
   * @brief Default construct a new Inverse Matrix ob
   *
   */
  InverseMatrixCG();


  /**
   * Constructor.
   * Solver will be controlled by a \c ReductionControl object.
   */
  InverseMatrixCG(const MatrixType& matrix,
                  const PreconditionerType& preconditioner,
                  const value_type tol = 1.0e-12,
                  const value_type reduction = 1.0e-12);


  void initialize(const MatrixType& matrix,
                  const PreconditionerType& preconditioner,
                  const value_type tol = 1.0e-12,
                  const value_type reduction = 1.0e-12);

  /**
   * Function for carrying out matrix-vector multiplication.
   * Under the hood the function is solving a linear system.
   */
  template <typename VectorType>
  void vmult(VectorType& dst, const VectorType& src) const;


  /**
   * @brief Get the control structure of the solver
   */
  const dealii::ReductionControl& get_control() const;


  dealii::ReductionControl& control();


 private:
  /**
   * Matrix to be inverted
   */
  dealii::SmartPointer<const MatrixType, this_type> ptr_matrix;

  /**
   * Preconditioner used for accelarating
   * the convergence of linear solver.
   */
  dealii::SmartPointer<const PreconditionerType, this_type> ptr_preconditioner;

  /**
   * @brief Solver control
   */
  std::unique_ptr<dealii::ReductionControl> ptr_control;
};


/* ************************************************** */
/**
 * Schur complement operator for a saddle point problem.
 * The saddle point problem takes the following form:
 * \f[
 * \begin{bmatrix}
 * A & B \\ B^\top & O
 * \end{bmatrix}
 * \f]
 * and the Schur Complement \f${S}\f$ of this system is defined as
 * \f${S = B^\top A^{-1} B}\f$
 */
/* ************************************************** */
template <typename BlockMatrixType, typename InverseMatrixType>
class SaddlePointSchurComplement
{
 public:
  using this_type =
    SaddlePointSchurComplement<BlockMatrixType, InverseMatrixType>;

  /**
   * Constructor.
   */
  SaddlePointSchurComplement(const BlockMatrixType& matrix,
                             const InverseMatrixType& A_inverse);


  /**
   * Multiplication with, or action upon a vector.
   */
  template <typename VectorType>
  void vmult(VectorType& dst, const VectorType& src) const;


 private:
  /**
   * Matrix to the block Stokes system.
   */
  dealii::SmartPointer<const BlockMatrixType, this_type> ptr_matrix;


  /**
   * Pointer to the inverse matrix.
   */
  dealii::SmartPointer<const InverseMatrixType, this_type> ptr_A_inverse;
};


/* ************************************************** */
/**
 * The base class for block Schur preconditioner
 */
/* ************************************************** */
template <typename MatrixType, typename VectorType>
struct MatrixPreconditionerBase : public dealii::Subscriptor
{
 public:
  using matrix_type = MatrixType;
  using vector_type = VectorType;
  using this_type = MatrixPreconditionerBase<matrix_type, vector_type>;

  /**
   * Construct a new \c BlockSchurPreconditionerBase object
   */
  MatrixPreconditionerBase(const MatrixType& matrix);

  /**
   * The matrix vmult
   */
  virtual void vmult(VectorType& dst, const VectorType& src) const = 0;

  /**
   * @brief Get the stokes matrix
   */
  const matrix_type& get_matrix() const;

  /**
   * This function is called when a change in sparsity pattern
   * is signaled (due to mesh refinement/coarsening).
   * In this case the preconditioner may require reinitialzation.
   */
  virtual void reinitialize() = 0;


 protected:
  /**
   * @brief Pointer to the Stokes system matrix
   */
  const dealii::SmartPointer<const MatrixType, this_type> ptr_matrix;
};


template <typename APreconditionerType,
          typename SPreconditionerType,
          typename MatrixType,
          typename VectorType>
class BlockSchurPreconditionerBase
  : public MatrixPreconditionerBase<MatrixType, VectorType>
{
 public:
  using A_preconditioner_type = APreconditionerType;
  using S_preconditioner_type = SPreconditionerType;
  using matrix_type = MatrixType;
  using matrix_block_type = typename MatrixType::BlockType;
  using vector_type = VectorType;
  using base_type = MatrixPreconditionerBase<matrix_type, vector_type>;
  using this_type = BlockSchurPreconditionerBase<A_preconditioner_type,
                                                 S_preconditioner_type,
                                                 matrix_type,
                                                 vector_type>;


  BlockSchurPreconditionerBase(const matrix_type& stokes_matrix);


  BlockSchurPreconditionerBase(const matrix_type& stokes_matrix,
                               const matrix_block_type& approx_schur_matrix,
                               const APreconditionerType& preconditioner_A,
                               const SPreconditionerType& preconditioner_S);


  void initialize(const matrix_block_type& approx_schur_matrix,
                  const APreconditionerType& preconditioner_A,
                  const SPreconditionerType& preconditioner_S);


  dealii::ReductionControl& inverse_S_control();


  void vmult(vector_type& dst, const vector_type& src) const override;


 protected:
  /**
   *  Pointer to the preconditioner matrix
   */
  // dealii::SmartPointer<const matrix_block_type, this_type>
  //   ptr_approx_schur_matrix;

  /**
   *  Pointer to the preconditioner for A.
   */
  dealii::SmartPointer<const APreconditionerType, this_type>
    ptr_preconditioner_A;

  /**
   * Control parameters for the Schur inverse.
   */
  InverseMatrixCG<matrix_block_type, SPreconditionerType> inverse_S;


  /**
   * Temp vector
   */
  mutable typename vector_type::BlockType tmp;
};


#ifdef FELSPA_HAS_MPI
/* ---------- */
namespace mpi
/* ---------- */
{
  /* ************************************************** */
  /**
   * Using the block Schur preconditioner and AMG.
   */
  /* ************************************************** */
  template <typename APreconditionerType,
            typename SPreconditionerType,
            typename MatrixType,
            typename VectorType>
  class BlockSchurPreconditionerBase
    : public ::FELSPA_NAMESPACE::BlockSchurPreconditionerBase<
        APreconditionerType,
        SPreconditionerType,
        MatrixType,
        VectorType>
  {
   public:
    using base_type =
      ::FELSPA_NAMESPACE::BlockSchurPreconditionerBase<APreconditionerType,
                                                       SPreconditionerType,
                                                       MatrixType,
                                                       VectorType>;
    using matrix_type = MatrixType;
    using matrix_block_type = typename MatrixType::BlockType;

    /**
     * @brief Construct a new Block Schur Preconditioner Base object
     */
    BlockSchurPreconditionerBase(const matrix_type& stokes_matrix,
                                 MPI_Comm mpi_comm)
      : base_type(stokes_matrix), mpi_communicator(mpi_comm = MPI_COMM_WORLD)
    {}


    BlockSchurPreconditionerBase(const matrix_type& stokes_matrix,
                                 const matrix_block_type& approx_schur_matrix,
                                 const APreconditionerType& preconditioner_A,
                                 const SPreconditionerType& preconditioner_S,
                                 MPI_Comm mpi_comm = MPI_COMM_WORLD)
      : base_type(stokes_matrix,
                  approx_schur_matrix,
                  preconditioner_A,
                  preconditioner_S),
        mpi_communicator(mpi_comm)
    {}

   protected:
    /**
     * MPI communicator
     */
    const MPI_Comm mpi_communicator;
  };
}  // namespace mpi //
#endif  // FELSPA_HAS_MPI


FELSPA_NAMESPACE_CLOSE
/* ------------ IMPLEMENTATIONS ------------------ */
#include "src/linear_operators.implement.h"
/* ----------------------------------------------- */
#endif  // _FELSPA_LINEAR_ALGEBRA_LINEAR_OPERATOR_H_ //
