#ifndef _FELSPA_LINEAR_ALGEBRA_LINEAR_OPERATOR_IMPLEMENTATION_H_
#define _FELSPA_LINEAR_ALGEBRA_LINEAR_OPERATOR_IMPLEMENTATION_H_

#include <felspa/base/exceptions.h>
#include <felspa/linear_algebra/linear_operators.h>

FELSPA_NAMESPACE_OPEN


/* ***********************************************/
/*              InverseMatrixCG                    */
/* ***********************************************/
template <typename MatrixType, typename PreconditionerType>
InverseMatrixCG<MatrixType, PreconditionerType>::InverseMatrixCG()
  : ptr_matrix(nullptr), ptr_preconditioner(nullptr), ptr_control(nullptr)
{}


template <typename MatrixType, typename PreconditionerType>
InverseMatrixCG<MatrixType, PreconditionerType>::InverseMatrixCG(
  const MatrixType& matrix, const PreconditionerType& preconditioner,
  const value_type tolerance, const value_type reduction_factor)
  : ptr_matrix(&matrix),
    ptr_preconditioner(&preconditioner),
    ptr_control(std::make_unique<dealii::ReductionControl>())
{
  ptr_control->set_tolerance(tolerance);
  ptr_control->set_reduction(reduction_factor);
}


template <typename MatrixType, typename PreconditionerType>
void InverseMatrixCG<MatrixType, PreconditionerType>::initialize(
  const MatrixType& matrix, const PreconditionerType& preconditioner,
  const value_type tol, const value_type reduction)
{
  ptr_matrix = &matrix;
  ptr_preconditioner = &preconditioner;
  ptr_control = std::make_unique<dealii::ReductionControl>();
  ptr_control->set_tolerance(tol);
  ptr_control->set_reduction(reduction);
}


template <typename MatrixType, typename PreconditionerType>
FELSPA_FORCE_INLINE const dealii::ReductionControl&
InverseMatrixCG<MatrixType, PreconditionerType>::get_control() const
{
  ASSERT(ptr_control != nullptr, ExcNullPointer());
  return *ptr_control;
}


template <typename MatrixType, typename PreconditionerType>
FELSPA_FORCE_INLINE dealii::ReductionControl&
InverseMatrixCG<MatrixType, PreconditionerType>::control()
{
  ASSERT(ptr_control != nullptr, ExcNullPointer());
  return *ptr_control;
}


template <typename MatrixType, typename PreconditionerType>
template <typename VectorType>
void InverseMatrixCG<MatrixType, PreconditionerType>::vmult(
  VectorType& dst, const VectorType& src) const
{
  ASSERT(ptr_matrix != nullptr, ExcNullPointer());
  ASSERT(ptr_preconditioner != nullptr, ExcNullPointer());
  ASSERT(ptr_control != nullptr, ExcNullPointer());
  ASSERT_SAME_SIZE(dst, src);
  ASSERT_MATRIX_VECTOR_MULTIPLICABLE(*ptr_matrix, src);

  using namespace dealii;

  ptr_control->set_max_steps(src.size());
  dealii::SolverCG<VectorType> solver(*ptr_control);

  try {
    solver.solve(*ptr_matrix, dst, src, *ptr_preconditioner);
  }
  catch (std::exception& e) {
    // any exceptions will be rethrown from here.
    THROW(e);
  }

  ASSERT_SOLVER_CONVERGED(*ptr_control);
#ifdef VERBOSE
  LOG_PREFIX("InverseMatrixCG");
  felspa_log << "Converged in " << ptr_control->last_step() << " steps using "
             << FELSPA_DEMANGLE(*ptr_preconditioner)
             << " preconditioner to value " << ptr_control->last_value()
             << std::endl;
#endif
}


/* ***********************************************/
/*                 SchurComplement               */
/* ***********************************************/

template <typename BlockMatrixType, typename InverseMatrixType>
SaddlePointSchurComplement<BlockMatrixType, InverseMatrixType>::
  SaddlePointSchurComplement(const BlockMatrixType& block_matrix,
                             const InverseMatrixType& A_inverse)
  : ptr_matrix(&block_matrix), ptr_A_inverse(&A_inverse)
{}


template <typename BlockMatrixType, typename InverseMatrixType>
template <typename VectorType>
void SaddlePointSchurComplement<BlockMatrixType, InverseMatrixType>::vmult(
  VectorType& dst, const VectorType& src) const
{
  ASSERT_SAME_SIZE(dst, src);
  ASSERT_MATRIX_VECTOR_MULTIPLICABLE(ptr_matrix->block(0, 1), src);

  VectorType B_src(ptr_matrix->block(0, 0).m()),
    Ainv_B_src(ptr_matrix->block(0, 0).m());

  ptr_matrix->block(0, 1).vmult(B_src, src);
  ptr_A_inverse->vmult(Ainv_B_src, B_src);
  ptr_matrix->block(1, 0).vmult(dst, Ainv_B_src);
}


/* ************************************************** */
/*            MatrixPreconditionerBase                */
/* ************************************************** */

template <typename MatrixType, typename VectorType>
FELSPA_FORCE_INLINE
MatrixPreconditionerBase<MatrixType, VectorType>::MatrixPreconditionerBase(
  const MatrixType& matrix)
  : ptr_matrix(&matrix)
{}


template <typename MatrixType, typename VectorType>
FELSPA_FORCE_INLINE auto
MatrixPreconditionerBase<MatrixType, VectorType>::get_matrix() const
  -> const matrix_type&
{
  ASSERT(ptr_matrix != nullptr, ExcNullPointer());
  return *ptr_matrix;
}

/* ************************************************** */
/*            BlockSchurPreconditioner                */
/* ************************************************** */

template <typename APreconditionerType, typename SPreconditionerType,
          typename MatrixType, typename VectorType>
FELSPA_FORCE_INLINE BlockSchurPreconditionerBase<
  APreconditionerType, SPreconditionerType, MatrixType, VectorType>::
  BlockSchurPreconditionerBase(const matrix_type& stokes_matrix,
                               const matrix_block_type& approx_schur_matrix,
                               const APreconditionerType& preconditioner_A,
                               const SPreconditionerType& preconditioner_S)
  : base_type(stokes_matrix),
    ptr_preconditioner_A(&preconditioner_A),
    inverse_S(approx_schur_matrix, preconditioner_S)
{}


template <typename APreconditionerType, typename SPreconditionerType,
          typename MatrixType, typename VectorType>
FELSPA_FORCE_INLINE BlockSchurPreconditionerBase<
  APreconditionerType, SPreconditionerType, MatrixType,
  VectorType>::BlockSchurPreconditionerBase(const matrix_type& stokes_matrix)
  : base_type(stokes_matrix)
{}


template <typename APreconditionerType, typename SPreconditionerType,
          typename MatrixType, typename VectorType>
void BlockSchurPreconditionerBase<
  APreconditionerType, SPreconditionerType, MatrixType,
  VectorType>::initialize(const matrix_block_type& approx_schur_matrix,
                          const APreconditionerType& precond_A,
                          const SPreconditionerType& precond_S)
{
  ptr_preconditioner_A = &precond_A;
  inverse_S.initialize(approx_schur_matrix, precond_S);
}


template <typename APreconditionerType, typename SPreconditionerType,
          typename MatrixType, typename VectorType>
dealii::ReductionControl&
BlockSchurPreconditionerBase<APreconditionerType, SPreconditionerType,
                             MatrixType, VectorType>::inverse_S_control()
{
  return inverse_S.control();
}


template <typename APreconditionerType, typename SPreconditionerType,
          typename MatrixType, typename VectorType>
void BlockSchurPreconditionerBase<
  APreconditionerType, SPreconditionerType, MatrixType,
  VectorType>::vmult(vector_type& dst, const vector_type& src) const
{
  ASSERT(ptr_preconditioner_A != nullptr, ExcNullPointer());
  ptr_preconditioner_A->vmult(dst.block(0), src.block(0));
  this->ptr_matrix->block(1, 0).residual(tmp, dst.block(0), src.block(1));
  tmp *= -1.0;
  inverse_S.vmult(dst.block(1), tmp);
}

FELSPA_NAMESPACE_CLOSE
#endif  // _FELSPA_LINEAR_ALGEBRA_LINEAR_OPERATOR_IMPLEMENTATION_H_
