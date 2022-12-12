#ifndef _FELSPA_LINEAR_ALGEBRA_LINEAR_SYSTEM_IMPL_H_
#define _FELSPA_LINEAR_ALGEBRA_LINEAR_SYSTEM_IMPL_H_

#include <felspa/base/log.h>
#include <felspa/base/numerics.h>
#include <felspa/linear_algebra/linear_system.h>

#include <type_traits>

#ifdef FELSPA_CXX_PARALLEL_ALGORITHM
#include <execution>
#endif


FELSPA_NAMESPACE_OPEN

/* --------------------------------------------------*/
/** \class LinearSystemBase */
/* --------------------------------------------------*/
template <int dim, typename MatrixType, typename VectorType>
FELSPA_FORCE_INLINE
LinearSystemBase<dim, MatrixType, VectorType>::LinearSystemBase(
  const dealii::DoFHandler<dim>& dof_handler)
  : ptr_dof_handler(&dof_handler), ptr_mapping(nullptr), populated(false)
{}


template <int dim, typename MatrixType, typename VectorType>
FELSPA_FORCE_INLINE
LinearSystemBase<dim, MatrixType, VectorType>::LinearSystemBase(
  const dealii::DoFHandler<dim>& dof_handler,
  const dealii::Mapping<dim>& mapping)
  : ptr_dof_handler(&dof_handler), ptr_mapping(&mapping), populated(false)
{}


template <int dim, typename MatrixType, typename VectorType>
FELSPA_FORCE_INLINE const dealii::DoFHandler<dim>&
LinearSystemBase<dim, MatrixType, VectorType>::get_dof_handler() const
{
  ASSERT(ptr_dof_handler != nullptr, ExcNullPointer());
  return *ptr_dof_handler;
}

template <int dim, typename MatrixType, typename VectorType>
FELSPA_FORCE_INLINE const dealii::Mapping<dim>&
LinearSystemBase<dim, MatrixType, VectorType>::get_mapping() const
{
  ASSERT(ptr_mapping != nullptr, ExcNullPointer());
  return *ptr_mapping;
}


template <int dim, typename MatrixType, typename VectorType>
FELSPA_FORCE_INLINE bool
LinearSystemBase<dim, MatrixType, VectorType>::is_populated() const
{
  return populated;
}


template <typename T>
void TEST(T t);

template <int dim, typename MatrixType, typename VectorType>
void LinearSystemBase<dim, MatrixType, VectorType>::zero_out(bool zero_lhs,
                                                             bool zero_rhs)
{
  if (zero_lhs) matrix = value_type();
  if (zero_rhs) rhs = value_type();
}


template <int dim, typename MatrixType, typename VectorType>
FELSPA_FORCE_INLINE auto LinearSystemBase<dim, MatrixType, VectorType>::size()
  const -> size_type
{
  ASSERT(matrix.m() == matrix.n(), ExcMatrixNotSquare(matrix.m(), matrix.n()));
  ASSERT(matrix.m() == rhs.size(),
         ExcMatrixVectorNotMultiplicable(matrix.m(), rhs.size()));
  return rhs.size();
}


template <int dim, typename MatrixType, typename VectorType>
FELSPA_FORCE_INLINE bool LinearSystemBase<dim, MatrixType, VectorType>::empty()
  const
{
  return size() == 0;
}


template <int dim, typename MatrixType, typename VectorType>
void LinearSystemBase<dim, MatrixType, VectorType>::clear()
{
  rhs.reinit(0);
  matrix.clear();
  populated = false;
}


template <int dim, typename MatrixType, typename VectorType>
bool LinearSystemBase<dim, MatrixType, VectorType>::is_all_zero() const
{
  for (const auto& entry : matrix)
    if (!numerics::is_zero(entry.value())) return false;

  return rhs.all_zero();
}




/* --------------------------------------------------*/
/** \class LinearSystem */
/* --------------------------------------------------*/

template <int dim, typename NumberType>
LinearSystem<dim, NumberType>::LinearSystem(const dealii::DoFHandler<dim>& dofh)
  : base_type(dofh)
{}


template <int dim, typename NumberType>
LinearSystem<dim, NumberType>::LinearSystem(const dealii::DoFHandler<dim>& dofh,
                                            const dealii::Mapping<dim>& mapping)
  : base_type(dofh, mapping)
{}


template <int dim, typename NumberType>
FELSPA_FORCE_INLINE auto LinearSystem<dim, NumberType>::get_sparsity_pattern()
  const -> const sparsity_type&
{
  return sparsity_pattern;
}


template <int dim, typename NumberType>
void LinearSystem<dim, NumberType>::print_sparsity(ExportFile& file) const
{
  switch (file.get_format()) {
    case ExportFileFormat::svg:
      this->sparsity_pattern.print_svg(file.access_stream());
      break;
    case ExportFileFormat::gnuplot:
      this->sparsity_pattern.print_gnuplot(file.access_stream());
      break;
    default:
      THROW(ExcNotImplementedInFileFormat(file.get_file_extension()));
  }
}


template <int dim, typename NumberType>
void LinearSystem<dim, NumberType>::clear()
{
  this->base_type::clear();
  this->sparsity_pattern.reinit(0, 0, 0);
}


/* --------------------------------------------------*/
/** \class BlockLinearSystem */
/* --------------------------------------------------*/
template <int dim, typename NumberType>
BlockLinearSystem<dim, NumberType>::BlockLinearSystem(
  const dealii::DoFHandler<dim>& dofh)
  : base_type(dofh)
{}


template <int dim, typename NumberType>
BlockLinearSystem<dim, NumberType>::BlockLinearSystem(
  const dealii::DoFHandler<dim>& dofh, const dealii::Mapping<dim>& mapping)
  : base_type(dofh, mapping)
{}


template <int dim, typename NumberType>
FELSPA_FORCE_INLINE auto
BlockLinearSystem<dim, NumberType>::get_sparsity_pattern() const
  -> const sparsity_type&
{
  return sparsity_pattern;
}


template <int dim, typename NumberType>
void BlockLinearSystem<dim, NumberType>::count_dofs(
  bool compute_ndofs_component, bool compute_ndofs_block)
{
  ASSERT(this->ptr_dof_handler->has_active_dofs(), ExcDoFHandlerNotInit());

  if (compute_ndofs_component)
    this->ndofs_per_component =
      dealii::DoFTools::count_dofs_per_fe_component(this->get_dof_handler());

  if (compute_ndofs_block)
    this->ndofs_per_block =
      dealii::DoFTools::count_dofs_per_fe_block(this->get_dof_handler());

#ifdef DEBUG
  LOG_PREFIX("BlockLinearSystem");
  felspa_log << "The block linear system has "
             << this->ptr_dof_handler->get_fe().n_components()
             << " components and " << this->ptr_dof_handler->get_fe().n_blocks()
             << " blocks" << std::endl;
#endif  // DEBUG //
}


template <int dim, typename NumberType>
void BlockLinearSystem<dim, NumberType>::clear()
{
  base_type::clear();
  this->sparsity_pattern.reinit(0, 0);
}


template <int dim, typename NumberType>
void BlockLinearSystem<dim, NumberType>::print_sparsity(ExportFile& file) const
{
  switch (file.get_format()) {
    case ExportFileFormat::gnuplot:
      this->sparsity_pattern.print_gnuplot(file.access_stream());
      break;
    default:
      THROW(ExcNotImplementedInFileFormat(file.get_file_extension()));
  }
}


template <int dim, typename NumberType>
FELSPA_FORCE_INLINE auto
BlockLinearSystem<dim, NumberType>::get_component_ndofs() const
  -> const std::vector<size_type>&
{
  return ndofs_per_component;
}


template <int dim, typename NumberType>
FELSPA_FORCE_INLINE auto
BlockLinearSystem<dim, NumberType>::get_component_ndofs(
  unsigned int component) const -> size_type
{
  ASSERT(component < ndofs_per_component.size(),
         ExcOutOfRange(component, 0, ndofs_per_component.size()));
  return ndofs_per_component[component];
}


template <int dim, typename NumberType>
FELSPA_FORCE_INLINE auto BlockLinearSystem<dim, NumberType>::get_block_ndofs()
  const -> const std::vector<size_type>&
{
  return ndofs_per_block;
}


template <int dim, typename NumberType>
FELSPA_FORCE_INLINE auto BlockLinearSystem<dim, NumberType>::get_block_ndofs(
  unsigned int block) const -> size_type
{
  ASSERT(block < ndofs_per_block.size(),
         ExcOutOfRange(block, 0, ndofs_per_block.size()));
  return ndofs_per_block[block];
}


FELSPA_NAMESPACE_CLOSE
#endif  // _FELSPA_LINEAR_ALGEBRA_LINEAR_SYSTEM_IMPL_H_ //
