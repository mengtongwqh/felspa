#ifndef _FELSPA_BASE_TYPES_H_
#define _FELSPA_BASE_TYPES_H_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>

#include <felspa/base/felspa_config.h>

FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/**
 * \namespace types
 * \brief numerical types used for the library
 */
/* ************************************************** */
/* ------------------------------ */
namespace types
/* ------------------------------ */
{
  /* ------------------------- */
  /** \name Number/Index types */
  /* ------------------------- */
  //@{
  /**
   * Type for DoF count
   */
  using DoFIndex = dealii::types::global_dof_index;


  /**
   * Particle index
   */
  using ParticleIndex = size_t;


  /**
   * Type for container size
   */
  using SizeType = DoFIndex;


  /**
   * Type for material index
   */
  using MaterialIndex = unsigned int;


  /**
   * Type for lower precision floating point number
   */
  using FloatType = float;


  /**
   * The type of double used in a simulator,
   * especially for spatial dimension
   */
  using DoubleType = double;


  /**
   * Type for long (higher precision) double type
   */
  using LongDoubleType = long double;


  /**
   * The type used for temporal discretization
   */
  using TimeStepType = double;


  /**
  * Type of scalar type in the dealii::TrilinosWrapper
  */
  using TrilinosScalar = dealii::TrilinosScalar;
  //@}


  /* ************************************************** */
  /**
   *  Given a sparse matrix type,
   *  Construct the blocked equivalence
   */
  /* ************************************************** */
  template <typename T>
  struct BlockedEquivalence;

  template <typename NumberType>
  struct BlockedEquivalence<dealii::SparseMatrix<NumberType>>
  {
    using type = dealii::BlockSparseMatrix<NumberType>;
  };

  template <typename NumberType>
  struct BlockedEquivalence<dealii::Vector<NumberType>>
  {
    using type = dealii::BlockVector<NumberType>;
  };

#ifdef DEAL_II_WITH_TRILINOS
  template <>
  struct BlockedEquivalence<dealii::TrilinosWrappers::MPI::Vector>
  {
    using type = dealii::TrilinosWrappers::MPI::BlockVector;
  };

  template <>
  struct BlockedEquivalence<dealii::TrilinosWrappers::SparseMatrix>
  {
    using type = dealii::TrilinosWrappers::BlockSparseMatrix;
  };
#endif // DEAL_II_WITH_TRILINOS
}  // namespace types

FELSPA_NAMESPACE_CLOSE

#endif  // _FELSPA_BASE_TYPES_H_
