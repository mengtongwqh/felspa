#ifndef _FELSPA_LINEAR_ALGEBRA_ILU_H_
#define _FELSPA_LINEAR_ALGEBRA_ILU_H_

#include <deal.II/base/timer.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_decomposition.h>
#include <felspa/base/exceptions.h>
#include <felspa/base/felspa_config.h>
#include <felspa/base/io.h>

#include <forward_list>

FELSPA_NAMESPACE_OPEN


/* ************************************************** */
/**
 * This is a re-implementation of deal.II SparseILU
 * class. The dealii::SparseILU only implements ILU(0)
 * preconditioning, with the possibility to extend sparsity
 * pattern to some elements further off diagonal. We
 * generalize the implementation to arbitraty level of fill.
 */
/* ************************************************** */
template <typename NumberType>
class SparseILU : public dealii::SparseMatrix<NumberType>,
                  public virtual dealii::Subscriptor
{
 public:
  /**
   * Control parameters
   */
  class AdditionalData;

  using value_type = NumberType;
  using typename dealii::SparseMatrix<NumberType>::size_type;
  using row_num_type = size_type;
  using col_num_type = size_type;
  using idx_type = std::size_t;
  using level_of_fill_type = unsigned int;
  using iterator = dealii::SparseMatrixIterators::Iterator<value_type, false>;
  using const_iterator =
    dealii::SparseMatrixIterators::Iterator<value_type, true>;

  constexpr static size_type invalid_size_type = static_cast<size_type>(-1);
  constexpr static level_of_fill_type invalid_level_of_fill =
    static_cast<level_of_fill_type>(-1);



/**
 * Constructor
 */
#ifdef PROFILING
  SparseILU()
    : timer(std::cout, dealii::TimerOutput::summary,
            dealii::TimerOutput::wall_times)
  {}
#else
  SparseILU() = default;
#endif  // PROFILING


  /**
   * The preconditioner must be initialized prior its usage.
   * -# Allocate and compute the
   *  sparsity pattern by symbolic factorization
   *  (or analyse).
   * -# Compute numeric factorization, which is an LU on the sparsity.
   */
  template <typename OtherNumberType>
  void initialize(const dealii::SparseMatrix<OtherNumberType>& spmatrix,
                  const AdditionalData& additional_data);


  /**
   * Multiplcation of the perconditioner with a vector.
   * Done by a computing a forward and backward solve
   * for dst such that: L*U*dst = src.
   */
  template <typename OtherNumberType>
  void vmult(dealii::Vector<OtherNumberType>& dst,
             const dealii::Vector<OtherNumberType>& src) const;


  /**
   * Multiplication of the transpose of the preconditioner with a vector.
   * Done by a computing a forward and backward solve
   * for dst such that: (L*U)'*dst = src.
   */
  template <typename OtherNumberType>
  void Tvmult(dealii::Vector<OtherNumberType>& dst,
              const dealii::Vector<OtherNumberType>& src) const;


  /**
   * @brief Get the sparse matrix object
   */
  const dealii::SparseMatrix<value_type>& get_sparse_matrix() const
  {
    return static_cast<const dealii::SparseMatrix<value_type>&>(*this);
  }


  /**
   *
   */
  const_iterator post_diagonal(row_num_type row) const
  {
    return one_after_diagonal[row];
  }


  /**
   * Print the sparsity pattern to .svg file.
   */
  void print_sparsity_svg(std::ostream& os) const;


  /**
   * @brief get a handle on the control parametes
   */
  AdditionalData& control() { return additional_data; }


 protected:
  /**
   * Helper function to compute symbolic factorization.
   * Also called the "Analyze" phase in some literatures.
   */
  auto symbolic_factorize(const dealii::SparsityPattern& linsys_sparsity,
                          const level_of_fill_type max_level_of_fill)
    -> const dealii::SparsityPattern*;


  /**
   * Numeric factorization. Or also called the "Factor" phase.
   */
  template <typename OtherNumberType>
  void numeric_factorize(const dealii::SparseMatrix<OtherNumberType>&);


  /**
   * When this function is called,\c one_after_diagonal
   * vector will be populated with iterator pointing the element
   * immediately to the right of the diagonal.
   */
  void cache_diagonal_iterators();


  /**
   * Pointer to the sparsity that is created
   * and managed by the \c SparseILU preconditioner.
   */
  std::unique_ptr<dealii::SparsityPattern> ptr_own_sparsity = nullptr;


  /**
   * Pointer to sparsity pattern that is
   * actually used by the \c SparseILU preconditioner.
   */
  const dealii::SparsityPattern* ptr_sparsity_in_use;


  /**
   * Level of fill for each row
   * \todo  use a vector for O(1) retrieval
   */
  std::vector<std::vector<level_of_fill_type>> row_level_of_fills;


  /**
   * An iterator pointing to the column immediately after diagonal.
   */
  std::vector<const_iterator> one_after_diagonal;


  /**
   * @brief Additional control parameters
   */
  AdditionalData additional_data;


  /**
   * Maximum level of fill for this preconditioner.
   */
  // level_of_fill_type max_level_of_fill = invalid_level_of_fill;


 private:
  /**
   * @brief  Compute symbolic factorization for one level.
   * This function will treat all the nonzero entries in the input \c sparsity
   * as LoF = 0.
   */
  auto symbolic_factorize_levelwise(const dealii::SparsityPattern& sparsity)
    -> const dealii::SparsityPattern*;


  /**
   * @brief
   */
  void row_factorization(
    row_num_type irow_begin, row_num_type irow_end,
    dealii::DynamicSparsityPattern& dsp,
    const std::vector<typename dealii::SparsityPatternIterators::Iterator>&
      linsys_post_diagonal,
    const dealii::SparsityPattern& linsys_sparsity);


  /**
   * @brief Brute-force symbolic factorization.
   * Purely serial, works for any level of fill.
   */
  auto symbolic_factorize_generic(const dealii::SparsityPattern& sparsity,
                                  level_of_fill_type max_lof)
    -> const dealii::SparsityPattern*;


  /**
   * @brief helper function to generate one entry for .svg file. One entry in
   * the file corresponds to a nonzero element in the symbolic factorized
   * sparisty pattern.
   */
  void generate_svg_entry(std::ostream& os, level_of_fill_type lof,
                          row_num_type row, col_num_type col) const;


  size_type block_size_per_thread;


#ifdef PROFILING
  /**
   *  Timing the analyze-factor-solve phases for the perconditioner.
   */
  mutable dealii::TimerOutput timer;
#endif  // PROFILING
};


/* ************************************************** */
/**
 * Control parameters for SparseILU decomposition.
 */
/* ************************************************** */
template <typename NumberType>
class SparseILU<NumberType>::AdditionalData
{
 public:
  using level_of_fill_type = typename SparseILU<NumberType>::level_of_fill_type;
  using size_type = typename SparseILU<NumberType>::size_type;

  /**
   * Constructor
   */
  AdditionalData(const bool use_previous_sparsity_ = false,
                 const level_of_fill_type level_of_fill = 0,
                 const size_type block_size = 1000)
    : max_level_of_fill(level_of_fill),
      use_previous_sparsity(use_previous_sparsity_),
      block_size_per_thread(block_size)
  {
    ASSERT(block_size > 0, ExcArgumentCheckFail());
  }

  /**
   * Maximum level of fill.
   */
  level_of_fill_type max_level_of_fill;

  /**
   * If set to \c true, then the perconditioner will keep using the previous
   * sparsity and entries. The preconditioner will not compute a symbolic or
   * numeric factorization.
   */
  bool use_previous_sparsity;


  /**
   *  Size of the matrix to be processed by each thread.
   */
  size_type block_size_per_thread;
};

// --------------------------
namespace internal
// --------------------------
{
  /* ************************************************** */
  /**
   *  Base class for all shared-memory parallel
   * substitution implementation
   */
  /* ************************************************** */
  template <typename NumberType>
  class SubstitutionBase
  {
   public:
    using value_type = NumberType;
    using size_type = typename SparseILU<NumberType>::size_type;
    using row_num_type = typename SparseILU<NumberType>::row_num_type;
    using col_num_type = typename SparseILU<NumberType>::col_num_type;
    using sparse_matrix_type = dealii::SparseMatrix<value_type>;
    using vector_type = dealii::Vector<value_type>;


   protected:
    /**
     * @brief Construct a new SubstitutionPartitioner object
     */
    SubstitutionBase(const SparseILU<NumberType>& sparse_ilu,
                     size_type interval)
      : ptr_ilu(&sparse_ilu), block_size(interval)
    {
      ASSERT(interval > 0, ExcArgumentCheckFail());
    }


    /**
     * SparseILU object.
     */
    const SparseILU<value_type>* ptr_ilu;

    /**
     * size of each block.
     * We have int(N/block_size) + 1 to deal with.
     */
    size_type block_size;


    std::vector<dealii::SparseMatrixIterators::Iterator<value_type, true>>
      iterator_cache;


    /**
     * Destination vector
     */
    vector_type* ptr_dst = nullptr;


    /**
     * Mutex object
     */
    std::mutex mtx;
  };


  /* ************************************************** */
  /**
   * @brief Parallel implementation for ILU forward substitution.
   * Solving L * y = b where L is the lower-diagonal part.
   * All diagonal elements are equal to 1.0
   */
  /* ************************************************** */
  template <typename NumberType>
  class ForwardSubstitution : public SubstitutionBase<NumberType>
  {
   public:
    using value_type = NumberType;
    using base_type = SubstitutionBase<NumberType>;
    using typename base_type::size_type;
    using typename base_type::vector_type;

    ForwardSubstitution(const SparseILU<NumberType>& sparse_ilu,
                        size_type block_size)
      : base_type(sparse_ilu, block_size)
    {
      for (size_type irow = 0; irow < this->ptr_ilu->m(); ++irow)
        this->iterator_cache.push_back(++this->ptr_ilu->begin(irow));
    }

    void run(vector_type& dst);

   protected:
    void compute_diagonal_block(size_type col_begin, size_type col_end);

    void compute_off_diagonal_block(size_type row_begin, size_type row_end,
                                    size_type col_begin, size_type col_end);
  };


  /* ************************************************** */
  /**
   * @brief
   */
  /* ************************************************** */
  template <typename NumberType>
  class BackwardSubstitution : public SubstitutionBase<NumberType>
  {
   public:
    using value_type = NumberType;
    using base_type = SubstitutionBase<NumberType>;
    using typename base_type::size_type;
    using typename base_type::vector_type;

    BackwardSubstitution(const SparseILU<NumberType>& sparse_ilu,
                         size_type block_size)
      : base_type(sparse_ilu, block_size)
    {
      for (size_type irow = 0; irow < this->ptr_ilu->m(); ++irow)
        this->iterator_cache.push_back(this->ptr_ilu->post_diagonal(irow));
    }

    void run(vector_type& dst);

   protected:
    void compute_diagonal_block(size_type begin, size_type end);

    void compute_off_diagonal_block(size_type row_begin, size_type row_end,
                                    size_type col_begin, size_type col_end);
  };

}  // namespace internal


FELSPA_NAMESPACE_CLOSE

/* ----------  IMPLEMENTATION ----------- */
#include "src/ilu.implement.h"
/* -------------------------------------- */

#endif  // _FELSPA_LINEAR_ALGEBRA_ILU_H_ //
