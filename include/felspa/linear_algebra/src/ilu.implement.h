#ifndef _FELSPA_LINEAR_ALGEBRA_ILU_LEVEL_OF_FILL_IMPLEMENT_H_
#define _FELSPA_LINEAR_ALGEBRA_ILU_LEVEL_OF_FILL_IMPLEMENT_H_

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <felspa/base/constants.h>
#include <felspa/base/exceptions.h>
#include <felspa/base/log.h>
#include <felspa/base/numerics.h>
#include <felspa/base/utilities.h>
#include <felspa/linear_algebra/ilu.h>

#include <algorithm>

// #define VERBOSE
// due to the sparsity of the matrix and the fact that there are no elements far
// away from the diagonal, serial vmult is sufficient and preserves better cache
// coherence.
#define USE_SERIAL_VMULT

// use serial brute force symbolic factorization even for level 1
// #define FORCE_SERIAL_SYMBOLIC_FACTORIZE

FELSPA_NAMESPACE_OPEN

/* ************************************************** */
// SparseILU //
/* ************************************************** */

template <typename NumberType>
template <typename OtherNumberType>
void SparseILU<NumberType>::initialize(
  const dealii::SparseMatrix<OtherNumberType>& spmatrix,
  const AdditionalData& additional_data_)
{
  using namespace dealii;
  LOG_PREFIX("SparseILU");

  this->additional_data = additional_data_;

  if (!additional_data.use_previous_sparsity) {
    // the symbolic factorization will compute a brand new sparsity
    ptr_sparsity_in_use = symbolic_factorize(spmatrix.get_sparsity_pattern(),
                                             additional_data.max_level_of_fill);
    // allocate space for SparseMatrix
    dealii::SparseMatrix<NumberType>::reinit(*ptr_sparsity_in_use);
  }

  ASSERT(this->m() == this->n(), ExcMatrixNotSquare(this->m(), this->n()));
  ASSERT(ptr_sparsity_in_use != nullptr,
         EXCEPT_MSG("Pointer to sparsity in NULL. We don't have a sparsity "
                    "defined for the preconditioner to use."));

  // compute numeric factorization using ptr_sparsity_in_use
  numeric_factorize(spmatrix);
}


template <typename NumberType>
const dealii::SparsityPattern* SparseILU<NumberType>::symbolic_factorize(
  const dealii::SparsityPattern& linsys_sparsity, level_of_fill_type max_lof)
{
#ifdef PROFILING
  dealii::TimerOutput::Scope t(this->timer, "symbolic_factor");
#endif  // PROFILING

  ASSERT(
    linsys_sparsity.n_rows() == linsys_sparsity.n_cols(),
    ExcMatrixNotSquare(linsys_sparsity.n_rows(), linsys_sparsity.n_cols()));

  felspa_log << "Running symbolic factorization with LoF = " << max_lof
             << " on a " << linsys_sparsity.n_rows() << " x "
             << linsys_sparsity.n_cols() << " matrix with "
             << linsys_sparsity.n_nonzero_elements() << " entries."
             << std::endl;

  // release the previously stored sparsity and row_level_of_fills
  this->ptr_own_sparsity.reset();
  this->clear();
  row_level_of_fills.clear();

#ifndef FORCE_SERIAL_SYMBOLIC_FACTORIZE
  if (max_lof == 1)
    return symbolic_factorize_levelwise(linsys_sparsity);
  else
#endif  // !FORCE_SERIAL //
    return symbolic_factorize_generic(linsys_sparsity, max_lof);
}


template <typename NumberType>
void SparseILU<NumberType>::row_factorization(
  row_num_type irow_begin, row_num_type irow_end,
  dealii::DynamicSparsityPattern& dsp,
  const std::vector<typename dealii::SparsityPatternIterators::Iterator>&
    linsys_post_diagonal,
  const dealii::SparsityPattern& sparsity)
{
  for (row_num_type irow = irow_begin; irow < irow_end; ++irow) {
    std::forward_list<col_num_type> col_num_list;
    std::vector<typename std::forward_list<col_num_type>::const_iterator>
      list_left_iterators;
    auto nnz_row = sparsity.row_length(irow);

    // load the entries of the original matrix on this row
    auto it_list = col_num_list.before_begin();
    for (auto it_col = ++sparsity.begin(irow); it_col != sparsity.end(irow);
         ++it_col) {
      auto insert = col_num_list.insert_after(it_list, it_col->column());
      ++it_list;
      if (it_col->column() < irow) list_left_iterators.push_back(insert);
    }

    bool inserted = false;
    if (col_num_list.empty() && nnz_row == 1) {
      // the only entry on this row is the diagonal
      col_num_list.push_front(irow);
      inserted = true;
    } else {
      // find the position to insert diagonal to preserve sorting
      for (auto it_front = col_num_list.before_begin(),
                it_back = col_num_list.begin();
           it_front != col_num_list.end(); ++it_front, ++it_back) {
        if ((it_front == col_num_list.before_begin() && *it_back > irow) ||
            (it_back == col_num_list.end() && *it_front < irow) ||
            (*it_front < irow && irow < *it_back)) {
          ASSERT(inserted == false, ExcInternalErr());
          col_num_list.insert_after(it_front, irow);
          inserted = true;
          break;
        }
      }
    }  // iterate over col_num_list

    ASSERT(std::is_sorted(col_num_list.begin(), col_num_list.end()),
           ExcInternalErr());
    ASSERT(inserted, ExcInternalErr());

    // walk thru elements on the right half of the row
    auto it_list_left = list_left_iterators.begin();
    for (auto isp_col_left = ++sparsity.begin(irow);
         isp_col_left != linsys_post_diagonal[irow];
         ++isp_col_left, ++it_list_left) {
      // get (global) column number
      col_num_type col_num_left = isp_col_left->column();

      // find the corresponding row in the linsys matrix
      // and extract the entries right of the diagonal
      // and load them into the linked list
      auto ilst_front = *it_list_left;

      for (auto jsp_row_right = linsys_post_diagonal[col_num_left];
           jsp_row_right != sparsity.end(col_num_left);
           ++jsp_row_right) {
        // load them into the linked list
        col_num_type jsp_col_num = jsp_row_right->column();

        for (; ilst_front != col_num_list.end(); ++ilst_front) {
          auto ilst_back = ilst_front;
          ++ilst_back;
          if ((ilst_front == col_num_list.before_begin() &&
               *ilst_back > jsp_col_num) ||
              (ilst_back == col_num_list.end() && *ilst_front < jsp_col_num) ||
              (*ilst_front < jsp_col_num && jsp_col_num < *ilst_back)) {
            col_num_list.insert_after(ilst_front, jsp_col_num);
            ++nnz_row;
            break;
          }

          // if we have duplicate entries then simply skip
          if (ilst_front != col_num_list.before_begin() &&
              *ilst_front == jsp_col_num)
            break;
          if (ilst_back != col_num_list.end() && *ilst_back == jsp_col_num)
            break;
        }  // ilst_front-loop

        // insertion/duplicate-skipping never happened
        ASSERT(ilst_front != col_num_list.end(), ExcInternalErr());
      }
    }  // jsp_row_right

    ASSERT(std::is_sorted(col_num_list.begin(), col_num_list.end()),
           ExcInternalErr());
    ASSERT(std::distance(col_num_list.begin(), col_num_list.end()) == nnz_row,
           ExcInternalErr());

    // dump the col_num_list to std::vector
    std::vector<col_num_type> tmp_col_nums(nnz_row);
    std::copy(col_num_list.begin(), col_num_list.end(), tmp_col_nums.begin());
    dsp.add_entries(irow, tmp_col_nums.cbegin(), tmp_col_nums.cend(), true);

  }  // irow-loop
}

template <typename NumberType>
auto SparseILU<NumberType>::symbolic_factorize_levelwise(
  const dealii::SparsityPattern& linsys_sparsity)
  -> const dealii::SparsityPattern*
{
  using namespace dealii;
  felspa_log << "Using level-wise factorization..." << std::endl;

  const row_num_type N = linsys_sparsity.n_rows();
  DynamicSparsityPattern dsp(N);
  std::vector<SparsityPatternIterators::Iterator> linsys_post_diagonal(
    N, linsys_sparsity.begin());

  // cache diagonal position
  for (row_num_type irow = 0; irow < N; ++irow) {
    bool inserted = false;
    for (auto it = ++linsys_sparsity.begin(irow);
         it != linsys_sparsity.end(irow);
         ++it) {
      if (it->column() > irow) {
        linsys_post_diagonal[irow] = it;
        inserted = true;
        break;
      }
    }  // loop thru each off-diagonal entry
    // nothing is on the right of the diagonal
    if (!inserted) linsys_post_diagonal[irow] = linsys_sparsity.end(irow);
  }

  // a parallel for loop using OpenMP
#ifdef FORCE_SERIAL_SYMBOLIC_FACTORIZE
  row_factorization(0, N, dsp, linsys_post_diagonal, linsys_sparsity);
#else
  dealii::parallel::apply_to_subranges(
    0, N,
    std::bind(&SparseILU<NumberType>::row_factorization, this,
              std::placeholders::_1, std::placeholders::_2, std::ref(dsp),
              std::cref(linsys_post_diagonal), std::cref(linsys_sparsity)),
    block_size_per_thread);
#endif  // FORCE_SERIAL //

  ptr_own_sparsity = std::make_unique<dealii::SparsityPattern>();
  ptr_own_sparsity->copy_from(dsp);
  felspa_log << "Factorized fill count: "
             << this->ptr_own_sparsity->n_nonzero_elements() << std::endl;

  return ptr_own_sparsity.get();
}


template <typename NumberType>
auto SparseILU<NumberType>::symbolic_factorize_generic(
  const dealii::SparsityPattern& sparsity,
  const level_of_fill_type max_level_of_fill) -> const dealii::SparsityPattern*
{
  felspa_log << "Using serial generic brute force factorization..."
             << std::endl;

  using namespace dealii;
  // this->max_level_of_fill = max_level_of_fill;

  // ILU(0) simply use the sparsity of the linear system.
  // So we will just use that sparsity,
  // nothing further needs to be done.
  if (max_level_of_fill == 0) return &sparsity;

  const row_num_type N = sparsity.n_rows();  // n_rows() == n_cols()

  row_level_of_fills.resize(N);

  DynamicSparsityPattern dsp(N);
  std::vector<size_type> row_diagonals(N);
  std::vector<level_of_fill_type> row_lof_scatter(N, invalid_level_of_fill);

  // loop through each row of the matrix
  for (row_num_type irow = 0; irow < N; ++irow) {
    // load the entries from original matrix into
    // the singly linked list and set their LoFs to 0
    std::forward_list<col_num_type> row_col_num;
    auto nnz_row = sparsity.row_length(irow);

    auto it_list = row_col_num.before_begin();
    for (decltype(nnz_row) icol = 1; icol < nnz_row; ++icol) {
      row_col_num.insert_after(it_list, sparsity.column_number(irow, icol));
      ++it_list;
    }

    ASSERT(irow == sparsity.column_number(irow, 0), ExcInternalErr());
    ASSERT(std::is_sorted(row_col_num.begin(), row_col_num.end()),
           ExcInternalErr());

    // query the position to insert diagonal entry
    bool inserted = false;

    if (row_col_num.empty() && nnz_row == 1) {
      // in this case the only entry on this row is the diagonal
      row_col_num.push_front(irow);
      inserted = true;
    } else {
      for (auto it_front = row_col_num.before_begin(),
                it_back = row_col_num.begin();
           it_front != row_col_num.end(); ++it_front, ++it_back) {
        if ((it_front == row_col_num.before_begin() && *it_back > irow) ||
            (it_back == row_col_num.end() && *it_front < irow) ||
            (*it_front < irow && irow < *it_back)) {
          ASSERT(!inserted, ExcInternalErr());
          row_col_num.insert_after(it_front, irow);
          inserted = true;
          break;
        }
      }  // loop thru row_col_num
    }

    ASSERT(std::is_sorted(row_col_num.begin(), row_col_num.end()),
           ExcInternalErr());
    ASSERT(inserted, ExcInternalErr());

    for (auto col_num : row_col_num) row_lof_scatter[col_num] = 0;


    // ----------------------------------------------------
    // for each column index i in the linked list,
    // go to the i-th row of the matrix and merge the
    // induced fills that are <= max_level_of_fill
    // into the row_col_idx_list
    // ----------------------------------------------------

    // Walk thru the row corresponding to the column indices
    // that live on left half of this row
    for (auto it_col_num = row_col_num.begin(); *it_col_num < irow;
         ++it_col_num) {
      const auto row_num_left = *it_col_num;
      auto it_front = it_col_num;

      // if this entry has an LoF == max_level_of_fill, then
      // all fills generated by this entry will be greater than
      // max_level_of_fill and thus excluded anyways.
      if (row_lof_scatter[row_num_left] < max_level_of_fill) {
        auto nnz = dsp.row_length(row_num_left);

        // now walk thru the right half of the row corresponding to
        // *it_col_idx
        for (col_num_type col_idx_right = row_diagonals[row_num_left] + 1;
             col_idx_right != nnz;
             ++col_idx_right) {
          col_num_type col_num_right =
            dsp.column_number(row_num_left, col_idx_right);

          level_of_fill_type level_ith_col = row_lof_scatter[row_num_left];
          level_of_fill_type level_jth_col = row_lof_scatter[col_num_right];

          level_of_fill_type level_ij = std::min(
            row_level_of_fills[row_num_left][col_idx_right] + level_ith_col + 1,
            level_jth_col);

          if (level_ij <= max_level_of_fill) {
            // this fill is new, add it to the linked list
            // else the fill already exists and therefore there is
            // no need to add it again.
            if (row_lof_scatter[col_num_right] == invalid_level_of_fill) {
              // find the insertion point
              bool inserted = false;
              for (; it_front != row_col_num.end(); ++it_front) {
                auto it_back = it_front;
                it_back++;
                // walk to the correct insertion point
                if ((it_back == row_col_num.end() ||
                     *it_back > col_num_right) &&
                    *it_front < col_num_right) {
                  ASSERT(inserted == false, ExcInternalErr());
                  row_col_num.insert_after(it_front, col_num_right);
                  inserted = true;
                  break;
                }
              }
              ASSERT(inserted, ExcInternalErr());

              // update LoF entry
              row_lof_scatter[col_num_right] = level_ij;
              ++nnz_row;
            }  // if insertion to row_lof

          }  // if (level_ij <= max_level_of_fill)
        }    // loop: columns right of diagonal on row_left
      }      // if (row_lof[row_num_left] < max_level_of_fill)
    }        // loop through left half of the row

    // now write these entries into the dynamic sparsity pattern
    // First write into a temporary and then pass into dsp.add_entries()
    // because the latter requires iterator to implement '<' operator
    ASSERT(std::is_sorted(row_col_num.begin(), row_col_num.end()),
           ExcInternalErr());
    std::vector<col_num_type> tmp_col_num(nnz_row);
    std::copy(row_col_num.cbegin(), row_col_num.cend(), tmp_col_num.begin());
    dsp.add_entries(irow, tmp_col_num.cbegin(), tmp_col_num.cend(), true);
    ASSERT(dsp.row_length(irow) == nnz_row, ExcInternalErr());


    // Move level of fills. row_lof now longer valid.
    std::vector<level_of_fill_type> tmp_lof(nnz_row);
    std::transform(tmp_col_num.cbegin(), tmp_col_num.cend(), tmp_lof.begin(),
                   [&](col_num_type i) {
                     level_of_fill_type lof = row_lof_scatter[i];
                     row_lof_scatter[i] = invalid_level_of_fill;
                     return lof;
                   });
    row_level_of_fills[irow] = std::move(tmp_lof);

#ifdef DEBUG
    // Make sure the row_lof_scatter is all reset
    // Beware that his check will slow this function to O(N^2)
    std::for_each(row_lof_scatter.begin(), row_lof_scatter.end(),
                  [](level_of_fill_type lof) {
                    ASSERT(lof == invalid_level_of_fill, ExcInternalErr());
                  });
#endif  // DEBUG //

    // determine the position of the diagonal
    inserted = false;
    size_type diag_idx = 0;
    for (const auto col_num : tmp_col_num) {
      if (col_num == irow) {
        inserted = true;
        row_diagonals[irow] = diag_idx;
        break;
      }
      ++diag_idx;
    }
    ASSERT(inserted, ExcInternalErr());
  }  // loop thru all rows

  // copy the dynamic sparsity into sparisty pattern
  ptr_own_sparsity = std::make_unique<dealii::SparsityPattern>();
  ptr_own_sparsity->copy_from(dsp);

  felspa_log << "Factorized fill count: "
             << this->ptr_own_sparsity->n_nonzero_elements() << std::endl;

  // return a built-in pointer to the newly-built sparsity
  return this->ptr_own_sparsity.get();
}


template <typename NumberType>
template <typename OtherNumberType>
void SparseILU<NumberType>::numeric_factorize(
  const dealii::SparseMatrix<OtherNumberType>& linsys_matrix)
{
  ASSERT(linsys_matrix.m() == linsys_matrix.n(),
         ExcMatrixNotSquare(this->m(), this->n()));
  ASSERT(linsys_matrix.m() == this->m(),
         ExcSizeMismatch(linsys_matrix.m(), this->m()));
  ASSERT(linsys_matrix.n() == this->n(),
         ExcSizeMismatch(linsys_matrix.n(), this->n()));

#ifdef PROFILING
  dealii::TimerOutput::Scope t(this->timer, "numeric_factorize");
#endif  // PROFILING

  felspa_log << "Running numeric factorization..." << std::endl;

  // iterator to the location of the 1st elmt right of the diagonal
  // In deal.II there is a similar function:
  // dealii::SparseLUDecomposition<NumberType>::prebuild_lower_bound();
  // but it generates raw pointer to protected colnum vector.
  // However since we don't have access to that vector, we choose to
  // cache the iterators.
  this->cache_diagonal_iterators();

  const row_num_type N = this->m();
  auto const_this = static_cast<const SparseILU<value_type>*>(this);

  // NOTE: The row work vector should be only updated
  // where the entry has a meaningful level-of-fill.
  // Or else entries from the previous rows
  // will cause excessive fill-in that won't be
  // zeroed out after processing this row
  // That's why we need the filter vector.
  std::vector<bool> row_filter(N, false);

  // scatter the entries of the row.
  // This has the advantage of avoid worrying about the
  // out-of-sequence diagonal element.
  std::vector<value_type> row_scatter(N, 0.0);

  for (row_num_type irow = 0; irow < N; ++irow) {
    // load the original entries of the matrix into the scatter vector
    for (auto icol = linsys_matrix.begin(irow); icol != linsys_matrix.end(irow);
         ++icol)
      row_scatter[icol->column()] = icol->value();

    // turn on filter for valid fills in this row
    for (auto icol = this->begin(irow); icol != this->end(irow); ++icol)
      row_filter[icol->column()] = true;

    // now loop through each column to the left of the diagonal on this row
    // and find the correspoding row with the same index
    for (auto icol = ++const_this->begin(irow);
         icol != one_after_diagonal[irow];
         ++icol) {
      const col_num_type icol_num = icol->column();

      // compute multiplier at this column
      ASSERT(!numerics::is_nearly_equal(this->diag_element(icol_num), 0.0),
             ExcDividedByZero());
      const value_type mult =
        row_scatter[icol_num] / this->diag_element(icol_num);
      row_scatter[icol_num] = mult;

      // subtract the entries right of this column number
      for (auto jcol = this->one_after_diagonal[icol_num];
           jcol != this->end(icol_num);
           ++jcol) {
        const size_type jcol_num = jcol->column();
        row_scatter[jcol_num] -=
          mult * jcol->value() * static_cast<value_type>(row_filter[jcol_num]);
      }
    }  // col_idx-loop


    // gather the values from the scattered vector
    for (auto icol = this->begin(irow); icol != this->end(irow); ++icol) {
      const row_num_type colnum = icol->column();
      icol->value() = row_scatter[colnum];
      row_filter[colnum] = false;
      row_scatter[colnum] = 0.0;
    }

#ifdef DEBUG
    // check row_filter and row_scatter is properly reset
    std::for_each(row_filter.begin(), row_filter.end(),
                  [](bool i) { ASSERT(i == false, ExcInternalErr()); });
    std::for_each(row_scatter.begin(), row_scatter.end(), [](value_type i) {
      ASSERT(numerics::is_zero(i), ExcInternalErr());
    });
#endif  // DEBUG //

  }  // irow-loop

#ifdef DEBUG
  for (row_num_type i = 0; i < N; ++i) {
    // check for zero diagonal
    value_type diag = this->diag_element(i);
    ASSERT(!numerics::is_zero(diag), ExcInternalErr());
  }
#endif  // DEBUG //
}


#ifdef USE_SERIAL_VMULT
template <typename NumberType>
template <typename OtherNumberType>
void SparseILU<NumberType>::vmult(
  dealii::Vector<OtherNumberType>& dst,
  const dealii::Vector<OtherNumberType>& src) const
{
  ASSERT_SAME_SIZE(dst, src);
  ASSERT(this->m() == src.size(), ExcSizeMismatch(this->m(), src.size()));

#ifdef PROFILING
  dealii::TimerOutput::Scope t(this->timer, "vmult");
#endif  // PROFILING //

  // solving the following system:
  // L * U * dst = src
  // by forward/backward substitution

  const auto N = this->m();

  dst = src;

  // Forward solve L * y = src
  // The diagonals on L matrix is 1
  for (row_num_type irow = 0; irow < N; ++irow) {
    OtherNumberType dstval = dst[irow];

    for (auto icol = ++this->begin(irow); icol != one_after_diagonal[irow];
         ++icol)
      dstval -= icol->value() * dst[icol->column()];

    dst[irow] = dstval;
  }

  // Backward solve U * x = y;
  // Iterate from the bottom to the top
  // due to data dependency
  for (row_num_type irow1 = N; irow1 > 0; --irow1) {
    row_num_type irow = irow1 - 1;
    const OtherNumberType diag = this->diag_element(irow);

    ASSERT(diag == this->begin(irow)->value(), ExcInternalErr());
    ASSERT(!numerics::is_zero(diag), ExcInternalErr());

    OtherNumberType dstval = dst[irow];

    for (auto icol = one_after_diagonal[irow]; icol != this->end(irow); ++icol)
      dstval -= icol->value() * dst[icol->column()];

    dst[irow] = dstval / diag;
  }
}
#else  // utilizing shared memory parallelization

template <typename NumberType>
template <typename OtherNumberType>
void SparseILU<NumberType>::vmult(
  dealii::Vector<OtherNumberType>& dst,
  const dealii::Vector<OtherNumberType>& src) const
{
#ifdef PROFILING
  dealii::TimerOutput::Scope t(this->timer, "vmult");
#endif  // PROFILING //

  ASSERT_SAME_SIZE(dst, src);
  ASSERT(this->m() == src.size(), ExcSizeMismatch(this->m(), src.size()));
  dst = src;
  {
    internal::ForwardSubstitution<NumberType> forward_subs(
      *this, this->block_size_per_thread);
    forward_subs.run(dst);
  }
  {
    internal::BackwardSubstitution<NumberType> backward_subs(
      *this, this->block_size_per_thread);
    backward_subs.run(dst);
  }
}

#endif  // USE_SERIAL_VMULT


template <typename NumberType>
template <typename OtherNumberType>
void SparseILU<NumberType>::Tvmult(
  dealii::Vector<OtherNumberType>& dst,
  const dealii::Vector<OtherNumberType>& src) const
{
  ASSERT_SAME_SIZE(dst, src);
  ASSERT(this->m() == src.size(), ExcSizeMismatch(this->m(), src.size()));

#ifdef PROFILING
  dealii::TimerOutput::Scope t(this->timer, "Tvmult");
#endif  // PROFILING //

  // This routine solves (L*U)' * dst = src
  // by forward/backward substution
  const size_type N = dst.size();

  dst = src;
  dealii::Vector<OtherNumberType> tmp_sum(N);

  // Forward solve U' * y = src
  for (size_type irow = 0; irow < N; ++irow) {
    dst[irow] = (dst[irow] - tmp_sum[irow]) / this->diag_element(irow);
    // update is now completed for rows = 0, 1, ... irow

    const OtherNumberType dst_row = dst[irow];
    for (auto it = one_after_diagonal[irow]; it != this->end(irow); ++it)
      tmp_sum[it->column()] += it->value() * dst_row;
  }  // irow-loop

  tmp_sum = 0.0;

  // Backward solve: L' * x = src
  // Note here the diagonals are 1.0
  for (size_type irow1 = N; irow1 > 0; --irow1) {
    size_type irow = irow1 - 1;
    dst[irow] -= tmp_sum[irow];
    // update is now completed for row = irow, irow + 1, ... N

    const OtherNumberType dst_row = dst[irow];

    // The problem here is that deal.II only implements forward iteration
    // throught the sparse matrix. We have to implement something
    // that emulate it. But this is slower that direct pointer access.
    /** \todo consider using offset from &diag_element(irow) */
    auto counter = one_after_diagonal[irow] - this->begin() - 1;
    (this->get_sparsity_pattern().begin(0))--;

    while (counter >= 0) {
      auto it = this->begin(irow) + counter;
      tmp_sum[it->column()] += dst_row * it->value();
      --counter;
    }
  }  // irow-loop
}


template <typename NumberType>
void SparseILU<NumberType>::print_sparsity_svg(std::ostream& out) const
{
  const unsigned int m = this->m();
  const unsigned int n = this->n();

  out << "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" "
         "viewBox=\"0 0 "
      << n + 2 << " " << m + 2
      << " \">\n"
         "   <rect width=\""
      << n + 2 << "\" height=\"" << m + 2
      << "\" fill=\"rgb(128, 128, 128)\"/>\n"
         "   <rect x=\"1\" y=\"1\" width=\""
      << n + 0.1 << "\" height=\"" << m + 0.1
      << "\" fill=\"rgb(255, 255, 255)\"/>\n\n";


  for (row_num_type irow = 0; irow < m; ++irow) {
    auto it_lof = row_level_of_fills[irow].cbegin();

    // left of diagonal
    for (auto it = ++this->begin(irow); it != this->one_after_diagonal[irow];
         ++it, ++it_lof)
      generate_svg_entry(out, *it_lof, it->row(), it->column());

    // take note of the diagonal lof position
    auto it_lof_diag = it_lof++;

    // right of diagonal
    for (auto it = this->one_after_diagonal[irow]; it != this->end(irow);
         ++it, ++it_lof)
      generate_svg_entry(out, *it_lof, it->row(), it->column());

    //  the diagonal entry
    generate_svg_entry(out, *it_lof_diag, this->begin(irow)->row(),
                       this->begin(irow)->column());
  }  // row loop

  out << "</svg>" << std::endl;
}


template <typename NumberType>
void SparseILU<NumberType>::generate_svg_entry(std::ostream& os,
                                               level_of_fill_type lof,
                                               row_num_type row,
                                               col_num_type col) const
{
  ASSERT(lof <= additional_data.max_level_of_fill, ExcInternalErr());
  ASSERT(
    additional_data.max_level_of_fill != invalid_level_of_fill,
    EXCEPT_MSG(
      "Level of fill has invalid number. You may need to run initialize()"));

  std::array<float, 3> rgb{0.0, 0.0, 0.0};
  if (lof)
    rgb = util::hsv_to_rgb(
      lof / static_cast<float>(additional_data.max_level_of_fill), 1.0f, 1.0f);

  os << "  <rect class=\"pixel\" fill=\"rgb(" << static_cast<int>(rgb[0] * 255)
     << ',' << static_cast<int>(rgb[1] * 255) << ','
     << static_cast<int>(rgb[2] * 255) << ")\" x=\"" << col + 1 << "\" y=\""
     << row + 1 << "\" width=\".9\" height=\".9\"/>\n";
}


template <typename NumberType>
void SparseILU<NumberType>::cache_diagonal_iterators()
{
  const row_num_type N = this->m();
  one_after_diagonal.clear();
  one_after_diagonal.resize(N, this->begin());
  bool inserted;

  for (row_num_type irow = 0; irow < N; ++irow) {
    inserted = false;
    // start from the second entry
    // because the first entry is the diagonal
    for (auto jcol = ++this->begin(irow); jcol != this->end(irow); ++jcol) {
      if (jcol->column() > irow) {
        ASSERT(inserted == false, ExcInternalErr());
        one_after_diagonal[irow] = jcol;
        inserted = true;
        break;
      }
    }  // column-loop

    // there are no elements right of diagonal
    if (!inserted) one_after_diagonal[irow] = this->end(irow);
  }  // row-loop

  ASSERT(one_after_diagonal.back() == this->end(N - 1), ExcInternalErr());
}


/* ------------------- */
namespace internal
/* ------------------- */
{
  /* ************************************************** */
  /* ForwardSubstitution */
  /* ************************************************** */
  template <typename NumberType>
  void ForwardSubstitution<NumberType>::compute_diagonal_block(size_type begin,
                                                               size_type end)
  {
#ifdef VERBOSE
    LOG_PREFIX("ForwardSubs");
    felspa_log << "Computing diagonal block: [" << begin << ", " << end << ')'
               << std::endl;
#endif  // VERBOSE //

    ASSERT(begin < end, ExcInternalErr());

    vector_type& dst = *this->ptr_dst;

    for (size_type irow = begin; irow < end; ++irow) {
      // destination
      value_type delta = 0.0;

      auto it_col = this->iterator_cache[irow];
      for (; it_col != this->ptr_ilu->post_diagonal(irow); ++it_col) {
        // only work on columns in our computation domain
        size_type col_num = it_col->column();
        if (col_num < begin) continue;
        delta -= it_col->value() * dst[col_num];
      }

      this->mtx.lock();
      this->iterator_cache[irow] = it_col;
      dst[irow] += delta;
      this->mtx.unlock();
    }
  }

  template <typename NumberType>
  void ForwardSubstitution<NumberType>::compute_off_diagonal_block(
    size_type row_begin, size_type row_end, size_type col_begin,
    size_type col_end)
  {
#ifdef VERBOSE
    LOG_PREFIX("ForwardSubs");
    felspa_log << "Computing lower-diagonal block: [" << row_begin << ", "
               << row_end << ") x [" << col_begin << ", " << col_end << ')'
               << std::endl;
#endif  // VERBOSE //

    // make sure the block is in a lower diagonal position
    // by make sure diagonal element is situated right of/at the col_end

    ASSERT(row_begin < row_end, ExcArgumentCheckFail());
    ASSERT(col_begin < col_end, ExcArgumentCheckFail());
    ASSERT(this->ptr_ilu->begin(col_end)->column() == col_end,
           ExcInternalErr());

    vector_type& dst = *this->ptr_dst;

    for (size_type irow = row_begin; irow < row_end; ++irow) {
      value_type delta = 0.0;

      auto it_col = this->iterator_cache[irow];

      for (; it_col != this->ptr_ilu->end(irow) && it_col->column() < col_end;
           ++it_col) {
        const size_type col_num = it_col->column();
        if (col_num < col_begin) continue;
        delta -= it_col->value() * dst[col_num];
      }

      dst[irow] += delta;
      this->iterator_cache[irow] = it_col;
    }
  }


  template <typename NumberType>
  void ForwardSubstitution<NumberType>::run(vector_type& dst)
  {
    // std::cout << "block size: " << this->block_size << std::endl;
    using this_type = ForwardSubstitution<NumberType>;
    const auto N = this->ptr_ilu->m();
    const unsigned int n_remainder = N % this->block_size;
    const unsigned int n_task_cycle =
      N / this->block_size + static_cast<unsigned int>(n_remainder != 0);
    this->ptr_dst = &dst;

    std::vector<size_type> block_sizes(n_task_cycle, this->block_size);
    if (n_remainder) block_sizes.back() = n_remainder;

    std::vector<size_type> block_bound;
    block_bound.push_back(0);
    std::partial_sum(block_sizes.crbegin(), block_sizes.crend(),
                     std::back_inserter(block_bound));


    for (auto it_col_front = block_bound.cbegin(),
              it_col_back = ++block_bound.cbegin();
         it_col_back != block_bound.cend(); ++it_col_front, ++it_col_back) {
      // first compute the diagonal block in serial
      compute_diagonal_block(*it_col_front, *it_col_back);

      // then partition the blocks right underneath the diagonal
      // and compute them in parallel
      std::vector<dealii::Threads::Thread<void>> threads;

      auto it_row_front = it_col_front, it_row_back = it_col_back;
      ++it_row_front;
      ++it_row_back;
      while (it_row_back != block_bound.cend())
        threads.push_back(dealii::Threads::new_thread(
          &this_type::compute_off_diagonal_block, *this, *it_row_front++,
          *it_row_back++, *it_col_front, *it_col_back));

      // synchonize and proceed
      for (auto& t : threads) t.join();
    }
  }

  /* ************************************************** */
  /* BackwardSubstitution */
  /* ************************************************** */
  template <typename NumberType>
  void BackwardSubstitution<NumberType>::compute_diagonal_block(size_type begin,
                                                                size_type end)
  {
    ASSERT(begin < end, ExcInternalErr());
    ASSERT(this->ptr_ilu->begin(begin)->column() == begin, ExcInternalErr());

#ifdef VERBOSE
    LOG_PREFIX("BackwardSubs");
    felspa_log << "Computing diagonal block: [" << begin << ", " << end << ')'
               << std::endl;
#endif  // VERBOSE //

    vector_type& dst = *this->ptr_dst;

    for (size_type irow1 = end; irow1 > begin; --irow1) {
      size_type irow = irow1 - 1;
      value_type delta = 0.0;

      auto it_col = this->ptr_ilu->post_diagonal(irow);
      for (; it_col != this->ptr_ilu->end(irow); ++it_col) {
        size_type col_num = it_col->column();
        if (col_num >= end) break;
        delta -= it_col->value() * dst[col_num];
      }
      dst[irow] += delta;

      ASSERT(!numerics::is_zero(this->ptr_ilu->diag_element(irow)),
             ExcDividedByZero());

      dst[irow] /= this->ptr_ilu->diag_element(irow);
    }
  }

  template <typename NumberType>
  void BackwardSubstitution<NumberType>::compute_off_diagonal_block(
    size_type row_begin, size_type row_end, size_type col_begin,
    size_type col_end)
  {
    ASSERT(row_begin < row_end, ExcInternalErr());
    ASSERT(col_begin < col_end, ExcInternalErr());
    ASSERT(this->ptr_ilu->begin(row_end)->column() <= col_begin,
           ExcInternalErr());

#ifdef VERBOSE
    LOG_PREFIX("BackwardSubs");
    felspa_log << "Computing upper-diagonal block: [" << row_begin << ", "
               << row_end << ") x [" << col_begin << ", " << col_end << ')'
               << std::endl;
#endif  // VERBOSE //

    vector_type& dst = *this->ptr_dst;

    for (size_type irow = row_begin; irow < row_end; ++irow) {
      value_type delta = 0.0;

      auto it_col = this->iterator_cache[irow];

      for (; it_col != this->ptr_ilu->end(irow) && it_col->column() < col_end;
           ++it_col) {
        const size_type col_num = it_col->column();
        if (col_num < col_begin) continue;
        delta -= it_col->value() * dst[col_num];
      }

      dst[irow] += delta;
    }  // irow -loop
  }


  template <typename NumberType>
  void BackwardSubstitution<NumberType>::run(vector_type& dst)
  {
    using this_type = BackwardSubstitution<NumberType>;
    const size_type N = this->ptr_ilu->m();
    const size_type n_remainder = N % this->block_size;

    this->ptr_dst = &dst;

    const unsigned int n_task_cycle =
      N / this->block_size + static_cast<unsigned int>(n_remainder != 0);

    std::vector<size_type> block_sizes(n_task_cycle, this->block_size);
    if (n_remainder) block_sizes.front() = n_remainder;

    std::vector<size_type> block_bound;
    block_bound.push_back(0);
    std::partial_sum(block_sizes.begin(), block_sizes.end(),
                     std::back_inserter(block_bound));
    std::for_each(block_bound.begin(), block_bound.end(),
                  [&](size_type& arg) { arg = N - arg; });

    for (auto it_col_front = block_bound.cbegin(),
              it_col_back = ++block_bound.cbegin();
         it_col_back != block_bound.end(); ++it_col_front, ++it_col_back) {
      // compute the diagonal
      compute_diagonal_block(*it_col_back, *it_col_front);

      // compute blocks that are above this diagonal block
      // dealii::Threads::TaskGroup<void> tasks;
      auto it_row_front = it_col_front, it_row_back = it_col_back;
      ++it_row_front;
      ++it_row_back;

      std::vector<dealii::Threads::Thread<void>> threads;
      while (it_row_back != block_bound.cend())
        threads.push_back(dealii::Threads::new_thread(
          &this_type::compute_off_diagonal_block, *this, *it_row_back++,
          *it_row_front++, *it_col_back, *it_col_front));

      for (auto& t : threads) t.join();
    }
    // std::exit(1);
  }
}  // namespace internal
FELSPA_NAMESPACE_CLOSE
#endif  // _FELSPA_LINEAR_ALGEBRA_ILU_LEVEL_OF_FILL_IMPLEMENT_H_ //
