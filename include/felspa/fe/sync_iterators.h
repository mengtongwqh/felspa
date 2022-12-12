#ifndef _FELSPA_FE_SYNC_ITERATORS_H_
#define _FELSPA_FE_SYNC_ITERATORS_H_

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <felspa/base/exceptions.h>
#include <felspa/base/felspa_config.h>

#include <initializer_list>

FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/**
 * \ingroup Exceptions
 */
/* ************************************************** */
DECL_EXCEPT_0(ExcIteratorsNotSynced,
              "Iterators are not synchronized to the same cell.");


/* ************************************************** */
/**
 * A vector of mesh iterators that will be
 * incremented/decremented synchronously.
 * This can be seen as an enhancement to
 * \c dealii::SynchronousIterators
 * by incorporating some \c CellIterator member functions.
 */
/* ************************************************** */
template <typename TriaIteratorType>
class SyncedIterators
{
 public:
  using tria_iterator_type = TriaIteratorType;
  using accessor_type = typename TriaIteratorType::AccessorType;
  using size_type = typename std::vector<TriaIteratorType>::size_type;

  static constexpr int dimension = accessor_type::dimension, dim = dimension;
  static constexpr int space_dimension = accessor_type::space_dimension;
  static constexpr int structure_dimension = accessor_type::structure_dimension;

  /**
   * Constructor.
   */
  SyncedIterators(std::initializer_list<TriaIteratorType> iters);


  /**
   * Add an iterator entry.
   */
  void append(const TriaIteratorType& iter);


  /**
   * Get the vector of iterators
   */
  const std::vector<TriaIteratorType> get_iterators() const
  {
    return iterators;
  }

  /**
   * Return i-th iterator of the synced iterators.
   */
  const TriaIteratorType& get(size_type i) const { return iterators[i]; }


  /**
   * Get the current level the iterator is on.
   * \pre The object is not empty.
   */
  int level() const;


  /**
   * Get the current index the iterator is on.
   * \pre The object is not empty.
   */
  int index() const;


  /**
   * \name Iterator Actions
   */
  //@{
  /**
   * Dereference the first iterator
   */
  accessor_type operator*()
  {
    ASSERT_NON_EMPTY(iterators);
    return *iterators[0];
  }

  /**
   * Dereference the first iterator, const overload
   */
  const accessor_type operator*() const
  {
    ASSERT_NON_EMPTY(iterators);
    return *iterators[0];
  }

  /**
   * Get members of the accessor pointed to by the first (primary) iterator.
   */
  tria_iterator_type operator->() { return iterators[0]; }

  /**
   * Same above, const overload.
   */
  const tria_iterator_type operator->() const { return iterators[0]; }

  /**
   * Get the i-th accessor
   */
  accessor_type& operator[](size_type i) { return *iterators[i]; }

  /**
   * Get the i-th accessor, const overload
   */
  const accessor_type& operator[](size_type i) const { return *iterators[i]; }

  /**
   * Increment all iterators by one step
   */
  SyncedIterators<TriaIteratorType>& operator++();

  /**
   * Increment all iterators by n steps
   */
  SyncedIterators<TriaIteratorType>& operator+(size_type n);
  //@}


  /**
   * Given an iterator of tria_iterator_type,
   * synchronize the all iterators to the cell pointed to by the iterator
   */
  void synchronize_to(const TriaIteratorType& iter);


  /**
   * Check that all iterators are pointing to the same cell
   */
  bool is_synchronized() const;


  /**
   * \name Conversion to Primary TriaIterator.
   */
  //@{
  /**
   * Conversion to a reference TriaIterator.
   * Return the first entry, which is the primary (leading) iterator.
   */
  operator TriaIteratorType&() { return *iterators.begin(); }
  // explicit operator TriaIteratorType&() { return *iterators.begin(); }

  /**
   * Conversion to a const reference to TriaIterator.
   * Const overload of the above.
   */
  // explicit operator const TriaIteratorType&() const
  operator const TriaIteratorType&() const
  {
    return *iterators.begin();
  }

  /**
   * Conversion operator to return a TriaIterator.
   */
  explicit operator TriaIteratorType() const
  {
    return TriaIteratorType(*iterators.begin());
  }
  //@}


  DECL_EXCEPT_0(ExcMeshNotSame,
                "Iterators pointing to different mesh cannot be admitted into "
                "SyncedMeshiterators");


 protected:
  /**
   * Vector of synchronized iterators.
   */
  std::vector<TriaIteratorType> iterators;
};


/**
 * Testing equality between iterators.
 */
template <typename TriaIteratorType>
bool operator==(const SyncedIterators<TriaIteratorType>& lhs,
                const SyncedIterators<TriaIteratorType>& rhs);

/**
 * Testing inequality between iterators.
 */
template <typename TriaIteratorType>
bool operator!=(const SyncedIterators<TriaIteratorType>& lhs,
                const SyncedIterators<TriaIteratorType>& rhs);


/* ************************************************** */
/**
 * Type aliases for array of active iterators.
 */
/* ************************************************** */
template <int dim>
using SyncedActiveIterators =
  SyncedIterators<typename dealii::DoFHandler<dim>::active_cell_iterator>;


FELSPA_NAMESPACE_CLOSE
/* ----------  Implementations  ------------ */
#include "src/sync_iterators.implement.h"
/* ----------------------------------------- */

#endif  // _FELSPA_FE_SYNC_ITERATORS_H_ //
