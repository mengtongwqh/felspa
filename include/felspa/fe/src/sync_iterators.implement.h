#ifndef _FELSPA_FE_SYNC_ITERATORS_IMPLEMENT_H_
#define _FELSPA_FE_SYNC_ITERATORS_IMPLEMENT_H_

#include <felspa/fe/sync_iterators.h>

FELSPA_NAMESPACE_OPEN

template <typename TriaIteratorType>
SyncedIterators<TriaIteratorType>::SyncedIterators(
  std::initializer_list<TriaIteratorType> iters)
  : iterators{iters}
{
  ASSERT(is_synchronized(), ExcIteratorsNotSynced());
}


template <typename TriaIteratorType>
void SyncedIterators<TriaIteratorType>::append(const TriaIteratorType& it)
{
#ifdef DEBUG
  if (!iterators.empty()) {
    ASSERT(it.state() != dealii::IteratorState::invalid,
           EXCEPT_MSG("Cannot add invalid iterators"));

    if (it.state() == dealii::IteratorState::past_the_end) {
      ASSERT(iterators[0].state() == dealii::IteratorState::past_the_end,
             ExcInternalErr());
    } else {
      ASSERT(it->index() == this->index() && it->level() == this->level(),
             EXCEPT_MSG(
               "appended iterator is not at the cell that the object is at."));
    }
  }
#endif

  iterators.push_back(it);
}


template <typename TriaIteratorType>
int SyncedIterators<TriaIteratorType>::index() const
{
  ASSERT_NON_EMPTY(iterators);
  ASSERT(is_synchronized(), ExcIteratorsNotSynced());
  return (*iterators.begin())->index();
}


template <typename TriaIteratorType>
int SyncedIterators<TriaIteratorType>::level() const
{
  ASSERT_NON_EMPTY(iterators);
  ASSERT(is_synchronized(), ExcIteratorsNotSynced());
  return (*iterators.begin())->level();
}


template <typename TriaIteratorType>
auto SyncedIterators<TriaIteratorType>::operator++()
  -> SyncedIterators<TriaIteratorType>&
{
  for (auto& iter : iterators) ++iter;
  return *this;
}


template <typename TriaIteratorType>
auto SyncedIterators<TriaIteratorType>::operator+(size_type n)
  -> SyncedIterators<TriaIteratorType>&
{
  using std::advance;
  for (auto& iter : iterators) advance(iter, n);
  return *this;
}


template <typename TriaIteratorType>
bool SyncedIterators<TriaIteratorType>::is_synchronized() const
{
  ASSERT_NON_EMPTY(iterators);
  if (iterators.size() == 1) return true;

  const auto cell_id = iterators[0]->index();
  const auto cell_level = iterators[0]->level();

  for (const auto& iter : iterators)
    if (iter->index() != cell_id || iter->level() != cell_level) return false;

  return true;
}


template <typename TriaIteratorType>
void SyncedIterators<TriaIteratorType>::synchronize_to(
  const TriaIteratorType& ref_iter)
{
  using namespace dealii::IteratorState;
  ASSERT(ref_iter->state() == valid, EXCEPT_MSG("Iterator is invalid"));

  const auto ref_id = ref_iter->index();
  const auto ref_level = ref_iter->level();
  const auto ptr_mesh = &ref_iter->get_triangulation();

  // TODO check the type of the accessor before assigning
  // Probably we need templatize this function
  for (auto& iter : iterators) {
    iter =
      TriaIteratorType(ptr_mesh, ref_level, ref_id, &iter->get_dof_handler());
    ASSERT(ptr_mesh == &iter->get_triangulation(), ExcMeshNotSame());
  }
}


template <typename TriaIteratorType>
bool operator==(const SyncedIterators<TriaIteratorType>& lhs,
                const SyncedIterators<TriaIteratorType>& rhs)
{
  return !(lhs != rhs);
}


template <typename TriaIteratorType>
bool operator!=(const SyncedIterators<TriaIteratorType>& lhs,
                const SyncedIterators<TriaIteratorType>& rhs)
{
  return lhs.get_iterators() != rhs.get_iterators();
}

FELSPA_NAMESPACE_CLOSE
#endif  // _FELSPA_FE_SYNC_ITERATORS_IMPLEMENT_H_ //
