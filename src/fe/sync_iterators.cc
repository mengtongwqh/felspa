#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/grid/tria_accessor.h>
#include <felspa/fe/sync_iterators.h>

FELSPA_NAMESPACE_OPEN

using dealii::TriaActiveIterator;
using dealii::DoFCellAccessor;


#if FELSPA_DEAL_II_VERSION_GTE(9, 3, 0)

template class SyncedIterators<TriaActiveIterator<DoFCellAccessor<1, 1, true>>>;
template class SyncedIterators<TriaActiveIterator<DoFCellAccessor<2, 2, true>>>;
template class SyncedIterators<TriaActiveIterator<DoFCellAccessor<3, 3, true>>>;

#else

using dealii::DoFHandler;
template class SyncedIterators<
  TriaActiveIterator<DoFCellAccessor<DoFHandler<1>, true>>>;
template class SyncedIterators<
  TriaActiveIterator<DoFCellAccessor<DoFHandler<2>, true>>>;
template class SyncedIterators<
  TriaActiveIterator<DoFCellAccessor<DoFHandler<3>, true>>>;

#endif


FELSPA_NAMESPACE_CLOSE
