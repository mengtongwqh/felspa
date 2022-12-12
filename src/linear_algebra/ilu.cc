#include <felspa/linear_algebra/ilu.h>
#include <felspa/base/types.h>


/* ------------------- */
FELSPA_NAMESPACE_OPEN

/* ---------------- EXPLICIT INSTANTIATION -------------------- */
using namespace types;

template class SparseILU<DoubleType>;

template void SparseILU<DoubleType>::initialize<DoubleType>(
  const dealii::SparseMatrix<DoubleType>&, const AdditionalData&);

template void SparseILU<DoubleType>::vmult(
  dealii::Vector<DoubleType>& dst, const dealii::Vector<DoubleType>& src) const;

template void SparseILU<DoubleType>::Tvmult(
  dealii::Vector<DoubleType>& dst, const dealii::Vector<DoubleType>& src) const;
/* ------------------------------------------------------------ */


FELSPA_NAMESPACE_CLOSE
