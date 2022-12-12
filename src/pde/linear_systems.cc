#include <felspa/pde/linear_systems.h>

FELSPA_NAMESPACE_OPEN

/* -------- Explicit Instantiations ----------*/
template class dg::DGLinearSystem<1, types::DoubleType>;
template class dg::DGLinearSystem<2, types::DoubleType>;
template class dg::DGLinearSystem<3, types::DoubleType>;

template class dg::DGLinearSystem<1, types::FloatType>;
template class dg::DGLinearSystem<2, types::FloatType>;
template class dg::DGLinearSystem<3, types::FloatType>;
/* -------------------------------------------*/

FELSPA_NAMESPACE_CLOSE
