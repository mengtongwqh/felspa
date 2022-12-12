#include <felspa/pde/pde_tools.h>

FELSPA_NAMESPACE_OPEN

template struct VelocityExtractor<TensorFunction<1, 1, types::DoubleType>>;
template struct VelocityExtractor<TensorFunction<1, 2, types::DoubleType>>;
template struct VelocityExtractor<TensorFunction<1, 3, types::DoubleType>>;

template class CFLEstimator<TensorFunction<1, 1, types::DoubleType>>;
template class CFLEstimator<TensorFunction<1, 2, types::DoubleType>>;
template class CFLEstimator<TensorFunction<1, 3, types::DoubleType>>;

FELSPA_NAMESPACE_CLOSE
