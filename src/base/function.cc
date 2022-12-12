#include <felspa/base/function.h>

FELSPA_NAMESPACE_OPEN

// Explicit instantiate
// some commonly-used instances
template class ScalarFunction<1, types::DoubleType>;
template class ScalarFunction<2, types::DoubleType>;
template class ScalarFunction<3, types::DoubleType>;

template class TensorFunction<1, 1, types::DoubleType>;
template class TensorFunction<1, 2, types::DoubleType>;
template class TensorFunction<1, 3, types::DoubleType>;

FELSPA_NAMESPACE_CLOSE
