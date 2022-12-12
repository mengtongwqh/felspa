#include <felspa/physics/viscous_flow.h>

FELSPA_NAMESPACE_OPEN

template class NewtonianFlow<1, types::DoubleType>;
template class NewtonianFlow<2, types::DoubleType>;
template class NewtonianFlow<3, types::DoubleType>;

template class SphericalInclusion<1, types::DoubleType>;
template class SphericalInclusion<2, types::DoubleType>;
template class SphericalInclusion<3, types::DoubleType>;

FELSPA_NAMESPACE_CLOSE
