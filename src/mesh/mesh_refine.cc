#include <felspa/base/exceptions.h>
#include <felspa/mesh/mesh_refine.h>

FELSPA_NAMESPACE_OPEN

template class MeshRefiner<1, types::DoubleType>;
template class MeshRefiner<2, types::DoubleType>;
template class MeshRefiner<3, types::DoubleType>;

FELSPA_NAMESPACE_CLOSE
