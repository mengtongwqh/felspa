#include <felspa/level_set/material_stack.h>

FELSPA_NAMESPACE_OPEN

template class ls::MaterialStack<1, types::DoubleType>;
template class ls::MaterialStack<2, types::DoubleType>;
template class ls::MaterialStack<3, types::DoubleType>;

template class MaterialAccessor<ls::MaterialStack<1, types::DoubleType>>;
template class MaterialAccessor<ls::MaterialStack<2, types::DoubleType>>;
template class MaterialAccessor<ls::MaterialStack<3, types::DoubleType>>;

FELSPA_NAMESPACE_CLOSE
