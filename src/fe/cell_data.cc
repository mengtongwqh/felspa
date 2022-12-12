#include <felspa/fe/cell_data.h>

FELSPA_NAMESPACE_OPEN

std::ostream& operator<<(std::ostream& os, FEValuesEnum assembly_entity)
{
  switch (assembly_entity) {
    case FEValuesEnum::cell:
      os << "Cell";
      break;
    case FEValuesEnum::face:
      os << "Face";
      break;
    case FEValuesEnum::subface:
      os << "Subface";
      break;
    default:
      THROW(ExcInternalErr());
  }
  return os;
}


// explicit instantiation
// template class CellScratchDataBase<1>;
template class CellScratchData<2>;
template class CellScratchData<3>;
template class CellCopyData<2, types::DoubleType>;
template class CellCopyData<3, types::DoubleType>;

FELSPA_NAMESPACE_CLOSE
