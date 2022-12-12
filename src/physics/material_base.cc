#include <felspa/physics/material_base.h>

FELSPA_NAMESPACE_OPEN


std::ostream& operator<<(std::ostream& os, MaterialParameter parameter)
{
  switch (parameter) {
    case MaterialParameter::density:
      os << "density";
      break;
    case MaterialParameter::viscosity:
      os << "viscosity";
      break;
    default:
      THROW(ExcNotImplemented());
  }
  return os;
}

FELSPA_NAMESPACE_CLOSE