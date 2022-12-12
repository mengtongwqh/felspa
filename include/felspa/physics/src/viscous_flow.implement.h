#ifndef _FELSPA_PHYSICS_VISCOUS_FLOW_IMPLEMENT_H_
#define _FELSPA_PHYSICS_VISCOUS_FLOW_IMPLEMENT_H_

#include <felspa/physics/viscous_flow.h>

FELSPA_NAMESPACE_OPEN

template <int dim, typename NumberType>
SphericalInclusion<dim, NumberType>::SphericalInclusion(
  value_type radius,
  value_type matrix_viscosity_,
  value_type matrix_density_,
  value_type inclusion_viscosity_,
  value_type inclusion_density_,
  const std::string& label)
  : MaterialBase<dim, NumberType>(label),
    inclusion_radius(radius),
    matrix_density(matrix_density_),
    matrix_viscosity(matrix_viscosity_),
    inclusion_density(inclusion_density_),
    inclusion_viscosity(inclusion_viscosity_)
{
  ASSERT(radius > 0.0, ExcArgumentCheckFail());
  ASSERT(matrix_viscosity_ > 0.0, ExcArgumentCheckFail());
  ASSERT(matrix_density_ > 0.0, ExcArgumentCheckFail());
  ASSERT(inclusion_viscosity_ > 0.0, ExcArgumentCheckFail());
  ASSERT(inclusion_density_ > 0.0, ExcArgumentCheckFail());

  for (int idim = 0; idim < dim; ++idim) inclusion_center[idim] = 0.0;
}


template <int dim, typename NumberType>
void SphericalInclusion<dim, NumberType>::set_inclusion_center(
  const dealii::Point<dim, value_type>& pt)
{
  inclusion_center = pt;
}


template <int dim, typename NumberType>
void SphericalInclusion<dim, NumberType>::scalar_values(
  MaterialParameter mp, const PointsField<dim, value_type>& pfield,
  std::vector<value_type>& vals) const
{
  ASSERT_SAME_SIZE(pfield, vals);
  ASSERT_SAME_SIZE(*pfield.ptr_pts, vals);

  auto p_pts = pfield.ptr_pts->cbegin();

  if (mp == MaterialParameter::density) {
    for (auto& val : vals) {
      val = (p_pts->distance(inclusion_center) < inclusion_radius)
              ? inclusion_density
              : ((p_pts->norm() > inclusion_radius)
                   ? matrix_density
                   : 0.5 * (matrix_density + inclusion_density));
      ++p_pts;
    }
  } else if (mp == MaterialParameter::viscosity) {
    for (auto& val : vals) {
      val = (p_pts->distance(inclusion_center) < inclusion_radius)
              ? inclusion_viscosity
              : ((p_pts->norm() > inclusion_radius)
                   ? matrix_viscosity
                   : 0.5 * (matrix_viscosity + inclusion_viscosity));
      ++p_pts;
    }
  } else {
    THROW(ExcNotImplemented());
  }
}


template <int dim, typename NumberType>
void SphericalInclusion<dim, NumberType>::print(std::ostream& os) const
{
  os << "Spherical inclusion: " << this->get_label_string()
     << " | matrix/inclusion density = " << matrix_density << '/'
     << inclusion_density << " | "
     << " matrix/inclusion viscosity = " << matrix_viscosity << '/'
     << inclusion_viscosity << std::endl;
}

FELSPA_NAMESPACE_CLOSE

#endif  // _FELSPA_PHYSICS_VISCOUS_FLOW_IMPLEMENT_H_
