#include <felspa/level_set/velocity_field.h>

FELSPA_NAMESPACE_OPEN

/* -------------------------------------------*/
namespace ls
/* -------------------------------------------*/
{
  /* ************************************************** */
  /** RigidBodyTranslation */
  /* ************************************************** */
  template <int dim, typename NumberType>
  RigidBodyTranslation<dim, NumberType>::RigidBodyTranslation(
    std::initializer_list<value_type> velo_components)
  {
    auto iter = velo_components.begin();
    for (int idim = 0; idim < dim; ++idim) velo_field[idim] = *iter++;
  }


  template <int dim, typename NumberType>
  RigidBodyTranslation<dim, NumberType>::RigidBodyTranslation(
    const dealii::Tensor<1, dim, value_type>& velo)
    : velo_field(velo)
  {}


  /* ************************************************** */
  /** RigidBodyRotation */
  /* ************************************************** */
  template <typename NumberType>
  RigidBodyRotation<2, NumberType>::RigidBodyRotation(
    const point_type& center_, value_type angular_velocity_)
    : center(center_), angular_velocity(angular_velocity_)
  {}


  template <typename NumberType>
  auto RigidBodyRotation<2, NumberType>::evaluate(const point_type& pt) const
    -> tensor_type
  {
    point_type xx = static_cast<point_type>(pt - center);
    point_type velo_field;
    velo_field(0) = angular_velocity * xx(1);
    velo_field(1) = -angular_velocity * xx(0);
    return velo_field;
  }


  /* ************************************************** */
  /** \class SingleVortex */
  /* ************************************************** */
  template <typename NumberType>
  SingleVortex<2, NumberType>::SingleVortex(value_type period,
                                            const point_type& center_,
                                            value_type scaling_coeff_)
    : center(center_), T(period), scaling_coeff(scaling_coeff_)
  {
    ASSERT(period > 0.0, ExcArgumentCheckFail());
  }

  template <typename NumberType>
  auto SingleVortex<2, NumberType>::evaluate(const point_type& pt) const
    -> tensor_type
  {
    value_type sinx = sin(scaling_coeff * pt[0]);
    value_type cosx = cos(scaling_coeff * pt[0]);
    value_type siny = sin(scaling_coeff * pt[1]);
    value_type cosy = cos(scaling_coeff * pt[1]);
    value_type cost = cos(scaling_coeff * this->get_time() / T);

    tensor_type velo;
    velo[0] = -2.0 * sinx * sinx * cosy * cost;
    velo[1] = 2.0 * siny * siny * cosx * cost;

    return velo;
  }

}  // namespace ls

/* -------- Explicit Instantiation -----------*/
#include "velocity_field.inst"
/* -------------------------------------------*/
FELSPA_NAMESPACE_CLOSE
