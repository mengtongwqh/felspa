#ifndef _FELSPA_PHYSICS_VISCOUS_FLOW_H_
#define _FELSPA_PHYSICS_VISCOUS_FLOW_H_

#include <felspa/physics/material_base.h>

FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/**
 * Material definition for the Newtonian flow.
 * Possess constant viscosity and density.
 */
/* ************************************************** */
template <int dim, typename NumberType>
class NewtonianFlow : public MaterialBase<dim, NumberType>
{
 public:
  using value_type = NumberType;


  /**
   * Constructor.
   */
  NewtonianFlow(value_type density, value_type viscosity,
                const std::string& id_string = "NewtonianMaterial")
    : MaterialBase<dim, value_type>(id_string),
      density_constant(density),
      viscosity_constant(viscosity)
  {
    ASSERT(density > 0.0, ExcArgumentCheckFail());
    ASSERT(viscosity > 0.0, ExcArgumentCheckFail());
  }


  /**
   * Return material parameters.
   */
  void scalar_values(MaterialParameter parameter_type,
                     const PointsField<dim, value_type>& pfield,
                     std::vector<value_type>& scalar_props) const override
  {
    UNUSED_VARIABLE(pfield);

    switch (parameter_type) {
      case MaterialParameter::density:
        std::fill(scalar_props.begin(), scalar_props.end(), density_constant);
        break;
      case MaterialParameter::viscosity:
        std::fill(scalar_props.begin(), scalar_props.end(), viscosity_constant);
        break;
      default:
        THROW(ExcNotImplemented());
    }
  }


  /**
   * Generate an accessor to obtain parameter values.
   */
  std::shared_ptr<MaterialAccessorBase<dim, NumberType>> generate_accessor(
    const dealii::Quadrature<dim>& quadrature) const override
  {
    return std::make_shared<MaterialAccessor<NewtonianFlow<dim, value_type>>>(
      *this, quadrature);
  }


  /**
   * Print the Newtonian flow info.
   */
  void print(std::ostream& os) const override
  {
    os << "Newtonian Flow: " << this->get_label_string()
       << " | density = " << density_constant << " | "
       << "viscosity = " << viscosity_constant << std::endl;
  }


 protected:
  value_type density_constant; /**< density */

  value_type viscosity_constant; /**< viscosity */
};


/* ************************************************** */
/**
 * This is a special case of a Newtonian material
 */
/* ************************************************** */
template <int dim, typename NumberType>
class SphericalInclusion : public MaterialBase<dim, NumberType>
{
 public:
  using value_type = NumberType;
  constexpr static int dimension = dim;


  /**
   * Constructor
   */
  SphericalInclusion(value_type radius = 0.25,
                     value_type matrix_density_ = 1000.,
                     value_type matrix_viscosity_ = 1.,
                     value_type inclusion_density_ = 1000.,
                     value_type inclusion_viscosity_ = 1000.,
                     const std::string& label = "SphericalInclusion");


  /**
   * Change the center of the inclusion.
   */
  void set_inclusion_center(const dealii::Point<dim, value_type>& pt);


  /**
   * Compute scalar material values
   */
  void scalar_values(MaterialParameter mp,
                     const PointsField<dim, value_type>& pfield,
                     std::vector<value_type>& vals) const override;

  /**
   * Generate an accessor to obtain parameter values.
   */
  std::shared_ptr<MaterialAccessorBase<dim, NumberType>> generate_accessor(
    const dealii::Quadrature<dim>& quadrature) const override
  {
    return std::make_shared<
      MaterialAccessor<SphericalInclusion<dim, value_type>>>(*this, quadrature);
  }


  /**
   * Print material info.
   */
  void print(std::ostream& os) const override;


 private:
  dealii::Point<dim, value_type> inclusion_center;

  value_type inclusion_radius;

  value_type matrix_density;

  value_type matrix_viscosity;

  value_type inclusion_density;

  value_type inclusion_viscosity;
};


FELSPA_NAMESPACE_CLOSE

/* ***** IMPLEMENTATIONS ***** */
#include "src/viscous_flow.implement.h"
/*******************************/

#endif  // _FELSPA_PHYSICS_VISCOUS_FLOW_H_ //
