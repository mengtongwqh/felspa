#ifndef _FELSPA_TEST_TEST_RAYLEIGH_TAYLOR_H_
#define _FELSPA_TEST_TEST_RAYLEIGH_TAYLOR_H_

#define PROFILING

#include <deal.II/base/multithread_info.h>
#include <deal.II/grid/grid_generator.h>
#include <felspa/base/quadrature.h>
#include <felspa/coupled/level_set_stokes.h>
#include <felspa/pde/boundary_conditions.h>
#include <felspa/physics/functions.h>
#include <felspa/physics/material_base.h>
#include <felspa/physics/viscous_flow.h>

using namespace felspa;
using dealii::Point;


/**
 * @brief Geometry for the top and bottom boundaries
 */
template <int dim, typename NumberType>
class TopBottomGeometry : public BCGeometry<dim, NumberType>
{
 public:
  using value_type = NumberType;
  using number_type = NumberType;
  using point_type = Point<dim, NumberType>;

  TopBottomGeometry(const point_type& lower_, const point_type& upper_)
    : lower(lower_), upper(upper_)
  {}

  bool at_boundary(const point_type& pt) const override
  {
    using numerics::is_equal;
    int idim = dim - 1;
    if (is_equal(pt[idim], lower[idim]) || is_equal(pt[idim], upper[idim]))
      return true;
    return false;
  }

 private:
  point_type lower;
  point_type upper;
};


/**
 * @brief Geometry for the vertical boundaries
 */
template <int dim, typename NumberType>
class VerticalGeometry : public BCGeometry<dim, NumberType>
{
 public:
  using value_type = NumberType;
  using number_type = NumberType;
  using point_type = Point<dim, NumberType>;

  VerticalGeometry(const point_type& lower_, const point_type& upper_)
    : lower(lower_), upper(upper_)
  {}

  bool at_boundary(const point_type& pt) const override
  {
    using numerics::is_equal;
    for (int idim = 0; idim != dim - 1; ++idim)
      if (is_equal(pt[idim], lower[idim]) || is_equal(pt[idim], upper[idim])) 
        return true;
      
    return false;
  }

 private:
  point_type lower;
  point_type upper;
};

/**
 * @brief Free-slip boundary condition
 */
template <int dim, typename NumberType>
class FreeSlipBC : public BCFunction<dim, NumberType>
{
 public:
  using point_type = Point<dim, NumberType>;
  FreeSlipBC(const point_type& lower_, const point_type& upper_)
    : BCFunction<dim, NumberType>(BCCategory::no_normal_flux, dim + 1)
  {
    this->set_geometry(
      std::make_shared<VerticalGeometry<dim, NumberType>>(lower_, upper_));
  }
};

/**
 * @brief No-slip boundary condition
 */
template <int dim, typename NumberType>
class NoSlipBC : public BCFunction<dim, NumberType>
{
 public:
  using value_type = NumberType;
  using number_type = NumberType;
  using point_type = Point<dim, number_type>;

  NoSlipBC(const point_type& lower_, const point_type& upper_)
    : BCFunction<dim, NumberType>(BCCategory::dirichlet, dim + 1,
                                  make_component())
  {
    this->set_geometry(
      std::make_shared<TopBottomGeometry<dim, NumberType>>(lower_, upper_));
  }

  number_type value(const point_type&, unsigned int) const override
  {
    return 0.0;
  }

  void vector_value(const point_type& p,
                    dealii::Vector<double>& values) const override
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values(c) = this->value(p, c);
  }

 private:
  static std::vector<bool> make_component()
  {
    std::vector<bool> component(dim + 1, true);
    component[dim] = false;
    return component;
  }
};

/**
 * @brief Parameters for the Rayleigh-Taylor test case
 * @tparam dim
 * @tparam NumberType
 */
template <int dim, typename NumberType>
class RayleighTaylorParameters
{
 public:
  using value_type = NumberType;
  using number_type = NumberType;
  using point_type = Point<dim, number_type>;

  RayleighTaylorParameters();

  point_type box_lower;
  point_type box_upper;

  unsigned int max_refine_level;
  unsigned int min_refine_level;
  bool use_adaptive_refinement =true;

  StokesSolutionMethod solution_method;


  value_type upper_density;
  value_type lower_density;
  value_type upper_viscosity;
  value_type lower_viscosity;

  int level_of_fill = 0;

  std::vector<unsigned int> subdivisions;
};

template <int dim, typename NumberType>
RayleighTaylorParameters<dim, NumberType>::RayleighTaylorParameters()
  : subdivisions(dim, 2)
{
  if constexpr (dim == 2) {
    box_lower = dealii::Point<dim, value_type>(0.0, 0.0);
    box_upper = dealii::Point<dim, value_type>(0.9142, 1.0);

    upper_density = 1010.;
    lower_density = 1000.;
    upper_viscosity = 100.0;
    lower_viscosity = 10.0;

    max_refine_level = 5;  // the finest level is 256x256 elements
    min_refine_level = 1;  // the finest level is 256x256 elements
    // max_refine_level = 6;  // the finest level is 256x256 elements
  }

  else if constexpr (dim == 3) {
    box_lower = dealii::Point<dim, value_type>(0.0, 0.0, 0.0);
    box_upper = dealii::Point<dim, value_type>(1.0, 1.0, 1.0);

    upper_density = 1010.;
    lower_density = 1000.;
    upper_viscosity = 100.0;
    lower_viscosity = 1.0;

    max_refine_level = 3;
    min_refine_level = 1;
  }
}

#endif  // _FELSPA_TEST_TEST_RAYLEIGH_TAYLOR_H_ //
