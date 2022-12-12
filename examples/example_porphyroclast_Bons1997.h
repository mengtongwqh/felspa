
#ifndef _FELSPA_EXAMPLE_PORPHYROCLAST_BONS1997_
#define _FELSPA_EXAMPLE_PORPHYROCLAST_BONS1997_

#include <felspa/pde/boundary_conditions.h>

using namespace felspa;

template <int dim, typename NumberType>
class TopBottomGeometry : public BCGeometry<dim, NumberType>
{
 public:
  using value_type = NumberType;
  TopBottomGeometry(const dealii::Point<dim, value_type>& bottom,
                    const dealii::Point<dim, value_type>& top)
    : lower_left(bottom), upper_right(top)
  {}

  bool at_boundary(const dealii::Point<dim, value_type>& pt) const override
  {
    return numerics::is_zero(pt[dim - 1] - upper_right[dim - 1]) ||
           numerics::is_zero(pt[dim - 1] - lower_left[dim - 1]);
  }

 private:
  const dealii::Point<dim, value_type>& lower_left;
  const dealii::Point<dim, value_type>& upper_right;
};


template <int dim, typename NumberType>
class TopBottomBC : public BCFunction<dim, NumberType>
{
 public:
  using value_type = NumberType;

  TopBottomBC(const dealii::Point<dim, value_type>& bottom,
              const dealii::Point<dim, value_type>& top,
              NumberType velo_magnitude)
    : BCFunction<dim, value_type>(BCCategory::dirichlet, dim + 1,
                                  std::vector<bool>(dim + 1, false)),
      lower_left(bottom),
      upper_right(top),
      velocity_magnitude(velo_magnitude)
  {
    this->component_mask.set(0, true);
    this->set_geometry(
      std::make_shared<TopBottomGeometry<dim, value_type>>(bottom, top));
  }


  value_type value(const dealii::Point<dim, value_type>& pt,
                   unsigned int component) const override
  {
    if (component == 0) {
      if (numerics::is_zero(pt[dim - 1] - lower_left[dim - 1]))
        return -velocity_magnitude;
      if (numerics::is_zero(pt[dim - 1] - upper_right[dim - 1]))
        return velocity_magnitude;
      THROW(ExcInternalErr());
    }
    return 0.0;
  }

 private:
  const dealii::Point<dim, value_type>& lower_left;
  const dealii::Point<dim, value_type>& upper_right;
  const value_type velocity_magnitude;
};

template <int dim, typename NumberType>
class TopBottomSimpleShearBC : public TopBottomBC<dim, NumberType>
{
 public:
  using value_type = NumberType;
  TopBottomSimpleShearBC(const dealii::Point<dim, value_type>& bottom,
                       const dealii::Point<dim, value_type>& top,
                       NumberType velo_magnitude)
    : TopBottomBC<dim, NumberType>(bottom, top, velo_magnitude)
  {
    for (int idim = 0; idim != dim; ++idim)
      this->component_mask.set(idim, true);
  }
};


template <int dim, typename NumberType>
class LeftRightGeometry : public BCGeometry<dim, NumberType>
{
 public:
  using value_type = NumberType;
  LeftRightGeometry(const dealii::Point<dim>& left,
                    const dealii::Point<dim>& right)
    : lower_left_back(left), upper_right_front(right)
  {}


  bool at_boundary(const dealii::Point<dim>& pt) const override
  {
    return numerics::is_zero(pt[0] - lower_left_back[0]) ||
           numerics::is_zero(pt[0] - upper_right_front[0]);
  }


 private:
  const dealii::Point<dim, value_type>& lower_left_back;
  const dealii::Point<dim, value_type>& upper_right_front;
};


template <int dim, typename NumberType>
class LeftRightBC : public BCFunction<dim, NumberType>
{
 public:
  using value_type = NumberType;

  LeftRightBC(const dealii::Point<dim, value_type>& left,
              const dealii::Point<dim, value_type>& right)
    : BCFunction<2, NumberType>(BCCategory::dirichlet, dim,
                                std::vector<bool>(dim + 1, false)),
      lower_left(left),
      upper_right(right)
  {
    for (int idim = 1; idim < dim; ++idim) this->component_mask.set(idim, true);
    this->set_geometry(
      std::make_shared<LeftRightGeometry<dim, value_type>>(left, right));
  }

  value_type value(const dealii::Point<dim, value_type>& pt,
                   unsigned int component) const override
  {
    return 0.0;
  }

 private:
  const dealii::Point<dim, value_type>& lower_left;
  const dealii::Point<dim, value_type>& upper_right;
};


#endif  // _FELSPA_EXAMPLE_PORPHYROCLAST_BONS1997_ //