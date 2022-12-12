#include <felspa/base/exceptions.h>
#include <felspa/base/utilities.h>
#include <felspa/level_set/geometry.h>

FELSPA_NAMESPACE_OPEN

/* -------------------------------------------*/
namespace ls
/* -------------------------------------------*/
{
  /* ************************************************** */
  /*                    HyperSphere                     */
  /* ************************************************** */
  template <int dim, typename NumberType>
  HyperSphere<dim, NumberType>::HyperSphere(const point_type& center_,
                                            value_type radius_)
    : ICBase<dim, NumberType>(true), center(center_), radius(radius_)
  {
    ASSERT(radius > 0.0, ExcArgumentCheckFail());
  }


  template <int dim, typename NumberType>
  auto HyperSphere<dim, NumberType>::evaluate(const point_type& pt) const
    -> value_type
  {
    return pt.distance(center) - radius;
  }


  /* ************************************************** */
  /*                   HyperPlane                       */
  /* ************************************************** */
  template <int dim, typename NumberType>
  auto HyperPlaneBase<dim, NumberType>::evaluate(const point_type& pt) const
    -> value_type
  {
    dealii::Tensor<1, dim, value_type> dist_vector = pt - ref_point;
    return -scalar_product(dist_vector, normal_vector);
  }


  template <typename NumberType>
  HyperPlane<2, NumberType>::HyperPlane(
    const point_type& pt,
    const dealii::Tensor<1, dim, value_type>& tangent_vector)
    : HyperPlaneBase<2, NumberType>(pt)
  {
    value_type tangent_norm = tangent_vector.norm();
    ASSERT(!numerics::is_nearly_equal(tangent_norm, 0.0),
           ExcArgumentCheckFail());

    // normalize the normal vector
    this->normal_vector[0] = -tangent_vector[1] / tangent_norm;
    this->normal_vector[1] = tangent_vector[0] / tangent_norm;
  }


  template <typename NumberType>
  HyperPlane<3, NumberType>::HyperPlane(
    const point_type& pt, const dealii::Tensor<1, dim, value_type>& t1,
    const dealii::Tensor<1, dim, value_type>& t2)
    : HyperPlaneBase<3, value_type>(pt)
  {
    this->normal_vector = dealii::cross_product_3d(t1, t2);
    value_type normal_norm = this->normal_vector.norm();
    ASSERT(!numerics::is_nearly_equal(normal_norm, 0.0), ExcDividedByZero());
    this->normal_vector /= normal_norm;
  }


  /* ************************************************** */
  /*                  HyperRectangle                    */
  /* ************************************************** */
  template <typename NumberType>
  auto HyperRectangle<1, NumberType>::evaluate(const point_type& pt) const
    -> value_type
  {
    if (pt(0) < this->pt1(0))
      return this->pt1(0) - pt(0);
    else if (pt(0) > this->pt2(0))
      return pt(0) - this->pt2(0);
    else
      return -std::min(pt(0) - this->pt1(0), this->pt2(0) - pt(0));
  }


  template <typename NumberType>
  auto HyperRectangle<2, NumberType>::evaluate(const point_type& pt) const
    -> value_type
  {
    const unsigned int dim = 2;
    value_type xyz[dim][2];

    for (unsigned int idim = 0; idim < dim; ++idim) {
      xyz[idim][0] = this->pt1(idim);
      xyz[idim][1] = this->pt2(idim);
    }

    point_type vtx[2][2];  // [lower/upper, left/right]
    for (unsigned int i = 0; i < 2; ++i) {
      for (unsigned int j = 0; j < 2; ++j) {
        vtx[i][j](0) = xyz[0][i];
        vtx[i][j](1) = xyz[1][j];
      }  // j-loop
    }    // i-loop

    if (pt(0) < this->pt1(0)) {
      if (pt(1) < this->pt1(1))
        return pt.distance(vtx[0][0]);
      else if (pt(1) > this->pt2(1))
        return pt.distance(vtx[0][1]);
      else
        return xyz[0][0] - pt(0);
    } else if (pt(0) > this->pt2(0)) {
      if (pt(1) < this->pt1(1))
        return pt.distance(vtx[1][0]);
      else if (pt(1) > this->pt2(1))
        return pt.distance(vtx[1][1]);
      else
        return pt(0) - xyz[0][1];
    } else {
      if (pt(1) < this->pt1(1))
        return this->pt1(1) - pt(1);
      else if (pt(1) > this->pt2(1))
        return pt(1) - this->pt2(1);
      else {
        value_type dist[4] = {this->pt1(0) - pt(0), pt(0) - this->pt2(0),
                              this->pt1(1) - pt(1), pt(1) - this->pt2(1)};
        return util::array_max(dist);
      }
    }
    THROW(ExcInternalErr());
  }


  template <typename NumberType>
  auto HyperRectangle<3, NumberType>::evaluate(const point_type& pt) const
    -> value_type
  {
    using util::norm;
    const int dim = 3;
    // list of corner points, [bottom/top, lower/upper, left/right]
    point_type vtx[2][2][2];
    value_type xyz[dim][2];

    for (unsigned int idim = 0; idim < dim; ++idim) {
      xyz[idim][0] = this->pt1(idim);
      xyz[idim][1] = this->pt2(idim);
    }

    for (unsigned int i = 0; i < 2; ++i) {
      for (unsigned int j = 0; j < 2; ++j) {
        for (unsigned int k = 0; k < 2; ++k) {
          vtx[i][j][k](0) = xyz[0][i];
          vtx[i][j][k](1) = xyz[1][j];
          vtx[i][j][k](2) = xyz[2][k];
        }  // k-loop
      }    // j-loop
    }      // i-loop

    // dimension by dimension, resulting in 27 cases...
    // very tedious... better way?
    if (pt(0) < xyz[0][0]) {
      if (pt(1) < xyz[1][0]) {
        if (pt(2) < xyz[2][0])
          return pt.distance(vtx[0][0][0]);
        else if (pt(2) > xyz[2][1])
          return pt.distance(vtx[0][0][1]);
        else
          return norm({pt(0) - xyz[0][0], pt(1) - xyz[1][0]});
      } else if (pt(1) > xyz[1][1]) {
        if (pt(2) < xyz[2][0])
          return pt.distance(vtx[0][1][0]);
        else if (pt(2) > xyz[2][1])
          return pt.distance(vtx[0][1][1]);
        else
          return norm({pt(0) - xyz[0][0], pt(1) - xyz[1][1]});
      } else {
        if (pt(2) < xyz[2][0])
          return norm({pt(0) - xyz[0][0], pt(2) - xyz[2][0]});
        else if (pt(2) > xyz[2][1])
          return norm({pt(0) - xyz[0][0], pt(2) - xyz[2][1]});
        else
          return xyz[0][0] - pt(0);
      }
    } else if (pt(0) > xyz[0][1]) {
      if (pt(1) < xyz[1][0]) {
        if (pt(2) < xyz[2][0])
          return pt.distance(vtx[1][0][0]);
        else if (pt(2) > xyz[2][1])
          return pt.distance(vtx[1][0][1]);
        else
          return norm({pt(0) - xyz[0][1], pt(1) - xyz[1][0]});
      } else if (pt(1) > xyz[1][1]) {
        if (pt(2) < xyz[2][0])
          return pt.distance(vtx[1][1][0]);
        else if (pt(2) > xyz[2][1])
          return pt.distance(vtx[1][1][1]);
        else
          return norm({pt(0) - xyz[0][1], pt(1) - xyz[1][1]});
      } else {
        if (pt(2) < xyz[2][0])
          return norm({pt(0) - xyz[0][1], pt(2) - xyz[2][0]});
        else if (pt(2) > xyz[2][1])
          return norm({pt(0) - xyz[0][1], pt(2) - xyz[2][1]});
        else
          return pt(0) - xyz[0][1];
      }
    } else {
      if (pt(1) < xyz[1][0]) {
        if (pt(2) < xyz[2][0])
          return norm({pt(1) - xyz[1][0], pt(2) - xyz[2][0]});
        else if (pt(2) > xyz[2][1])
          return norm({pt(1) - xyz[1][0], pt(2) - xyz[2][1]});
        else
          return xyz[1][0] - pt(1);
      } else if (pt(1) > xyz[1][1]) {
        if (pt(2) < xyz[2][0])
          return norm({pt(1) - xyz[1][1], pt(2) - xyz[2][0]});
        else if (pt(2) > xyz[2][1])
          return norm({pt(1) - xyz[1][1], pt(2) - xyz[2][1]});
        else
          return pt(1) - xyz[1][1];
      } else {
        if (pt(2) < xyz[2][0])
          return xyz[2][0] - pt[2];
        else if (pt(2) > xyz[2][1])
          return pt[2] - xyz[2][1];
        else {
          value_type dist[6] = {this->pt1(0) - pt(0), pt(0) - this->pt2(0),
                                this->pt1(1) - pt(1), pt(1) - this->pt2(1),
                                this->pt1(2) - pt(2), pt(2) - this->pt2(2)};
          return util::array_max(dist);
        }
      }
    }
    THROW(ExcInternalErr());
  }


  /* ************************************************** */
  /*                       Step                         */
  /* ************************************************** */
  template <int dim, typename NumberType>
  Step<dim, NumberType>::Step(const point_type& center_, value_type width_,
                              value_type height_)
    : ICBase<dim, NumberType>(false),
      center(center_),
      width(width_),
      height(height_)
  {
    ASSERT(width_ > 0.0, ExcArgumentCheckFail());
    ASSERT(height_ > 0.0, ExcArgumentCheckFail());
  }


  template <int dim, typename NumberType>
  auto Step<dim, NumberType>::evaluate(const point_type& pt) const -> value_type
  {
    point_type xx = static_cast<point_type>(pt - center);
    xx /= (width * 0.5);
    bool in_step = true;
    for (unsigned int idim = 0; idim < dim; ++idim)
      in_step = in_step && std::abs(xx(idim)) <= 1.0;
    return static_cast<value_type>(in_step) * height;
  }


  /* ************************************************** */
  /*                  CosineCone                        */
  /* ************************************************** */
  template <int dim, typename NumberType>
  CosineCone<dim, NumberType>::CosineCone(const point_type& center_,
                                          value_type radius_)
    : ICBase<dim, NumberType>(false), center(center_), radius(radius_)
  {
    ASSERT(radius > 0.0, ExcArgumentCheckFail());
  }


  template <int dim, typename NumberType>
  auto CosineCone<dim, NumberType>::evaluate(const point_type& pt) const
    -> value_type
  {
    using constants::PI;
    point_type xx = static_cast<point_type>(pt - center);
    xx /= radius;
    value_type phi = 1.0;
    bool in_cone = (xx.norm() <= 1.0);

    for (int idim = 0; idim < dim; ++idim)
      phi *= 0.5 * (1.0 + cos(PI * xx(idim))) * in_cone;

    return phi;
  }


  /* ************************************************** */
  /*             SmoothedPerturbedSphere                */
  /* ************************************************** */
  template <int dim, typename NumberType>
  SmoothPerturbedSphere<dim, NumberType>::SmoothPerturbedSphere(
    const point_type center_, value_type radius_, value_type perturb_coeff_,
    point_type ref_point)
    : ICBase<dim, NumberType>(false),
      center(center_),
      radius(radius_),
      perturb_coeff(perturb_coeff_),
      reference_point(ref_point)
  {
    ASSERT(radius > 0.0, ExcArgumentCheckFail());
    ASSERT(perturb_coeff > 0.0, ExcArgumentCheckFail());
  }


  template <int dim, typename NumberType>
  void SmoothPerturbedSphere<dim, NumberType>::set_reference_point(
    std::initializer_list<value_type> xyz)
  {
    ASSERT(xyz.size() == dim,
           EXCEPT_MSG("Point dimension must be the same as spatial dimension"));

    unsigned int i = 0;
    for (auto component : xyz) reference_point[i++] = component;
  }


  template <int dim, typename NumberType>
  auto SmoothPerturbedSphere<dim, NumberType>::evaluate(
    const point_type& pt) const -> value_type
  {
    value_type phi_circle = pt.norm() - this->radius;
    value_type phi_perturb =
      pt.distance_square(this->reference_point) + this->perturb_coeff;

    return phi_circle * phi_perturb;
  }


  /* ************************************************** */
  /*               CosineTensorProduct                  */
  /* ************************************************** */
  template <int dim, typename NumberType>
  CosineTensorProduct<dim, NumberType>::CosineTensorProduct(
    const point_type& center_, const point_type& period_, value_type amplitude_)
    : ICBase<dim, NumberType>(false),
      center(center_),
      period(period_),
      amplitude(amplitude_)
  {}


  template <int dim, typename NumberType>
  auto CosineTensorProduct<dim, NumberType>::evaluate(
    const point_type& pt) const -> value_type
  {
    using constants::PI;
    value_type val = amplitude;
    for (int idim = 0; idim < dim; ++idim)
      val *= cos(2 * PI * period(idim) * pt(idim));
    return val;
  }


  /* ************************************************** */
  /*               RayleighTaylorLower                  */
  /* ************************************************** */
  template <int dim, typename NumberType>
  RayleighTaylorLower<dim, NumberType>::RayleighTaylorLower(
    const point_type& center_, value_type period_, value_type amplitude_)
    : ICBase<dim, NumberType>(false),
      center(center_),
      period(period_),
      amplitude(amplitude_)
  {
    ASSERT(period > 0, ExcArgumentCheckFail());
    static_assert(dim == 2 || dim == 3, "Only implemented for dim 2 or 3");
  }


  template <int dim, typename NumberType>
  auto RayleighTaylorLower<dim, NumberType>::surface_fcn(
    const point_type& pt) const -> value_type
  {
    using constants::PI;
    if constexpr (dim == 2) {
      // use the standardized Rayleigh-Taylor perturbation
      value_type xx = pt[0] - center[0];
      return amplitude * std::cos(PI * xx / period) + center[1];
      // if (xx >= 0.5 * period || xx <= -0.5 * period) return center[1];
      // return amplitude * std::cos(2 * PI / period * xx) + amplitude +
      // center[1];
    } else if (dim == 3) {
      value_type xx = pt[0] - center[0];
      value_type yy = pt[1] - center[1];
      value_type radius = std::sqrt(xx * xx + yy * yy);
      if (radius >= 0.5 * period) return center[2];
      return amplitude * std::cos(2 * PI / period * radius) + amplitude +
             center[2];
    }
  }


  template <int dim, typename NumberType>
  auto RayleighTaylorLower<dim, NumberType>::evaluate(
    const point_type& pt) const -> value_type
  {
    using constants::PI;
    const value_type d = pt[dim - 1] - surface_fcn(pt);
    const value_type alpha = 2.0 * PI / period;
    ASSERT(!numerics::is_zero(period), ExcDividedByZero());
    return d;

    if constexpr (dim == 2) {
      value_type xx = pt[0] - center[0];

      if (xx >= 0.5 * period || xx <= -0.5 * period) return d;
      value_type slope = -alpha * amplitude * std::sin(alpha * xx);
      return d / std::sqrt(slope * slope + 1.0);

    } else if (dim == 3) {
      value_type xx = pt[0] - center[0];
      value_type yy = pt[1] - center[1];
      value_type radius = std::sqrt(xx * xx + yy * yy);

      // outside the perturbation
      if (radius >= 0.5 * period) return d;

      const value_type coeff =
        alpha * amplitude * std::sin(alpha * radius) / radius;

      dealii::Tensor<1, dim, value_type> dir_drv;
      dir_drv[0] = coeff * xx;
      dir_drv[1] = coeff * yy;
      dir_drv[2] = 1.0;
      return d / dir_drv.norm();
    }
  }
}  // namespace ls

/* -------- Explicit Instantiation -----------*/
#include "geometry.inst"
template class ls::HyperPlane<2>;
template class ls::HyperPlane<3>;
template class ls::RayleighTaylorLower<2>;
template class ls::RayleighTaylorLower<3>;
/* -------------------------------------------*/
FELSPA_NAMESPACE_CLOSE