#ifndef _FELSPA_PHYSICS_FUNCTIONS_H_
#define _FELSPA_PHYSICS_FUNCTIONS_H_

#include <felspa/base/constants.h>
#include <felspa/base/function.h>

FELSPA_NAMESPACE_OPEN


template <int dim, typename NumberType>
class GravityFunction : public TensorFunction<1, dim, NumberType>
{
 public:
  using value_type = NumberType;
  using base_type = TensorFunction<1, dim, NumberType>;
  using typename base_type::gradient_type;
  using typename base_type::point_type;
  using typename base_type::tensor_type;

  /**
   * Constructor
   */
  GravityFunction(value_type gravity_const)
    : gravity_constant(gravity_const = constants::earth_gravity)
  {}


  /**
   *  Return gravity constant at each point
   */
  tensor_type evaluate(const point_type&) const override
  {
    tensor_type result;
    result[dim - 1] = gravity_constant;
    return result;
  }


  /**
   * Return gradient at each point: zero tensor
   */
  gradient_type evaluate_gradient(const point_type&) const override
  {
    gradient_type grad;
    for (auto it = grad.begin_raw(); it != grad.end_raw(); ++it) *it = 0.0;
    return grad;
  }


 protected:
  /**
   *  Gravity constant returned by this function
   */
  value_type gravity_constant;
};


FELSPA_NAMESPACE_CLOSE
#endif  // _FELSPA_PHYSICS_FUNCTIONS_H_ //