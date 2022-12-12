#ifndef _FESLPA_BASE_FUNCTION_H_
#define _FESLPA_BASE_FUNCTION_H_

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <felspa/base/base_classes.h>
#include <felspa/base/types.h>

#include <algorithm>
#include <vector>

FELSPA_NAMESPACE_OPEN


/* ************************************************** */
/**
 * Specialization of \class dealii::Function class
 * with scalar return value and overloaded \c operator()
 * \tparam dim
 * \tparam NumberType
 */
/* ************************************************** */
template <int dim, typename NumberType = types::DoubleType>
class ScalarFunction : public TimedPhysicsObject<NumberType>
{
 public:
  constexpr static int dimension = dim;
  using value_type = NumberType; /**< Return value type */
  using base_type = TimedPhysicsObject<NumberType>;
  using point_type = dealii::Point<dim>;
  using typename base_type::time_step_type;

  /**
   * Allow function to be converted to a dealii::Function
   */
  class DealIIFunctionAdaptor;


  /**
   * Virtual destructor
   */
  virtual ~ScalarFunction() = default;


  /**
   * Override this function to define the function
   */
  virtual value_type evaluate(const point_type& pt) const = 0;


  /**
   * Single point evaluation for \c operator() .
   */
  value_type operator()(const point_type& pt) const { return evaluate(pt); }


  /**
   * Evaluation at multiple points.
   */
  std::vector<value_type> operator()(const std::vector<point_type>& pts) const;


  /**
   * Put the multiple points into a result vector
   */
  template <typename VectorType>
  void operator()(const std::vector<point_type>& pts,
                  VectorType& results) const;


  /**
   * Convert the object to a \c dealii::Function.
   */
  operator const dealii::Function<dim, value_type>&() const;


 protected:
  /**
   * Constructor, hidden since this is a base class.
   */
  ScalarFunction(time_step_type time = 0.0) : base_type(true, time) {}


  /**
   * Copy Constructor. Hidden to prevent slicing.
   */
  ScalarFunction(const ScalarFunction<dim, value_type>& that) = default;


  /**
   * Unique pointer to function adaptor.
   */
  mutable std::unique_ptr<const DealIIFunctionAdaptor> ptr_fcn_adaptor =
    nullptr;
};


/* ************************************************** */
/**
 * Specialization of \class dealii::TensorFunction
 * with scalar return value and overloaded \c operator()
 * \tparam dim
 * \tparam NumberType
 */
/* ************************************************** */
template <int rank, int dim, typename NumberType = types::DoubleType>
class TensorFunction : public TimedPhysicsObject<NumberType>
{
 public:
  constexpr static const int space_dim = dim, dimension = dim;
  using base_type = TimedPhysicsObject<NumberType>;
  using value_type = NumberType;
  using point_type = dealii::Point<dim>;
  using tensor_type = dealii::Tensor<rank, dim, value_type>;
  using gradient_type = dealii::Tensor<rank + 1, dim, value_type>;
  using typename base_type::time_step_type;

  class DealIIFunctionAdaptor;

  /**
   * Virtual destructor
   */
  virtual ~TensorFunction() = default;


  /**
   * Override this function to do function computations
   */
  virtual tensor_type evaluate(const point_type& pt) const = 0;


  /**
   * Evaluation of a single point
   */
  tensor_type operator()(const point_type& pt) const { return evaluate(pt); }


  /**
   * Evaluation of multiple points and return in \c std::vector
   */
  std::vector<tensor_type> operator()(const std::vector<point_type>& pts) const;


  /**
   * Override this function to compute gradient.
   */
  virtual gradient_type evaluate_gradient(const point_type&) const;


  /**
   * Evaluate the gradient at a single point
   */
  gradient_type gradient(const point_type& pt) const
  {
    return evaluate_gradient(pt);
  }


  /**
   * Evaluation of multiple points and put results in place
   */
  template <typename VectorType>
  void operator()(const std::vector<point_type>& pts,
                  VectorType& results) const;


  /**
   * Conversion to dealii::TensorFunction<rank, dim, value_type>
   */
  operator const dealii::TensorFunction<rank, dim, NumberType>&() const;


 protected:
  /**
   * Constructor.
   */
  TensorFunction(time_step_type initial_time = 0.0)
    : base_type(true, initial_time)
  {}


  /**
   * Unique pointer to a conversion adaptor.
   */
  mutable std::unique_ptr<const DealIIFunctionAdaptor> ptr_fcn_adaptor =
    nullptr;
};

FELSPA_NAMESPACE_CLOSE

/* ------ IMPLEMENTATIONS ------- */
#include "src/function.implement.h"
/* ------------------------------ */
#endif  // _FESLPA_BASE_FUNCTION_H_