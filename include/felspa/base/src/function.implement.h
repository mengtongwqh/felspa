#ifndef _FESLPA_BASE_FUNCTION_IMPLEMENT_H_
#define _FESLPA_BASE_FUNCTION_IMPLEMENT_H_

#include <felspa/base/function.h>

FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/*                  ScalarFunction                    */
/* ************************************************** */
/************************************/
/**  Adaptor to deal.II function.  */
/************************************/
template <int dim, typename NumberType>
class ScalarFunction<dim, NumberType>::DealIIFunctionAdaptor
  : public dealii::Function<dim, NumberType>
{
 public:
  /**
   * Constructor
   */
  DealIIFunctionAdaptor(const ScalarFunction<dim, value_type>& scalar_fcn_)
    : scalar_fcn(scalar_fcn_)
  {}

  /**
   *  Get values
   */
  value_type value(const dealii::Point<dim>& pt,
                   const unsigned int component = 0) const override;

 protected:
  /**
   * Const reference to ScalarFunction.
   */
  const ScalarFunction<dim, value_type>& scalar_fcn;
};


template <int dim, typename NumberType>
auto ScalarFunction<dim, NumberType>::DealIIFunctionAdaptor::value(
  const dealii::Point<dim>& pt, const unsigned int component) const
  -> value_type
{
  UNUSED_VARIABLE(component);
  ASSERT(component == 0,
         EXCEPT_MSG("Component must be 0 for scalar function."));
  return scalar_fcn(pt);
}


template <int dim, typename NumberType>
auto ScalarFunction<dim, NumberType>::operator()(
  const std::vector<point_type>& pts) const -> std::vector<value_type>
{
  std::vector<value_type> ret_vals;
  std::transform(pts.cbegin(), pts.cend(), std::back_inserter(ret_vals),
                 [this](const point_type& pt) { return (*this)(pt); });
  return ret_vals;
}


template <int dim, typename NumberType>
template <typename VectorType>
void ScalarFunction<dim, NumberType>::operator()(
  const std::vector<point_type>& pts, VectorType& results) const
{
  ASSERT_SAME_SIZE(pts, results);
  static_assert(
    std::is_convertible<value_type, typename VectorType::value_type>::value,
    "Result vector must hold data convertible from tensor_type");
  std::transform(pts.cbegin(), pts.cend(), results.begin(),
                 [this](const point_type& pt) { return (*this)(pt); });
}


template <int dim, typename NumberType>
ScalarFunction<dim, NumberType>::operator const dealii::Function<
  dim, NumberType>&() const
{
  if (ptr_fcn_adaptor == nullptr)
    ptr_fcn_adaptor = std::make_unique<DealIIFunctionAdaptor>(*this);
  return *ptr_fcn_adaptor;
}


/* ************************************************** */
/*                  TensorFunction                    */
/* ************************************************** */
/************************************/
/**  Adaptor to deal.II function.  */
/************************************/
template <int rank, int dim, typename NumberType>
class TensorFunction<rank, dim, NumberType>::DealIIFunctionAdaptor
  : public dealii::TensorFunction<rank, dim, NumberType>
{
 public:
  DealIIFunctionAdaptor(
    const TensorFunction<rank, dim, NumberType>& tensor_fcn_)
    : tensor_fcn(tensor_fcn_)
  {}


  tensor_type value(const dealii::Point<dim>& point) const override
  {
    return tensor_fcn(point);
  }


  gradient_type gradient(const dealii::Point<dim>& point) const override
  {
    return tensor_fcn.evaluate_gradient(point);
  }


 protected:
  const TensorFunction<rank, dim, NumberType>& tensor_fcn;
};


template <int rank, int dim, typename NumberType>
auto TensorFunction<rank, dim, NumberType>::operator()(
  const std::vector<point_type>& pts) const -> std::vector<tensor_type>
{
  std::vector<tensor_type> ret_vals;
  std::transform(pts.cbegin(), pts.cend(), std::back_inserter(ret_vals),
                 [&](const point_type& pt) { return (*this)(pt); });
  return ret_vals;
}


template <int rank, int dim, typename NumberType>
auto TensorFunction<rank, dim, NumberType>::evaluate_gradient(
  const point_type&) const -> gradient_type
{
  THROW(ExcUnimplementedVirtualFcn());
}


template <int rank, int dim, typename NumberType>
template <typename VectorType>
void TensorFunction<rank, dim, NumberType>::operator()(
  const std::vector<point_type>& pts, VectorType& results) const
{
  static_assert(
    std::is_convertible<tensor_type, typename VectorType::value_type>::value,
    "Result vector must hold data convertible from tensor_type");
  ASSERT_SAME_SIZE(pts, results);
  std::transform(pts.cbegin(), pts.cend(), results.begin(),
                 [&](const point_type& pt) { return (*this)(pt); });
}


template <int rank, int dim, typename NumberType>
TensorFunction<rank, dim, NumberType>::operator const dealii::TensorFunction<
  rank, dim, NumberType>&() const
{
  if (ptr_fcn_adaptor == nullptr)
    ptr_fcn_adaptor = std::make_unique<DealIIFunctionAdaptor>(*this);
  return *ptr_fcn_adaptor;
}

FELSPA_NAMESPACE_CLOSE
#endif  // _FESLPA_BASE_FUNCTION_IMPLEMENT_H_ //