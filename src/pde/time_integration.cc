#include <felspa/pde/time_integration.h>

FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/*  enum TempoCategory                                */
/* ************************************************** */
std::ostream& operator<<(std::ostream& os, TempoCategory category)
{
  switch (category) {
    case TempoCategory::exp:
      os << "explicit";
      break;
    case TempoCategory::imp:
      os << "implicit";
      break;
    case TempoCategory::mix:
      os << "mixed";
      break;
    default:
      THROW(ExcInternalErr());
  }
  return os;
}


/* ************************************************** */
/*  enum TempoMethod                                  */
/* ************************************************** */
std::ostream& operator<<(std::ostream& os, TempoMethod method)
{
  switch (method) {
    case TempoMethod::rktvd1:
      os << "1st order Runge Kutta TVD";
      break;
    case TempoMethod::rktvd2:
      os << "2nd order Runge Kutta TVD";
      break;
    case TempoMethod::rktvd3:
      os << "3rd order Runge Kutta TVD";
      break;
    default:
      THROW(ExcInternalErr());
  }
  return os;
}


/* ************************************************** */
/*               TempoIntegrator                      */
/* ************************************************** */
template <typename NumberType>
TempoIntegrator<NumberType>::TempoIntegrator()
  : time(0.0),
    step_count(0),
    substep_count(0),
    ptr_control(std::make_shared<TempoControl<NumberType>>())
{}


template <typename NumberType>
TempoIntegrator<NumberType>::TempoIntegrator(
  const std::shared_ptr<TempoControl<NumberType>>& pcontrol)
  : time(0.0), step_count(0), substep_count(0), ptr_control(pcontrol)
{}


template <typename NumberType>
TempoIntegrator<NumberType>::TempoIntegrator(
  const TempoIntegrator<NumberType>& that)
  : dealii::Subscriptor(that),
    time(that.time),
    step_count(that.step_count),
    substep_count(that.substep_count),
    ptr_control(std::make_shared<TempoControl<NumberType>>(*(that.ptr_control)))
{}


template <typename NumberType>
TempoIntegrator<NumberType>& TempoIntegrator<NumberType>::operator=(
  const TempoIntegrator& that)
{
  if (this != &that) {
    time = that.time;
    step_count = that.step_count;
    substep_count = that.substep_count;
    ptr_control =
      std::make_shared<TempoControl<NumberType>>(*(that.ptr_control));
  }
  return *this;
}


template <typename NumberType>
void TempoIntegrator<NumberType>::attach_control(
  const std::shared_ptr<TempoControl<NumberType>>& ctrl)
{
  ptr_control = ctrl;
}


template <typename NumberType>
void TempoIntegrator<NumberType>::initialize()
{
  time = 0.0;
  step_count = substep_count = 0;
  ptr_control->clear();
}


/* ************************************************** */
/*                 TempoControl                       */
/* ************************************************** */
template <typename NumberType>
TempoControl<NumberType>::TempoControl(TempoMethod method_,
                                       TempoCategory category_,
                                       value_type cfl_min_, value_type cfl_max_)
  : method(method_), category(category_), cfl_min(cfl_min_), cfl_max(cfl_max_)
{
  ASSERT(method_ != TempoMethod::undefined, ExcArgumentCheckFail());
  ASSERT(category_ != TempoCategory::undefined, ExcArgumentCheckFail());
  ASSERT(cfl_min_ > 0.0, ExcArgumentCheckFail());
  ASSERT(cfl_max_ > 0.0, ExcArgumentCheckFail());
  ASSERT(cfl_min_ < cfl_max_, ExcArgumentCheckFail());
}


template <typename NumberType>
void TempoControl<NumberType>::clear()
{
  n_substeps.clear();
  time_steps.clear();
}


template <typename NumberType>
void TempoControl<NumberType>::set_cfl(value_type cfl_min_, value_type cfl_max_)
{
  ASSERT(cfl_min_ > 0.0, ExcArgumentCheckFail());
  ASSERT(cfl_max_ > 0.0, ExcArgumentCheckFail());
  ASSERT(cfl_min_ < cfl_max_, ExcArgumentCheckFail());

  cfl_min = cfl_min_;
  cfl_max = cfl_max_;
}


template <typename NumberType>
void TempoControl<NumberType>::use_method(TempoMethod method_,
                                          TempoCategory category_)
{
  ASSERT(method_ != TempoMethod::undefined, ExcArgumentCheckFail());
  ASSERT(category_ != TempoCategory::undefined, ExcArgumentCheckFail());
  method = method_;
  category = category_;
}


template <typename NumberType>
bool TempoControl<NumberType>::is_initialized() const
{
  return method != TempoMethod::undefined &&
         category != TempoCategory::undefined && cfl_min > 0.0 &&
         cfl_max > 0.0 && cfl_min < cfl_max;
}


template <typename NumberType>
std::pair<TempoMethod, TempoCategory> TempoControl<NumberType>::query_method()
  const
{
  return {method, category};
}


template <typename NumberType>
auto TempoControl<NumberType>::get_cfl() const
  -> std::pair<value_type, value_type>
{
  return {cfl_min, cfl_max};
}


template <typename NumberType>
void TempoControl<NumberType>::scale_cfl(value_type coeff)
{
  ASSERT(coeff > 0.0, ExcArgumentCheckFail());
  cfl_min *= coeff;
  cfl_max *= coeff;
}


/* -------- Explicit Instantiations ----------*/
#include "time_integration.inst"
/* -------------------------------------------*/
FELSPA_NAMESPACE_CLOSE
