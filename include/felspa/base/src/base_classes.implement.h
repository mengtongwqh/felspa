#ifndef _FELSPA_BASE_BASE_CLASSES_IMPLEMENT_H_
#define _FELSPA_BASE_BASE_CLASSES_IMPLEMENT_H_

#include <felspa/base/base_classes.h>
#include <felspa/base/exceptions.h>
#include <felspa/base/numerics.h>

FELSPA_NAMESPACE_OPEN

template <typename TimeStepType>
FELSPA_FORCE_INLINE void TimedPhysicsObject<TimeStepType>::advance_time(
  time_step_type time_step)
{
  this->phsx_time += time_step;
  ASSERT(this->phsx_time >= 0.0, ExcArgumentCheckFail());
}


template <typename TimeStepType>
template <typename T>
FELSPA_FORCE_INLINE bool TimedPhysicsObject<TimeStepType>::is_synced_with(
  const T& obj) const
{
  return numerics::is_nearly_equal(phsx_time, obj.get_time());
}


template <typename TimeStepType>
FELSPA_FORCE_INLINE bool TimedPhysicsObject<TimeStepType>::is_synced_with(
  time_step_type time_) const
{
  return numerics::is_nearly_equal(phsx_time, time_);
}

FELSPA_NAMESPACE_CLOSE
#endif  // _FELSPA_BASE_CLASSES_IMPLEMENT_H_//
