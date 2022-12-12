#ifndef _FELSPA_BASE_BASE_CLASSES_H_
#define _FELSPA_BASE_BASE_CLASSES_H_

#include <deal.II/base/subscriptor.h>
#include <felspa/base/felspa_config.h>

FELSPA_NAMESPACE_OPEN

/*************************************************** */
/**
 * Base type for all objects that have a physics time
 * component
 */
/* ************************************************** */
template <typename TimeStepType>
class TimedPhysicsObject : public dealii::Subscriptor
{
 public:
  using time_step_type = TimeStepType;

  /**
   * Constructor.
   */
  explicit TimedPhysicsObject(bool allow_passive_update_ = true,
                              time_step_type time_ = TimeStepType())
    : phsx_time(time_), allow_passive_update(allow_passive_update_)
  {}


  /**
   * Set the boolean value for \c allow_passive_update
   */
  void set_allow_passive_update(bool flag = true)
  {
    allow_passive_update = flag;
  }


  /**
   * Get the \c allow_passive_update flag value
   */
  bool passive_update_allowed() const { return allow_passive_update; }


  /**
   * Copy constructor.
   */
  TimedPhysicsObject(const TimedPhysicsObject<TimeStepType>&) = default;


  /**
   * Copy assignment.
   */
  TimedPhysicsObject<TimeStepType>& operator=(
    const TimedPhysicsObject<TimeStepType>&) = default;


  /**
   * Virtual destructor.
   */
  virtual ~TimedPhysicsObject() = default;


  /**
   * Advance time for \c time_step.
   */
  virtual void advance_time(time_step_type time_step);


  /**
   * Set the time to \c time_.
   */
  virtual void set_time(time_step_type time_)
  {
    advance_time(time_ - this->get_time());
  }


  /**
   * Return the current time.
   */
  virtual time_step_type get_time() const { return phsx_time; }


  /**
   * Test if two objects with \c get_time() method is synchronized.
   */
  template <typename T>
  bool is_synced_with(const T& obj) const;


  /**
   * Test if the given time is the same as the time in this object
   */
  bool is_synced_with(time_step_type time_) const;


 protected:
  /**
   * Physics time.
   */
  time_step_type phsx_time;


  /**
   * When turned on, this object is a temporally passive object.
   * Its temporal behaviour can be updated by simulator in
   * \c set_time_temporal_passive_members().
   */
  bool allow_passive_update;
};

FELSPA_NAMESPACE_CLOSE

/* ----- IMPLEMENTATION ----- */
#include "src/base_classes.implement.h"
/* --------------------------- */

#endif // _FELSPA_BASE_BASE_CLASSES_H_ //