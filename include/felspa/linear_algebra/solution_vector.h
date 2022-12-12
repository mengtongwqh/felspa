#ifndef _FELSPA_LINEAR_ALGEBRA_SOLUTION_VECTOR_H_
#define _FELSPA_LINEAR_ALGEBRA_SOLUTION_VECTOR_H_

#include <felspa/base/types.h>
#include <felspa/base/base_classes.h>
#include <felspa/base/felspa_config.h>

FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/**
 * A wrapper around a shared pointer to a solution vector.
 * A time is attached to pointer.
 */
/* ************************************************** */
template <typename VectorType,
          typename TimeStepType = typename VectorType::value_type>
class TimedSolutionVector : public TimedPhysicsObject<TimeStepType>
{
 public:
  using vector_type = VectorType;
  using value_type = typename vector_type::value_type;
  using time_step_type = TimeStepType;


  /** \name Basic Object Behavior */
  //@{
  /**
   * Constructor
   */
  TimedSolutionVector(const std::shared_ptr<vector_type>& pvector,
                      time_step_type current_time = 0.0);

  /**
   * Copy constructor. Shallow copy.
   */
  TimedSolutionVector(
    const TimedSolutionVector<VectorType, TimeStepType>& that);

  /**
   * Copy assignment.
   */
  TimedSolutionVector<VectorType, TimeStepType>& operator=(
    const TimedSolutionVector<VectorType, TimeStepType>& that);

  /**
   * Default Destructor.
   */
  ~TimedSolutionVector() = default;

  /**
   * Allocate a fresh block of memory for the shared pointer.
   */
  void reinit(const std::shared_ptr<vector_type>& pvector,
              time_step_type current_time = 0.0);
  //@}


  /**\name Accessing/Updating the underlying vector */
  //@{
  /**
   * Convert to a reference to underlying vector
   */
  explicit operator vector_type&();

  /**
   * Const overload of the function above
   */
  explicit operator const vector_type&() const;

  /**
   * Getting the underlying vector
   */
  vector_type& operator*();

  /**
   * Getting the underlying vector. const overload
   */
  const vector_type& operator*() const;

  /**
   * Allows us to call methods of the underlying vector.
   */
  vector_type* operator->() { return ptr_vector.get(); }

  /**
   * Same above, const overload
   */
  const vector_type* operator->() const { return ptr_vector.get(); }

  /**
   * Update the solution vector and time. Const Lvalue.
   */
  void update(const vector_type& soln, time_step_type current_time);

  /**
   * Update the solution vector and time. Rvalue reference.
   */
  void update(vector_type&& soln, time_step_type current_time);
  //@}


  /**
   * The object which allocates the memory is the independent one.
   * All object shallow-copied from the primary one is dependent.
   */
  bool is_independent() const { return independent; }


  /** \name Import functions to modify time */
  //@{
  using TimedPhysicsObject<time_step_type>::get_time;

  using TimedPhysicsObject<time_step_type>::set_time;

  using TimedPhysicsObject<time_step_type>::advance_time;
  //@}

 private:
  /**
   * Pointer to solution vector
   */
  std::shared_ptr<vector_type> ptr_vector;


  /**
   * If the TimedSolutionVector is independent,
   * it will be reallocated in setup_system().
   */
  bool independent;
};


/* ************************************************** */
/*    TimedSolutionVector<VectorType, TimeStepType>   */
/* ************************************************** */
template <typename VectorType, typename TimeStepType>
TimedSolutionVector<VectorType, TimeStepType>::TimedSolutionVector(
  const std::shared_ptr<vector_type>& pvector, time_step_type current_time)
  : TimedPhysicsObject<TimeStepType>(current_time),
    ptr_vector(pvector),
    independent(true)
{}


template <typename VectorType, typename TimeStepType>
TimedSolutionVector<VectorType, TimeStepType>::TimedSolutionVector(
  const TimedSolutionVector<VectorType, TimeStepType>& that)
  : TimedPhysicsObject<TimeStepType>(that),
    ptr_vector(that.ptr_vector),
    independent(false)
{}


template <typename VectorType, typename TimeStepType>
TimedSolutionVector<VectorType, TimeStepType>&
TimedSolutionVector<VectorType, TimeStepType>::operator=(
  const TimedSolutionVector<VectorType, TimeStepType>& that)
{
  if (this != &that) {
    TimedPhysicsObject<TimeStepType>::operator=(that);
    ptr_vector = that.ptr_vector;
    independent = false;
  }
  return *this;
}


template <typename VectorType, typename TimeStepType>
void TimedSolutionVector<VectorType, TimeStepType>::reinit(
  const std::shared_ptr<vector_type>& pvector, time_step_type current_time)
{
  set_time(current_time);
  if (ptr_vector != pvector) {
    ptr_vector = pvector;
    independent = true;
  }
}


template <typename VectorType, typename TimeStepType>
TimedSolutionVector<VectorType, TimeStepType>::operator vector_type&()
{
  // return const_cast<vector_type&>(static_cast<const vector_type&>(*this));
  ASSERT(ptr_vector != nullptr, ExcNullPointer());
  return *ptr_vector;
}


template <typename VectorType, typename TimeStepType>
TimedSolutionVector<VectorType, TimeStepType>::operator const vector_type&()
  const
{
  ASSERT(ptr_vector != nullptr, ExcNullPointer());
  return *ptr_vector;
}


template <typename VectorType, typename TimeStepType>
auto TimedSolutionVector<VectorType, TimeStepType>::operator*() const
  -> const vector_type&
{
  return static_cast<const vector_type&>(*this);
}

template <typename VectorType, typename TimeStepType>
auto TimedSolutionVector<VectorType, TimeStepType>::operator*() -> vector_type&
{
  return static_cast<vector_type&>(*this);
}

template <typename VectorType, typename TimeStepType>
void TimedSolutionVector<VectorType, TimeStepType>::update(
  const vector_type& soln, time_step_type current_time)

{
  *ptr_vector = soln;
  this->set_time(current_time);
}


template <typename VectorType, typename TimeStepType>
void TimedSolutionVector<VectorType, TimeStepType>::update(
  vector_type&& soln, time_step_type current_time)
{
  ptr_vector->swap(soln);
  this->set_time(current_time);
}

FELSPA_NAMESPACE_CLOSE
#endif  // _FELSPA_LINEAR_ALGEBRA_SOLUTION_VECTOR_H_ //
