#ifndef _FELSPA_PDE_TIME_INTEGRATION_IMPLEMENT_H_
#define _FELSPA_PDE_TIME_INTEGRATION_IMPLEMENT_H_

#include <felspa/base/log.h>
#include <felspa/base/numerics.h>
#include <felspa/pde/time_integration.h>

FELSPA_NAMESPACE_OPEN

/* --------------------------------------------------*/
/** \class TempoIntegrator */
/* --------------------------------------------------*/

template <typename NumberType>
template <typename TempoSchemeType, TempoCategory category,
          typename SimulatorType>
auto TempoIntegrator<NumberType>::advance_time_step(SimulatorType& simulator,
                                                    value_type time_step)
  -> value_type
{
  if (numerics::is_zero(time_step)) return time_step;

  ASSERT(time_step >= 0.0, ExcArgumentCheckFail());
  LOG_PREFIX("TempoIntegrator");

  // integrate this time step
  TempoIntegrationSelector<TempoSchemeType, category>::integrate(
    simulator, time_step, time);

  // register the time step into the TempoIntegrator class
  time += time_step;
  ++step_count;
  ++substep_count;

  // make sure simulator is synchronized
  ASSERT(simulator.is_synchronized(), ExcSimulatorNotSync());
  felspa_log << "One time step " << time_step
             << " taken. Current simulation time = " << time << std::endl;

  return time_step;
}


template <typename NumberType>
template <typename TempoSchemeType, TempoCategory category,
          typename SimulatorType>
auto TempoIntegrator<NumberType>::advance_time_step(
  SimulatorType& simulator, value_type time_step,
  const cfl_estimate_helper_type& cfl_estimate_helper)
  -> std::vector<value_type>
{
  LOG_PREFIX("TempoIntegrator")

  // empty list of time steps
  std::vector<value_type> time_step_list;
  value_type time_remaining = time_step;
  value_type time_start = time;

  do {
    LOG_PREFIX("substep")

    // estimate CFL-number and optimize time step
    value_type max_velo_diam = cfl_estimate_helper(time);

    value_type time_substep = std::min(
      adjust_time_step_to_cfl_limit(time_step, max_velo_diam), time_remaining);

    felspa_log << "time substep = " << time_substep
               << " with CFL = " << time_substep * max_velo_diam << std::endl;

    // advance for the adjusted time scale
    TempoIntegrationSelector<TempoSchemeType, category>::integrate(
      simulator, time_substep, time);

    // register this time step
    time_step_list.push_back(time_substep);
    time_remaining -= time_substep;
    time += time_substep;
    ++substep_count;

    // make sure the simulator is all synchronized
    ASSERT(simulator.is_synchronized(), ExcSimulatorNotSync());

  } while (std::abs(time_remaining) > constants::DoubleTypeNearlyZero);

  felspa_log << "Time step " << time_step
             << " taken. Current simulation time = " << time << std::endl;

  // update full time step counter
  ++this->step_count;

  ASSERT(simulator.is_synchronized(), ExcSimulatorNotSync());
  ASSERT(numerics::is_zero(time_start + time_step - time), ExcInternalErr());

  return time_step_list;
}


template <typename NumberType>
FELSPA_FORCE_INLINE void TempoIntegrator<NumberType>::set_cfl(
  value_type cfl_min_, value_type cfl_max_) const
{
  ASSERT(cfl_min_ > 0.0, ExcArgumentCheckFail());
  ASSERT(cfl_max_ > 0.0, ExcArgumentCheckFail());
  ASSERT(cfl_min_ < cfl_max_, ExcArgumentCheckFail());

  ptr_control->cfl_max = cfl_max_;
  ptr_control->cfl_min = cfl_min_;
}


template <typename NumberType>
FELSPA_FORCE_INLINE auto
TempoIntegrator<NumberType>::adjust_time_step_to_cfl_limit(
  value_type time_step, value_type max_velo_diam) const -> value_type
{
  value_type cfl_min = ptr_control->cfl_min;
  value_type cfl_max = ptr_control->cfl_max;

  ASSERT(cfl_max > 0.0 && cfl_min > 0.0, ExcCflNotSet(cfl_min, cfl_max));
  ASSERT(time_step > 0.0, ExcArgumentCheckFail());
  ASSERT(max_velo_diam > 0.0, ExcArgumentCheckFail());

  value_type cfl_given = time_step * max_velo_diam;
  value_type time_step_adjusted = time_step;

  if (cfl_given < cfl_min)  // scale time step up
    time_step_adjusted = cfl_min / max_velo_diam;

  if (cfl_given > cfl_max)  // scale time step down
    time_step_adjusted = cfl_max / max_velo_diam;

  return time_step_adjusted;
}


/* -------------------------------------------------- */
/** \class RungeKuttaTVD */
/* -------------------------------------------------- */

template <typename SimulatorType>
void RungeKuttaTVD<1>::integrate_explicit(SimulatorType& simulator,
                                          const value_type time_step,
                                          const value_type current_time)
{
  ASSERT(current_time >= 0.0, ExcArgumentCheckFail());

  // get time derivative
  auto delta_phi = simulator.explicit_time_derivative(
    current_time, simulator.get_solution_vector());

  // now delta_phi is the updated soln
  delta_phi *= time_step;
  delta_phi += simulator.get_solution_vector();

  // update the solution and solution_time
  simulator.update_solution_and_sync(std::move(delta_phi),
                                     current_time + time_step);
}


template <typename SimulatorType>
void RungeKuttaTVD<2>::integrate_explicit(SimulatorType& simulator,
                                          const value_type time_step,
                                          const value_type current_time)
{
  ASSERT(current_time >= 0.0, ExcArgumentCheckFail());

  // make a copy of starting solution
  typename SimulatorType::vector_type soln_copy =
    simulator.get_solution_vector();

  // 1st predictor step
  auto delta_phi = simulator.explicit_time_derivative(current_time, soln_copy);
  soln_copy.add(time_step, delta_phi);

  // 2nd predictor step
  delta_phi =
    simulator.explicit_time_derivative(current_time + time_step, soln_copy);
  soln_copy.add(time_step, delta_phi);

  // averaging both predictor steps
  soln_copy += simulator.get_solution_vector();
  soln_copy *= 0.5;
  simulator.update_solution_and_sync(std::move(soln_copy),
                                     current_time + time_step);
}


template <typename SimulatorType>
void RungeKuttaTVD<3>::integrate_explicit(SimulatorType& simulator,
                                          const value_type time_step,
                                          const value_type current_time)
{
  ASSERT(current_time >= 0.0, ExcArgumentCheckFail());

  // --------------------------
  // make a copy of starting solution
  typename SimulatorType::vector_type soln_copy =
    simulator.get_solution_vector();
  // --------------------------

  // 1st predictor step
  auto phidot = simulator.explicit_time_derivative(current_time, soln_copy);
  soln_copy.add(time_step, phidot);

  // 2nd predictor step
  phidot =
    simulator.explicit_time_derivative(current_time + time_step, soln_copy);
  soln_copy.add(time_step, phidot);

  // 1st averaging step
  soln_copy.add(3.0, simulator.get_solution_vector());
  soln_copy *= 0.25;
  // now the soln_copy is at time + 0.5*delta_t
  // --------------------------

  // 3rd predictor step
  phidot = simulator.explicit_time_derivative(current_time + 0.5 * time_step,
                                              soln_copy);
  soln_copy.add(time_step, phidot);
  // now the soln_copy is at time + 1.5*delta_t

  // 2nd average step
  soln_copy *= 2.0;
  soln_copy += simulator.get_solution_vector();
  soln_copy *= 1.0 / 3.0;
  // now soln_copy is at time + delta_t

  // update solution and passive markers
  simulator.update_solution_and_sync(std::move(soln_copy),
                                     current_time + time_step);
  // --------------------------
}


FELSPA_NAMESPACE_CLOSE

#endif /* _FELSPA_PDE_TIME_INTEGRATION_IMPLEMENT_H_ */
