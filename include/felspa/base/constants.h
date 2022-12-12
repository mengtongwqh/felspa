#ifndef _FELSPA_BASE_CONSTANTS_H_
#define _FELSPA_BASE_CONSTANTS_H_

#include <deal.II/base/types.h>
#include <felspa/base/felspa_config.h>

#include <cmath>

FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/**
 * \namespace constants
 * \brief Some numerical / physical constants
 */
/* ************************************************** */

/* ------------------------------*/
namespace constants
/* ------------------------------*/
{
  /**
   * Threshold for nearly zero floating point number
   */
  constexpr double DoubleTypeNearlyZero = 1.0e-12;


  /**
   * Constant \f$ \pi \approx 3.1415926 \ldots \f$
   */
  constexpr double PI = dealii::numbers::PI;


  /**
   * Gravity constant of the earth
   */
  constexpr double earth_gravity = -9.8;


  /**
   * Seconds in a year
   */
  constexpr double seconds_per_year = 60 * 60 * 24 * 365.2425;

  /**
   * use the largest unsigned int value to represent an invalid return value
   */
  constexpr unsigned int invalid_unsigned_int = static_cast<unsigned int>(-1);


  /**
   * Maximum count of digits the output file counter can have.
   * If this number is 3, then we can output at most 999 time steps.
   */
  constexpr unsigned int max_export_numeric_digits = 9;


  /**
   * Alias for the current directory.
   */
  const std::string current_dir = "./";
}  // namespace constants

FELSPA_NAMESPACE_CLOSE
#endif  // _FELSPA_BASE_CONSTANTS_H_ //
