#ifndef _FELSPA_BASE_NUMERICS_H_
#define _FELSPA_BASE_NUMERICS_H_


#include <felspa/base/constants.h>
#include <felspa/base/exceptions.h>
#include <felspa/base/felspa_config.h>

#include <cmath>

FELSPA_NAMESPACE_OPEN

/* ------------------------------ */
namespace numerics
/* ------------------------------ */
{
  /* ************************************************** */
  /**
   * First construct an \c IsZero struct/
   * The condition is tested in the constructor of \c IsZero struct
   * The result of the test is stored in \c IsZero::value
   * so return the value of this member
   */
  /* ************************************************** */
  template <typename NumberType>
  bool is_zero(NumberType number, unsigned int scaling = 1);


  /* ************************************************** */
  /**
   * Strict equality for arithmetic types.
   */
  /* ************************************************** */
  template <typename NumberType>
  bool is_equal(const NumberType& a, const NumberType& b)
  {
    static_assert(std::is_integral_v<NumberType> ||
                  std::is_floating_point_v<NumberType>);

    if constexpr (std::is_integral_v<NumberType>)
      return a == b;
    else {
      // if (std::abs(a) < 1.0 && std::abs(b) < 1.0)
      // only allow for machine epsilon
      return std::abs(a - b) < std::numeric_limits<NumberType>::epsilon() *
                                 std::numeric_limits<NumberType>::round_error();
    }
  }

  /* ************************************************** */
  /**
   * Less strict equality comparison or floating point data types.
   * This is not the strictest test for equality and
   * can be used on data where floating-point error may
   * accumulate, e.g. cumulated physical time.
   */
  /* ************************************************** */
  template <typename NumberType,
            std::enable_if_t<std::is_floating_point<NumberType>::value, void*> =
              nullptr>
  bool is_nearly_equal(const NumberType& a, const NumberType& b);


  /* ************************************************** */
  /**
   * @brief Take the dim-th root of the input n
   */
  /* ************************************************** */
  template <
    int dim,
    typename NumberType,
    std::enable_if_t<std::is_integral<NumberType>::value, void*> = nullptr>
  NumberType dimth_root(NumberType n)
  {
    if constexpr (dim == 1)
      return n;
    else
      return static_cast<NumberType>(std::round(std::pow(n, 1.0 / dim)));
  }


  /* ************************************************** */
  /**
   * Compute the determinant of a matrix
   * that is flattened in a row-major order
   */
  /* ************************************************** */
  template <int dim, typename NumberType>
  NumberType flat_matrix_determinant(const NumberType* m);


  /* ************************************************** */
  /**
   * A specialization for eigenvalue/eigenvector computation
   * for 3x3 symmetric matrix
   * Reference:
   * Charles-Alban Deledalle, Loic Denis, Sonia Tabti, Florence Tupin.
   * Closed-form expressions of the eigen decomposition of 2 x 2
   * and 3 x 3 Hermitian matrices. [Research Report]
   * Université de Lyon. 2017.
   * https://hal.archives-ouvertes.fr/hal-01501221/document
   */
  /* ************************************************** */
  template <typename NumberType>
  std::array<NumberType, 3> flat_matrix_eigval3_symm(const NumberType* m);


  /* ************************************************** */
  /**
   * compute the Flinn slope with a flattened
   * deformation gradient F.
   */
  /* ************************************************** */
  template <typename NumberType>
  NumberType relative_stretching(const NumberType* F, bool compute_flinn_slope = false);

}  // namespace numerics
/* ------------------------------ */

FELSPA_NAMESPACE_CLOSE

#include "src/numerics.implement.h"
#endif  // _FELSPA_BASE_NUMERICS_H_ //
