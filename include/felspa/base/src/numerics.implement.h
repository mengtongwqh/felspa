#ifndef _FELSPA_BASE_NUMERICS_IMPLEMENT_H_
#define _FELSPA_BASE_NUMERICS_IMPLEMENT_H_
#include <felspa/base/numerics.h>


FELSPA_NAMESPACE_OPEN
namespace numerics
{
  /* ************************************************** */
  template <typename NumberType>
  bool is_zero(NumberType number, unsigned int scaling)
  {
    if constexpr (std::is_unsigned<NumberType>::value)
      return number <= scaling * std::numeric_limits<NumberType>::epsilon();
    else
      return std::abs(number) <=
             scaling * std::numeric_limits<NumberType>::epsilon();
  }


  /* ************************************************** */
  template <typename NumberType,
            std::enable_if_t<std::is_floating_point<NumberType>::value, void*>>
  bool is_nearly_equal(const NumberType& a, const NumberType& b)
  {
    using namespace ::FELSPA_NAMESPACE::constants;
    if (a < 1.0 && b < 1.0)
      return std::abs(a - b) < DoubleTypeNearlyZero;
    else
      return std::abs(a - b) <
             DoubleTypeNearlyZero * std::max(std::abs(a), std::abs(b));
  }


  /* ************************************************** */
  template <int dim, typename NumberType>
  NumberType flat_matrix_determinant(const NumberType* m)
  {
    static_assert(0 < dim && dim <= 3, "Spatial dimension must be in [1,3].");
    if constexpr (dim == 1)
      return m[0];
    else if (dim == 2)
      return m[0] * m[3] - m[1] * m[2];
    else if (dim == 3)
      return m[0] * (m[4] * m[8] - m[5] * m[7]) -
             m[1] * (m[3] * m[8] - m[5] * m[6]) +
             m[2] * (m[3] * m[7] - m[4] * m[6]);
    else
      THROW(ExcInternalErr());
  }

  /* ************************************************** */
  template <typename NumberType>
  std::array<NumberType, 3> flat_matrix_eigval3_symm(const NumberType* m)
  {
    using constants::PI;
    using std::atan;
    using std::cos;
    using std::sqrt;

    NumberType a = m[0], b = m[4], c = m[8], d = m[3], e = m[7], f = m[6];

    NumberType abc = a + b + c;

    NumberType ra = 2.0 * a - b - c;
    NumberType rb = 2.0 * b - a - c;
    NumberType rc = 2.0 * c - a - b;

    NumberType x1 = a * a + b * b + c * c - a * b - a * c - b * c +
                    3 * (d * d + f * f + e * e);
    NumberType x2 = -ra * rb * rc +
                    9.0 * (rc * d * d + rb * f * f + ra * e * e) -
                    54. * (d * e * f);

    NumberType tmp = sqrt(4.0 * x1 * x1 * x1 - x2 * x2);

    NumberType phi(0.0);
    if (x2 > 0)
      phi = atan(tmp / x2);
    else if (x2 < 0)
      phi = atan(tmp / x2) + PI;
    else
      phi = 0.5 * PI;

    std::array<NumberType, 3> eigvals;
    eigvals[0] = (abc - 2.0 * sqrt(x1) * cos(phi / 3.0)) / 3.0;
    eigvals[1] = (abc + 2.0 * sqrt(x1) * cos((phi - PI) / 3.0)) / 3.0;
    eigvals[2] = (abc + 2.0 * sqrt(x1) * cos((phi + PI) / 3.0)) / 3.0;

    return std::sort(eigvals.begin(), eigvals.end());
  }


  /* ************************************************** */
  template <typename NumberType>
  NumberType relative_stretching(const NumberType* F,
                                const bool compute_flinn_slope)
  {
    using std::sqrt;
    constexpr int dim = 3;
    auto idx3 = [](int i, int j) { return i * 3 + j; };

    // left Cauchy-Green tensor
    dealii::SymmetricTensor<2, dim, NumberType> FFt;

    for (int i = 0; i < dim; ++i)
      for (int j = i; j < dim; ++j)
        for (int k = 0; k < dim; ++k)
          FFt[i][j] += F[idx3(i, k)] * F[idx3(j, k)];

    std::array<NumberType, dim> eigvals = eigenvalues(FFt);
    ASSERT(eigvals[dim - 1] > 0.0,
           ExcUnexpectedValue<NumberType>(eigvals[dim - 1]));

    if (compute_flinn_slope)
      return sqrt(eigvals[0] * eigvals[2] / eigvals[1] * eigvals[1]);
    else
      return sqrt(eigvals[0] / eigvals[2]);
  }

}  // namespace numerics

FELSPA_NAMESPACE_CLOSE
#endif  // _FELSPA_BASE_NUMERICS_IMPLEMENT_H_ //