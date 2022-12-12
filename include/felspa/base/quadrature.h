
#ifndef _FELSPA_BASE_QUARATURE_H_
#define _FELSPA_BASE_QUARATURE_H_

#include <deal.II/base/quadrature.h>
#include <felspa/base/felspa_config.h>

#include <random>

FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/**
 * generate random particle distribution in the reference cell
 */
/* ************************************************** */
template <int dim>
class QEvenlySpaced : public dealii::Quadrature<dim>
{
 public:
  constexpr static int dimension = dim;

  explicit QEvenlySpaced(const unsigned int n = 3, bool randomize = true);
  

  /**
   * @brief Construct a new QEvenlySpaced object
   * 
   * @param that 
   */
  QEvenlySpaced(const QEvenlySpaced<dim>& that);

  QEvenlySpaced<dim>& operator=(const QEvenlySpaced<dim>& that);

  bool is_randomized() const;


 private:
  void randomize_dimension(int d, double dx);

  bool randomize_flag;

  unsigned int n_per_dim;
};


FELSPA_NAMESPACE_CLOSE

#endif  // _FELSPA_BASE_QUARATURE_H_ //
