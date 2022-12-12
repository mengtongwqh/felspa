#include  <felspa/base/quadrature.h>

FELSPA_NAMESPACE_OPEN

/* -------------------------------------------------- */
/*                 IMPLEMENTATIONS                    */
/* -------------------------------------------------- */
template <>
QEvenlySpaced<0>::QEvenlySpaced(const unsigned int n, bool randomize)
  : dealii::Quadrature<0>(1), randomize_flag(randomize), n_per_dim(n)
{
  this->weights[0] = 1;
}


template <>
QEvenlySpaced<1>::QEvenlySpaced(const unsigned int n, bool randomize)
  : Quadrature<1>(n), randomize_flag(randomize), n_per_dim(n)
{
  double dx = 1.0 / n;
  for (unsigned int i = 0; i < n; ++i) {
    this->quadrature_points[i](0) = (i + 0.5) * dx;
    this->weights[i] = 1.0 / n;
  }

  if (randomize) randomize_dimension(0, dx);
}


template <int dim>
QEvenlySpaced<dim>::QEvenlySpaced(const unsigned int n, bool randomize)
  : dealii::Quadrature<dim>(QEvenlySpaced<dim - 1>(n, false),
                            QEvenlySpaced<1>(n, false)),
    randomize_flag(randomize)
{
  if (randomize) {
    double dx = 1.0 / n;
    for (int idim = 0; idim < dim; ++idim) randomize_dimension(idim, dx);
  }
}


template <int dim>
QEvenlySpaced<dim>::QEvenlySpaced(const QEvenlySpaced<dim>& that)
  : QEvenlySpaced(that.n_per_dim, false)
{
  if (that.is_randomized()) {
    double dx = 1.0 / that.n_per_dim;
    randomize_flag = true;
    for (unsigned int i = 0; i < dim; ++i) randomize_dimension(i, dx);
  }
}


template <int dim>
FELSPA_FORCE_INLINE bool QEvenlySpaced<dim>::is_randomized() const
{
  return randomize_flag;
}


template <int dim>
void QEvenlySpaced<dim>::randomize_dimension(int d, double dx)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-0.5, 0.5);

  for (auto& pt : this->quadrature_points)
    // perturbation in [-0.5, 0.5]
    pt[d] += dis(gen) * dx;
}


// --- Explicit instantiation --- //
template class QEvenlySpaced<1>;
template class QEvenlySpaced<2>;
template class QEvenlySpaced<3>;
// ----------------------------- //

FELSPA_NAMESPACE_CLOSE