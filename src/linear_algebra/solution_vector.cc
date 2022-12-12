#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>
#include <felspa/linear_algebra/solution_vector.h>

FELSPA_NAMESPACE_OPEN

template class TimedSolutionVector<dealii::Vector<types::DoubleType>>;

FELSPA_NAMESPACE_CLOSE
