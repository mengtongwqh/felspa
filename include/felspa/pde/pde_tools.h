#ifndef _FELSPA_PDE_PDE_TOOLS_H_
#define _FELSPA_PDE_PDE_TOOLS_H_

#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <felspa/base/felspa_config.h>
#include <felspa/base/function.h>
#include <felspa/fe/cell_data.h>
#include <felspa/pde/pde_base.h>

#include <algorithm>
#include <typeinfo>

FELSPA_NAMESPACE_OPEN

enum class Norm
{
  L1,
  L2,
  Linf,
};


/* ************************************************** */
/**
 * @brief MeshGeometryInfo
 */
/* ************************************************** */
template <int dim>
struct MeshGeometryInfo;

template <>
struct MeshGeometryInfo<1>
{
  constexpr static unsigned int faces_around_vertex[][1] = {{0}, {1}};

  constexpr static unsigned int faces_normal_to_axis[1][2] = {{0, 1}};
};

template <>
struct MeshGeometryInfo<2>
{
  constexpr static unsigned int faces_around_vertex[][2] = {
    {0, 2}, {2, 1}, {0, 3}, {1, 3}};

  constexpr static unsigned int faces_normal_to_axis[2][2] = {{0, 1}, {2, 3}};
};

template <>
struct MeshGeometryInfo<3>
{
  constexpr static unsigned int faces_around_vertex[][3] = {
    {0, 2, 4}, {1, 2, 4}, {0, 3, 4}, {1, 3, 4},
    {0, 2, 5}, {2, 5, 6}, {0, 3, 5}, {1, 3, 5}};

  constexpr static unsigned int faces_normal_to_axis[3][2] = {
    {0, 1}, {2, 3}, {4, 5}};
};


/* ************************************************** */
/**
 * The abstract base class for velocity extractors.
 */
/* ************************************************** */
template <typename VeloFcn>
class VelocityExtractorBase
{
 public:
  constexpr static int dimension = VeloFcn::dimension, dim = dimension;
  using simulator_type = VeloFcn;
  using value_type = typename VeloFcn::value_type;
  using tensor_type = dealii::Tensor<1, dim, value_type>;

  /**
   * Virtual function. Extract the velocities for each cell
   * and put them in an velocities vector.
   */
  virtual void extract(const simulator_type& simulator,
                       const dealii::FEValuesBase<dim>& feval,
                       std::vector<tensor_type>& velocities) const
  {
    UNUSED_VARIABLE(simulator);
    UNUSED_VARIABLE(feval);
    UNUSED_VARIABLE(velocities);
    THROW(ExcUnimplementedVirtualFcn());
  }


  /**
   * Default implementation for extracting velocity magnitude.
   * Take norm on each velocity vector obtained from \c extract().
   */
  void extract_magnitude(const simulator_type& sim,
                         const dealii::FEValuesBase<dim>& feval,
                         std::vector<value_type>& velocity_norm)
  {
    std::vector<dealii::Tensor<1, dim, value_type>> velocity(
      feval.n_quadrature_points);
    extract(sim, feval, velocity);
    std::transform(velocity.cbegin(), velocity.cend(), velocity_norm.begin(),
                   [](const tensor_type& velo) { return velo.norm(); });
  }
};


/* ************************************************** */
/**
 * Specialize \c VelocityExtrator for existing simulators.
 * We do this by specializing the \c extract() function.
 */
/* ************************************************** */
template <typename T,
          bool = std::is_base_of<
            TensorFunction<1, T::dimension, typename T::value_type>, T>::value>
class VelocityExtractor; /**< Generic declaration */


template <typename VeloFcn>
class VelocityExtractor<VeloFcn, true> : public VelocityExtractorBase<VeloFcn>
{
 public:
  constexpr static int dimension = VeloFcn::dimension, dim = dimension;
  using value_type = typename VeloFcn::value_type;
  using simulator_type = VeloFcn;
  using tensor_type = dealii::Tensor<1, dim, value_type>;


  /**
   * Fill the velocities vector.
   * The \c extract_magnitude() function will use the default definition
   * in \c VelocityExtractorBase.
   */
  void extract(const simulator_type& tensor_function,
               const dealii::FEValuesBase<dim>& feval,
               std::vector<tensor_type>& velocities) const override
  {
    ASSERT(feval.n_quadrature_points == velocities.size(),
           ExcSizeMismatch(feval.n_quadrature_points, velocities.size()));

    const auto& qpts = feval.get_quadrature_points();
    std::transform(
      qpts.cbegin(), qpts.cend(), velocities.begin(),
      [&tensor_function](const dealii::Point<dim, value_type>& pt) {
        return tensor_function(pt);
      });
  }
};


/* ************************************************** */
/**
 *  For each cell described by a given FEValues,
 *  compute the velocities at the quadrature points
 *  and put them in the velocities vector.
 */
/* ************************************************** */
template <typename SimulatorType>
void extract_cell_velocities(
  const SimulatorType& sim,
  const dealii::FEValuesBase<SimulatorType::dimension>& feval,
  std::vector<dealii::Tensor<1, SimulatorType::dimension,
                             typename SimulatorType::value_type>>& velocities)
{
  // call the struct with implementations
  VelocityExtractor<SimulatorType>().extract(sim, feval, velocities);
}


template <typename SimulatorType>
void extract_cell_velocity_magnitude(
  const SimulatorType& sim,
  const dealii::FEValuesBase<SimulatorType::dimension>& feval,
  std::vector<typename SimulatorType::value_type>& velocity_norms)
{
  VelocityExtractor<SimulatorType>().extract_magnitude(sim, feval,
                                                       velocity_norms);
#ifdef DEBUG
  for (auto& norm : velocity_norms)
    ASSERT(norm >= 0.0, ExcUnexpectedValue(norm));
#endif  // DEBUG //
}


/* ************************************************** */
/**
 * Compute the velocity norm
 */
/* ************************************************** */
template <typename SimulatorType>
auto compute_velocity_norm(
  const SimulatorType& sim,
  const dealii::Quadrature<SimulatorType::dimension>& quad,
  Norm norm_type = Norm::L2) -> typename SimulatorType::value_type;


/* ************************************************** */
/**
 * CFL Estimator Base class. This class does not
 * require a SimulatorType as a template parameter
 * and therefore serves as a simulator-independent
 * public interface for CFL estimation.
 */
/* ************************************************** */
template <int dim, typename NumberType>
class CFLEstimatorBase
{
  constexpr static int dimension = dim;

 public:
  using cell_iterator_type =
    typename dealii::DoFHandler<dim>::active_cell_iterator;
  using value_type = NumberType;

  /**
   * Virtual destructor
   */
  virtual ~CFLEstimatorBase() = default;

  virtual value_type estimate(const cell_iterator_type& begin,
                              const cell_iterator_type& end)
  {
    UNUSED_VARIABLE(begin);
    UNUSED_VARIABLE(end);
    THROW(ExcUnimplementedVirtualFcn());
  }

  virtual value_type estimate(const cell_iterator_type& begin,
                              const cell_iterator_type& end,
                              const dealii::Mapping<dim>& mapping,
                              const dealii::FiniteElement<dim>& fe)
  {
    UNUSED_VARIABLE(begin);
    UNUSED_VARIABLE(end);
    UNUSED_VARIABLE(mapping);
    UNUSED_VARIABLE(fe);
    THROW(ExcUnimplementedVirtualFcn());
  }

  /**
   * If we have a simulator-derived type, then \c true.
   */
  const bool is_simulator;

 protected:
  /**
   * External constrution prohibited.
   */
  CFLEstimatorBase(bool is_simulator_) : is_simulator(is_simulator_) {}
};


/* ************************************************** */
/**
 * CFL Estimator specialized for a simulator.
 */
/* ************************************************** */
template <typename SimulatorType>
class CFLEstimator : public CFLEstimatorBase<SimulatorType::dimension,
                                             typename SimulatorType::value_type>
{
 public:
  constexpr static int dimension = SimulatorType::dimension, dim = dimension;
  using base_type = CFLEstimatorBase<SimulatorType::dimension,
                                     typename SimulatorType::value_type>;
  using simulator_type = SimulatorType;
  using value_type = typename SimulatorType::value_type;
  using tensor_type = dealii::Tensor<1, dim, value_type>;
  using cell_iterator_type =
    typename dealii::DoFHandler<dim>::active_cell_iterator;

  /** Scratch Data */
  struct ScratchData;

  /** Copy data */
  struct CopyData;

  CFLEstimator(const SimulatorType& sim)
    : base_type(is_simulator<SimulatorType>::value), ptr_simulator(&sim)
  {
    static_assert(
      is_simulator<SimulatorType>::value ||
        std::is_base_of<TensorFunction<1, SimulatorType::dimension,
                                       typename SimulatorType::value_type>,
                        SimulatorType>::value,
      "Only valid for a simulator or a tensor function");
  }


  /**
   * Run estimation for a simulator-derived class.
   * Use the default mapping and finite element in the simulator.
   */
  value_type estimate(const cell_iterator_type& begin,
                      const cell_iterator_type& end) override
  {
    if constexpr (is_simulator<SimulatorType>::value)
      return this->do_estimate(begin, end, ptr_simulator->get_mapping(),
                               ptr_simulator->get_fe());
    else
      THROW(EXCEPT_MSG("Cannot be used on non-simulator type."));
  }


  /**
   * Run estimation with \c FiniteElment and \c Mapping provided
   * for \c TensorFunction. Also will do a type checking.
   */
  value_type estimate(const cell_iterator_type& begin,
                      const cell_iterator_type& end,
                      const dealii::Mapping<dim>& mapping,
                      const dealii::FiniteElement<dim>& fe) override
  {
    bool is_tensor_function =
      std::is_base_of<TensorFunction<1, dim, value_type>, SimulatorType>::value;
    ASSERT(
      is_tensor_function,
      EXCEPT_MSG("Can only be used on types derived from TensorFunction."));
    return this->do_estimate(begin, end, mapping, fe);
  }


  /**
   * Run estimation with \c FiniteElment and \c Mapping provided.
   */
  value_type do_estimate(const cell_iterator_type& begin,
                         const cell_iterator_type& end,
                         const dealii::Mapping<dim>& mapping,
                         const dealii::FiniteElement<dim>& fe);


  /**
   * Estimate the CFL on quadrature points.
   */
  void local_estimate(const cell_iterator_type& cell,
                      ScratchData& s,
                      CopyData& c);


  /**
   * Compare the local CFL estimate to global CFL estimate.
   */
  void compare_local_to_global(const CopyData& copy);


 protected:
  /**
   * The result of the comparison.
   */
  value_type velo_diam = 0.0;


  /**
   * Pointer to simulator.
   */
  const dealii::SmartPointer<const SimulatorType, CFLEstimator<SimulatorType>>
    ptr_simulator;
};


namespace dg
{

  /* ************************************************** */
  /**
   * @brief minmod function
   */
  /* ************************************************** */
  template <typename NumberType>
  constexpr NumberType minmod(NumberType arg1)
  {
    // base case: do nothing
    return arg1;
  }

  template <typename NumberType, typename... RestType>
  constexpr NumberType minmod(NumberType arg1, RestType... args)
  {
    using util::sign;
    NumberType arg_rest = minmod(args...);
    if (sign(arg1) == sign(arg_rest))
      return static_cast<NumberType>(sign(arg1)) *
             std::min(std::abs(arg1), std::abs(arg_rest));
    else
      return 0.0;
  }

  template <typename NumberType>
  NumberType minmod(const std::vector<NumberType>& args)
  {
    ASSERT_NON_EMPTY(args);
    using ::FELSPA_NAMESPACE::util::sign;

    int s = sign(args[0]);
    NumberType value = std::abs(args[0]);

    auto it = ++args.begin();
    for (; it != args.end(); ++it) {
      if (s == sign(*it)) {
        s *= sign(*it);
        value = std::min(std::abs(value), std::abs(*it));
      } else {
        return 0.0;
      }
    }
    return s * value;
  }

  /* ************************************************** */
  /**
   * @brief Base class for shock cell identifier
   */
  /* ************************************************** */
  template <int dim, typename NumberType>
  class ShockDetectorBase
  {
   public:
    using ActiveCellIterator =
      typename dealii::DoFHandler<dim>::active_cell_iterator;
    using CellIterator =
      dealii::TriaIterator<dealii::DoFCellAccessor<dim, dim, false>>;


    /**
     * @brief Per task data
     */
    struct ScratchData
    {
      /**
       * @brief Constructor
       */
      ScratchData(const dealii::Mapping<dim, dim>& mapping,
                  const dealii::FiniteElement<dim, dim>& fe,
                  const dealii::Quadrature<dim>& quad);

      /**
       * @brief Copy constructor
       */
      ScratchData(const ScratchData&);

      /**
       * @brief FEValues of the current cell
       */
      dealii::FEValues<dim> feval_cell;

      /**
       * @brief FEvalues to allow solution extraction at vertices
       */
      dealii::FEValues<dim> feval_vertex;

      /**
       * @brief FEValues of the neighboring cell
       */
      dealii::FEValues<dim> feval_neighbor;
    };

    /**
     * @brief Empty copy data since there is no need to
     * assemble global matrix/vector
     */
    struct CopyData
    {};


    /**
     * @brief Constructor
     */
    ShockDetectorBase(Mesh<dim, NumberType>& mesh);

    /**
     * @brief Cycle through all cells and detect the cells with shock.
     * Use the Workstream framework
     */
    const std::vector<bool>& detect_shock_cells(
      dealii::IteratorRange<ActiveCellIterator> cell_range,
      const dealii::Mapping<dim, dim>& mapping,
      const dealii::FiniteElement<dim, dim>& fe,
      const dealii::Quadrature<dim>& quad);


    /**
     * @return true if the cell has shock
     * @return false if the solution in the cell is smooth
     */
    virtual bool cell_has_shock(const ActiveCellIterator& cell_iter,
                                ScratchData& scratch_data) const = 0;


    /**
     * @brief export the mesh to file
     * Assuming the material id field is not used
     */
    void export_mesh(ExportFile& file) const;


   protected:
    const dealii::SmartPointer<Mesh<dim, NumberType>,
                               ShockDetectorBase<dim, NumberType>>
      ptr_mesh;

    std::vector<bool> shock_cell_flags;
  };


  /* ************************************************** */
  /**
   * @brief Implementation of the minmod-type shock cell identifier
   * for different spatial dimension.
   * @tparam dim
   * @tparam NumberType
   * @todo correct for fine-coarse cell interface
   */
  /* ************************************************** */
  template <int dim, typename VectorType>
  class MinmodShockDetector
    : public ShockDetectorBase<dim, typename VectorType::value_type>
  {
   public:
    using NumberType = typename VectorType::value_type;
    using value_type = typename VectorType::value_type;
    using base_type = ShockDetectorBase<dim, NumberType>;
    using typename base_type::ActiveCellIterator;
    using typename base_type::CellIterator;
    using typename base_type::ScratchData;

    /**
     * @brief Constructor
     */
    MinmodShockDetector(Mesh<dim, NumberType>& mesh, const VectorType& soln,
                        NumberType beta_coeff = 1.0 + dim);

    /**
     * @brief the actual implementation of the troubled cell detection
     */
    bool cell_has_shock(const ActiveCellIterator& cell_iter,
                        ScratchData& scratch_data) const override;


   protected:
    const VectorType* const ptr_solution_vector;

    /**
     * @brief ceofficient to be multiplied onto the minmod function
     */
    NumberType beta_coeff;
  };


  /* ************************************************** */
  /**
   * @brief Implementation of the curvature-based shock cell detector
   * @tparam dim
   * @tparam NumberType
   */
  /* ************************************************** */
  template <int dim, typename NumberType>
  class CurvatureShockDetector : public ShockDetectorBase<dim, NumberType>
  {
   public:
    using base_type = ShockDetectorBase<dim, NumberType>;
    using typename base_type::ActiveCellIterator;
    using typename base_type::ScratchData;

    CurvatureShockDetector(Mesh<dim, NumberType>& mesh,
                           const dealii::Vector<NumberType>& initial_solution,
                           const dealii::BlockVector<NumberType>& left_grads,
                           const dealii::BlockVector<NumberType>& right_grads);

    bool cell_has_shock(const ActiveCellIterator& cell_iter,
                        ScratchData& scratch_data) const override;

   protected:
    const dealii::Vector<NumberType>* const ptr_initial_solution_vector;

    const dealii::BlockVector<NumberType>* const ptr_left_gradients;

    const dealii::BlockVector<NumberType>* const ptr_right_gradients;
  };


  /* ************************************************** */
  /**
   * @brief Apply WENO limiting to solution
   */
  /* ************************************************** */
  template <int dim, typename VectorType>
  class WENOLimiter
  {
   public:
    using this_type = WENOLimiter<dim, VectorType>;
    using NumberType = typename VectorType::value_type;
    using ActiveCellIterator =
      typename dealii::DoFHandler<dim>::active_cell_iterator;
    using CellIterator =
      dealii::TriaIterator<dealii::DoFCellAccessor<dim, dim, false>>;

    struct ScratchData
    {
      /**
       * @brief Constructor
       */
      ScratchData(const dealii::Mapping<dim>& mapping,
                  const dealii::FiniteElement<dim>& fe,
                  const dealii::Quadrature<dim>& quad);

      ScratchData(const ScratchData&);

      dealii::FEValues<dim> feval_home;

      dealii::FEValues<dim> feval_neighbor;
    };

    struct CopyData
    {
      bool assemble = true;

      std::vector<dealii::types::global_dof_index> local_dof_indices;

      std::vector<NumberType> local_soln;
    };

    /**
     * @brief Construct a new WENOLimiter object
     */
    WENOLimiter(VectorType& soln_vector,
                unsigned int max_derivative = 1,
                NumberType neighbor_cell_gamma = 0.1);

    /**
     * @brief Apply limiting to all cells in IteratorRange.
     *
     * @tparam Iterator
     * @param cells
     */
    template <typename Iterator>
    void apply_limiting(const dealii::IteratorRange<Iterator>& cells,
                        const dealii::Mapping<dim>& mapping,
                        const dealii::FiniteElement<dim>& fe,
                        const dealii::Quadrature<dim>& quad);

   protected:
    /**
     * @brief Find the set of neighoring cells
     */
    static std::vector<ActiveCellIterator> find_neighbor_cells(
      const ActiveCellIterator& cell);

    template <typename CellIteratorType>
    void apply_limiting_to_cell(const CellIteratorType& home_cell,
                                ScratchData& scratch_data,
                                CopyData& copy_data);

    void copy_local_to_global(const CopyData& copy_data);


    /**
     * @brief obtain the smoothness indicator the for this cell
     * @param n_deriv
     * @return NumberType
     */
    NumberType compute_smoothness_indicator(
      const dealii::FEValues<dim>& feval_home,
      const dealii::FEValues<dim>& feval_neighbor,
      unsigned int n_deriv = 1) const;

    /**
     * @brief  Copy of the solution vector since we are overwriting the solution
     * vector
     */
    const VectorType old_soln_vector;

    /**
     * @brief Pointer to the solution vector
     */
    const dealii::SmartPointer<VectorType, this_type> ptr_soln_vector;

    /**
     * @brief the cell weighting of the neighboring cell
     */
    NumberType neighbor_cell_gamma;

    NumberType epsilon = 1.0e-6;

    unsigned int max_smoothness_indicator_derivative;
  };


  /* ************************************************** */
  /**
   * @brief Generalized minmod type limiter applied to QLegendre basis
   * The limiter is applied to the expansion coefficient
   * of the basis functions in a recursive manner.
   * Reference:
   * Lilia Krivodonova, 2007, Journal of Computational Physics,
   * Limiters for high-order discontinuous Galerkin methods
   * https://doi.org/10.1016/j.jcp.2007.05.011
   */
  /* ************************************************** */
  template <int dim, typename VectorType>
  class MomentLimiter
  {
   public:
    using this_type = MomentLimiter<dim, VectorType>;
    using NumberType = typename VectorType ::value_type;
    using ActiveCellIterator =
      typename dealii::DoFHandler<dim>::active_cell_iterator;
    using CellIterator =
      dealii::TriaIterator<dealii::DoFCellAccessor<dim, dim, false>>;
    using ScratchData = typename WENOLimiter<dim, VectorType>::ScratchData;

    struct CopyData
    {
      std::vector<dealii::types::global_dof_index> local_dof_indices;

      dealii::Vector<NumberType> local_soln;
    };

    /**
     * @brief Constructor
     */
    MomentLimiter(VectorType& soln_vector, unsigned int degree);


    template <typename Iterator>
    void apply_limiting(const dealii::IteratorRange<Iterator>& cells,
                        const dealii::Mapping<dim>& mapping,
                        const dealii::FiniteElement<dim>& fe,
                        const dealii::Quadrature<dim>& quad);

   protected:
    /**
     * @brief Function to apply limiting to each cell
     */
    void apply_limiting_to_cell(const ActiveCellIterator& home_cell,
                                ScratchData& scratch_data,
                                CopyData& copy_data);

    void copy_local_to_global(const CopyData& copy_data) const;

    /**
     * convert lexigraphic (i,j) in 2d or (i,j,k) in 3d
     * to linear indexing
     */
    unsigned int idx(const std::array<int, dim>& nd_idx) const;

    void limit_coefficients(
      dealii::Vector<NumberType>& home_coeffs,
      const std::vector<dealii::Vector<NumberType>>& backward_coeffs,
      const std::vector<dealii::Vector<NumberType>>& forward_coeffs) const;

    void limit_coefficients_2d(
      dealii::Vector<NumberType>& home_coeffs,
      const std::vector<dealii::Vector<NumberType>>& backward_coeffs,
      const std::vector<dealii::Vector<NumberType>>& forward_coeffs) const;

    const VectorType old_soln_vector;

    const dealii::SmartPointer<VectorType, this_type> ptr_soln_vector;

    const unsigned int fe_degree;
  };


  /* ************************************************** */
  /**
   * @brief Helper function to compute cell average
   */
  /* ************************************************** */
  template <int dim, typename VectorType>
  auto compute_cell_average(const dealii::FEValues<dim>& feval,
                            const VectorType& soln_vector) ->
    typename VectorType::value_type;


  /* ************************************************** */
  /**
   * First label the troubled cell and
   * then apply WENO limiter to these cells
   */
  /* ************************************************** */
  template <int dim, typename VectorType>
  void apply_weno_limiter(Mesh<dim>& mesh,
                          const dealii::DoFHandler<dim>& dofh,
                          const dealii::Mapping<dim>& mapping,
                          const dealii::Quadrature<dim>& quadrature,
                          VectorType& soln_vector);


  /* ************************************************** */
  /**
   * apply moment limiter to all cells
   */
  /* ************************************************** */
  template <int dim, typename VectorType>
  void apply_moment_limiter(const dealii::DoFHandler<dim>& dofh,
                            const dealii::Mapping<dim>& mapping,
                            const dealii::Quadrature<dim>& quadrature,
                            VectorType& soln_vector);

}  // namespace dg


FELSPA_NAMESPACE_CLOSE
// -------- Implementaions -------- //
#include "src/pde_tools.implement.h"
// -------------------------------- //
#endif  // _FELSPA_PDE_PDE_TOOLS_H_ //
