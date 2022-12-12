#ifndef _FELSPA_LEVEL_SET_LEVEL_SET_H_
#define _FELSPA_LEVEL_SET_LEVEL_SET_H_

#include <deal.II/base/iterator_range.h>
#include <deal.II/base/point.h>
#include <felspa/base/exceptions.h>
#include <felspa/base/felspa_config.h>
#include <felspa/level_set/geometry.h>
#include <felspa/level_set/reinit.h>
#include <felspa/pde/advection.h>


FELSPA_NAMESPACE_OPEN

/* -------------------------------------------*/
namespace ls
/* -------------------------------------------*/
{
  /* ************************************************** */
  /**
   * \defgroup LevelSet Level Set
   */
  /* ************************************************** */

  /* ************************************************** */
  /**
   * Enum class of all smoothing method available for computing
   * (mollified/smoothed) Heaviside function.
   */
  /* ************************************************** */
  enum class HeavisideSmoothing
  {
    none,
    sine,
    linear,
    poly3,
    poly5,
  };


  /* ************************************************** */
  /**
   * Abstract base class describing the fundamental
   * functionalities of a level set class.
   * \todo consider merging this class into FESimulatorBase
   */
  /* ************************************************** */
  template <int dim, typename NumberType>
  class LevelSetBase
  {
   public:
    using value_type = NumberType;

    /**
     * Get the level set values at quadrature points.
     */
    virtual void extract_level_set_values(
      const dealii::FEValuesBase<dim>& fe,
      std::vector<value_type>& lvset_values) const = 0;


    /**
     * Get the level set gradients at quadrature points.
     */
    virtual void extract_level_set_gradients(
      const dealii::FEValuesBase<dim>& fe,
      std::vector<dealii::Tensor<1, dim, value_type>>& lvset_grads) const = 0;


    /**
     * Construct the FEValues object.
     * Done by side-casting to SimulatorBase<dim, NumberType>
     * \note The memory must be released by the programmer.
     */
    dealii::FEValues<dim>* fe_values(const dealii::Quadrature<dim>& quad,
                                     dealii::UpdateFlags update_flags) const;


    /**
     * Get the active cell iterators.
     * Again done by side-casting to SimulatorBase<dim, NumberType>
     */
    auto active_cell_iterators() const;


    /**
     * Return 1 if the point is in the domain defined by the level set,
     * while return 0 if it is outside. Note that at the transistion zone
     * near the interface, the value will be smoothed by the smoothing
     * method defined in the control parameters. Note that we define the level
     * set to be negative in the domain and positive outside the domain.
     */
    virtual value_type domain_identity(value_type x) const = 0;


    /**
     * Virtual destructor
     */
    virtual ~LevelSetBase() = default;
  };


  /* ************************************************** */
  /**
   * Control parameters for LevelSetSurface
   */
  /* ************************************************** */
  template <typename AdvectSimType, typename ReinitSimType>
  class LevelSetControl : public AdvectSimType::control_type
  {
   public:
    using advect_control_t = typename AdvectSimType::control_type;
    using reinit_control_t = typename ReinitSimType::control_type;
    using value_type = typename AdvectSimType::value_type;


    /**
     * Constructor.
     */
    LevelSetControl();


    /**
     * Set artificial viscosity for reinit simulator.
     */
    void set_artificial_viscosity(value_type viscosity);


    /**
     * Setting the mesh refinement and reinitialization interval
     * to the same value.
     */
    void set_refine_reinit_interval(unsigned int interval);


    /**
     * Pointer to reinit simulator control.
     */
    std::shared_ptr<reinit_control_t> ptr_reinit;


    /**
     * The flag to execute reinitialization or not.
     */
    bool execute_reinit = true;


    /**
     * Reinit will only be run after this many cycles.
     */
    unsigned int reinit_frequency = 10;


    /**
     * This value, when \c attach_control is called with argument of this
     * object, will be copied into \c AdvectWrapper.
     */
    value_type refinement_width_coeff = 3.0;


    /**
     * Smoothing method for Heaviside function.
     * This function is used for computing material parameter mixing.
     */
    HeavisideSmoothing heaviside_smoothing = HeavisideSmoothing::sine;
  };


  /* ************************************************** */
  /**
   * Interface Capture solver
   * \ingroup LevelSet
   * \todo this class can directly inherit from \c AdvectSimulator
   */
  /* ************************************************** */
  template <typename AdvectSimType, typename ReinitSimType>
  class LevelSetSurface
    : public AdvectSimType,
      public LevelSetBase<AdvectSimType::dimension,
                          typename AdvectSimType::value_type>
  {
   public:
    constexpr static int dim = AdvectSimType::spacedim;
    constexpr static int dimension = AdvectSimType::spacedim;
    constexpr static int spacedim = AdvectSimType::spacedim;
    using advect_solver_type = AdvectSimType;
    using reinit_solver_type = ReinitSimType;
    using time_step_type = typename AdvectSimType::time_step_type;
    using value_type = typename AdvectSimType::vector_type::value_type;
    using control_type = LevelSetControl<AdvectSimType, ReinitSimType>;
    using vector_type = typename AdvectSimType::vector_type;


    /** \name Basic object behavior */
    //@{
    /**
     * Constructor.
     */
    LevelSetSurface(Mesh<dim>& triag,
                    unsigned int pdegree,
                    const std::string& label = "LevelSet");

    /**
     * Attach control.
     */
    void attach_control(const std::shared_ptr<control_type>& pcontrol);


    /**
     * Initialize the solver.
     * Setting boundary conditions, velocity field and initial condition.
     * Initialize both advect_solver and reinit_solver.
     */
    template <typename VeloFcnType>
    void initialize(const ICBase<dim, value_type>& initial_condition,
                    const std::shared_ptr<VeloFcnType>& field,
                    bool refine_mesh = true);
    //@}


    /** Accessing and Setting Object Members */
    //@{
    /**
     * Define the quadrature to be used by both advection and reinit simulators
     */
    template <typename QuadratureType>
    void set_quadrature(const QuadratureType& quad);

    /**
     * brief Get the control object
     */
    const control_type& get_control() const;
    //@}


    /** \name Temporal Related Methods*/
    //@{
    /**
     * Advance the level set for given \c time_step.
     * Whenever a reinit_gradient_threshold is reached, then
     * reinitialization algorithm is automatically triggered.
     */
    void advance_time(time_step_type time_step) override;

    /**
     * Update the simulator time.
     */
    time_step_type advance_time(time_step_type time_step,
                                bool compute_single_cycle) override;

    /**
     * Testing if this object is synchronous with advection simulator.
     * And also test if the advection simulator is internally synchronized.
     */
    bool is_synchronized() const;
    //@}


    /**
     * \name Coupling Tools.
     */
    //@{
    /**
     * Return 1 if the point is in the domain defined by the level set,
     * while return 0 if it is outside. Note that at the transistion zone
     * near the interface, the value will be smoothed by the smoothing
     * method defined in the control parameters. Note that we define the level
     * set to be negative in the domain and positive outside the domain.
     */
    value_type domain_identity(value_type x) const override;


    /**
     * Compute level set values for given \c FEValuesBase object.
     */
    void extract_level_set_values(
      const dealii::FEValuesBase<dim>& feval,
      std::vector<value_type>& lvset_values) const override;


    /**
     * Compute level set gradients for given \c FEValuesBase object.
     */
    void extract_level_set_gradients(
      const dealii::FEValuesBase<dim>& feval,
      std::vector<dealii::Tensor<1, dim, value_type>>& lvset_grads)
      const override;


    /**
     * Add this entry to advect and reinit simulators.
     */
    void append_boundary_condition(
      const std::weak_ptr<BCFunction<dim, value_type>>& pbc);

    /**
     * Generate a set of filtered iterators that point to cells
     * whose vertices have level set values below given \c threshold.
     */
    template <typename Iterator>
    dealii::IteratorRange<dealii::FilteredIterator<Iterator>>
    cells_near_interface(dealii::IteratorRange<Iterator> cells,
                         value_type threshold) const;
    //@}


    /** \name Post-processing */
    //@{
    /**
     * Integrate the mass of the current solution vector.
     */
    value_type integrate_mass(const dealii::Quadrature<dim>& quadrature) const;

    /**
     * Integrate mass error.
     * \pre{Initialization is done by a function rather than a vector of nodal
     * values}
     */
    value_type compute_mass_error(
      const ScalarFunction<dim, value_type>& analytical_soln,
      const dealii::Quadrature<dim>& quadrature,
      bool compute_relative_error = true) const;

    /**
     * Integrate mass error.
     * Overloading the previous function with quadrature order = fe_degree + 1.
     * \pre{Initialization is done by a function rather than a vector of nodal
     * values}
     */
    value_type compute_mass_error(
      const ScalarFunction<dim, value_type>& analytical_soln,
      bool compute_relative_error = true) const;


    /**
     * Allocate a CurvatureEstimator object
     * and estimate the curvature
     */
    const vector_type& estimate_curvature() const;


    /**
     * Export the solution to file.
     * If the curvature is present, export that too.
     */
    void export_solution(ExportFile& file) const override;
    //@}


    /**
     * Level set reinitialization solver
     */
    ReinitSimType reinit_simulator;


   protected:
    /**
     * Predicate class to test if a cell iterator is close to the interface
     */
    class InterfaceCellFilter;

    /**
     * Predicate class to test whether
     * the isosurface at the cell center have a
     * dip angle within certain range
     */
    class DipAngleCellFilter;

    /**
     * Solve curvature in the weak sense
     */
    class CurvatureEstimator;


    /**
     * Refine elements that are close to the interface
     */
    void do_flag_mesh_for_coarsen_and_refine(
      const MeshControl<value_type>& mesh_control) const override;


   private:
    /**
     * Make sure that the types and spatial dimension is all consistent
     */
    void check_consistency() const;


    /**
     * A string to identify the current level set simulator.
     */
    const std::string label_string;


    /**
     * No of steps passed without reinitialization
     */
    unsigned int n_steps_without_reinit;


    /**
     * Pointer to the control structure
     */
    std::shared_ptr<control_type> ptr_control;


    /**
     * Curvature estimator.
     * Will be allocated when \c estimate_curvature() is called.
     */
    mutable std::unique_ptr<CurvatureEstimator> ptr_curvature_estimator =
      nullptr;
  };


  /* ---------------------------------- */
  /**
   * Test whether the dip angle of the cell center
   * is within the range
   */
  /* ---------------------------------- */
  template <typename Advect, typename Reinit>
  class LevelSetSurface<Advect, Reinit>::DipAngleCellFilter
  {
   public:
    /**
     * @brief Construct the dip angle cell filter
     */
    DipAngleCellFilter(const LevelSetSurface<Advect, Reinit>& level_set_sim);


    /**
     * @brief Set the range of the dip angle.
     * By default
     */
    void set_threshold(value_type lower_bound,
                       value_type upper_bound = 90.0,
                       bool convert_degree_to_radian = true);


    /**
     * @brief Predicate.
     */
    template <typename Iterator>
    bool operator()(const Iterator& it) const;


   private:
    /**
     * Pointer to FEValues
     */
    std::unique_ptr<dealii::FEValues<dim>> ptr_fevals;

    /**
     * Pointer to the solution vector
     */
    dealii::SmartPointer<const vector_type, DipAngleCellFilter> ptr_soln_vector;

    /**
     * lower limit of the dip angle
     */
    value_type lower_bound = -1.0;

    /**
     * upper limit of the dip angle
     */
    value_type upper_bound = -1.0;
  };


  /* ---------------------------------- */
  /**
   * A class for estimating the curvature
   * of the level set solution
   */
  /* ---------------------------------- */
  template <typename Advect, typename Reinit>
  class LevelSetSurface<Advect, Reinit>::CurvatureEstimator
  {
   public:
    using this_type = CurvatureEstimator;
    using level_set_type = LevelSetSurface<Advect, Reinit>;
    constexpr static const int dim = level_set_type::dim;

    using linsys_type = typename Advect::linsys_type;
    using CellIterator = typename dealii::DoFHandler<dim>::active_cell_iterator;
    using TimedVector = TimedSolutionVector<vector_type, value_type>;

    /**
     * Constructor
     */
    CurvatureEstimator(const LevelSetSurface<Advect, Reinit>& level_set);

    /**
     * Run curvature estimation
     */
    const vector_type& estimate();

    /**
     * Get the wrapped curvature vector
     */
    const TimedVector& get_curvature() const;

    /**
     * Get the curvature vector
     */
    const vector_type& get_curvature_vector() const;


   private:
    struct ScratchData;
    struct ScratchDataBox;
    using CopyData = CellCopyData<dim, value_type>;
    using CopyDataBox = CellCopyDataBox<CopyData>;

    /**
     * Assemble the rhs vector.
     */
    template <typename QuadratureType>
    auto assemble_rhs(const QuadratureType& quadrature) ->
      typename std::enable_if<
        std::is_base_of<dealii::Quadrature<dim>, QuadratureType>::value,
        void>::type;


    /**
     * Assemble per cell
     */
    void local_assembly(const CellIterator& cell,
                        ScratchDataBox& scratch_box,
                        CopyDataBox& copy_box);

    /**
     * Copy the local values to global values
     */
    void copy_local_to_global(const CopyDataBox& copy_box);


    /**
     * Execute cell-volume assembly
     */
    void cell_assembly(const ScratchData& scratch, CopyData& copy);


    /**
     * Execute face assemly
     */
    void face_assembly(const ScratchData& scratch_in,
                       const ScratchData& scratch_ex,
                       CopyData& copy_in,
                       CopyData& copy_ex);


    /**
     * Execute boundary assembly
     */
    void boundary_assembly(const ScratchData& s, CopyData& c);


    /**
     * Pointer to the DoFHandler in the level set simulator.
     */
    const dealii::SmartPointer<const dealii::DoFHandler<dim>, this_type>
      ptr_dof_handler;


    /**
     * Pointer to the solution vector
     */
    const dealii::SmartPointer<const TimedVector, this_type> ptr_lvset_soln;


    /**
     * Reuse the matrix in the advection simulator
     */
    const dealii::SmartPointer<const linsys_type, this_type> ptr_linear_system;


    /**
     * The rhs vector for gradient approximation
     */
    vector_type rhs;


    /**
     * The curvature solution
     */
    TimedVector curvature;


    /****************************/
    /** ThreadLocal ScratchData */
    /****************************/
    struct ScratchData : public CellScratchData<dim>
    {
      using base_type = CellScratchData<dim>;

      /**
       * Construct a new ScratchData
       */
      ScratchData(FEValuesEnum fevalenum, const vector_type& lvset_soln);


      /**
       * Initialize the cache vectors to appropriate size
       */
      void allocate();


      /**
       * \name Reinit FEValues-family objects
       */
      //@{
      /**
       * Reinitialize the ScratchData to a new cell
       */
      void reinit(const CellIterator& cell);

      /**
       * Reinitialize the ScratchData to a new cell face
       */
      void reinit(const CellIterator& cell, unsigned int face_no);

      /**
       * Reinitialize the ScratchData to a new cell face
       */
      void reinit(const CellIterator& cell,
                  unsigned int face_on,
                  unsigned int subface_no);

      /**
       * Reinitialize the local solution by computing
       * the solution values and the solution gradients
       * at quadrature points.
       */
      void reinit_local_solution();
      //@}


      /**
       * Pointer to the global solution vector
       */
      dealii::SmartPointer<const vector_type> ptr_lvset_soln;


      /**
       * Local solution at the cell quadrature points
       */
      std::vector<value_type> soln_qpt;


      /**
       *  Normalized solution gradient at qudrature points
       */
      std::vector<dealii::Tensor<1, dim, value_type>> soln_grad_qpt;
    };


    /*******************************/
    /** ThreadLocal ScratchDataBox */
    /*******************************/
    struct ScratchDataBox : public CellScratchDataBox<ScratchData>
    {
      using base_type = CellScratchDataBox<ScratchData>;

      /**
       * Constructor
       */
      template <typename QuadratureType>
      ScratchDataBox(const dealii::Mapping<dim>& mapping,
                     const dealii::FiniteElement<dim>& fe,
                     const QuadratureType& quad,
                     const dealii::UpdateFlags update_flags,
                     const vector_type& lvset_soln);
    };
  };


  /* ---------------------------------- */
  /**
   * Predicate class to test
   * if the cell is near the interface
   */
  /* ---------------------------------- */
  template <typename Advect, typename Reinit>
  class LevelSetSurface<Advect, Reinit>::InterfaceCellFilter
  {
   public:
    using outer_class = LevelSetSurface<Advect, Reinit>;
    using vector_type = typename outer_class::vector_type;
    constexpr static int dim = outer_class::dim;

    /**
     *  Construct a new InterfaceCellFilter object
     */
    InterfaceCellFilter(const LevelSetSurface<Advect, Reinit>& level_set_sim);


    /**
     * Set the threshold for the interface cell
     */
    void set_threshold(value_type thres);


    /**
     * Predicate function to test if
     * the cell is within the threshold
     */
    template <typename Iterator>
    bool operator()(const Iterator& it) const;


   protected:
    /**
     * If one of vertex in the cell has a level set value that is
     * within the threshold, then this cell will be included.
     */
    value_type threshold;

    /**
     * Pointer to FEValues.
     */
    std::shared_ptr<dealii::FEValues<dim>> ptr_fevals;

    /**
     * Pointer to the solution vector
     */
    dealii::SmartPointer<const vector_type, InterfaceCellFilter>
      ptr_soln_vector;
  };


  /* ---------------------------------------------- */
  /* Utilities functions for level set computations */
  /* ---------------------------------------------- */

  /* ************************************************** */
  /**
   * \name (Smoothed-) Heaviside Functions
   */
  /* ************************************************** */
  //@{
  /**
   * Base class for all smoothed Heaviside functions.
   */
  template <typename NumberType>
  class SmoothedHeavisideFunctionBase
  {
   public:
    using value_type = NumberType;

    virtual value_type operator()(value_type x) const = 0;

    value_type get_smoothing_width() const { return smoothing_width; }

   protected:
    /** Constructor */
    SmoothedHeavisideFunctionBase(value_type width) : smoothing_width(width)
    {
      ASSERT(width > 0.0, ExcArgumentCheckFail());
    }

    /** Destructor */
    virtual ~SmoothedHeavisideFunctionBase() = default;

   private:
    value_type smoothing_width;
  };

  /** Generic declaration */
  template <HeavisideSmoothing method = HeavisideSmoothing::none,
            typename NumberType = types::DoubleType>
  class HeavisideFunction;


  template <typename NumberType>
  class HeavisideFunction<HeavisideSmoothing::none, NumberType>
  {
   public:
    using value_type = NumberType;
    value_type operator()(value_type x) const
    {
      return static_cast<value_type>(x > 0.0 ? 1.0 : (x < 0.0 ? 0.0 : 0.5));
    }
  };


  template <typename NumberType>
  class HeavisideFunction<HeavisideSmoothing::linear, NumberType>
    : public SmoothedHeavisideFunctionBase<NumberType>
  {
   public:
    using value_type = NumberType;

    HeavisideFunction(value_type width)
      : SmoothedHeavisideFunctionBase<value_type>(width)
    {}

    value_type operator()(value_type x) const override;
  };

  template <typename NumberType>
  class HeavisideFunction<HeavisideSmoothing::sine, NumberType>
    : public SmoothedHeavisideFunctionBase<NumberType>
  {
   public:
    using value_type = NumberType;

    HeavisideFunction(value_type width)
      : SmoothedHeavisideFunctionBase<value_type>(width)
    {}

    value_type operator()(value_type x) const override;
  };
  //@}

}  // namespace ls

FELSPA_NAMESPACE_CLOSE

/* ----- IMPLEMENTATIONS ----- */
#include "src/level_set.implement.h"
/* -------------------------- */

#endif  // _FELSPA_LEVEL_SET_LEVEL_SET_H_ //
