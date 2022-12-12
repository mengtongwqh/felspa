#ifndef _FELPA_COUPLED_LEVEL_SET_STOKES_H_
#define _FELPA_COUPLED_LEVEL_SET_STOKES_H_

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/particle_handler.h>
#include <felspa/base/quadrature.h>
#include <felspa/level_set/level_set.h>
#include <felspa/level_set/material_stack.h>
#include <felspa/pde/pde_base.h>
#include <felspa/pde/stokes.h>

FELSPA_NAMESPACE_OPEN

// forward declaration
template <typename, typename>
class LevelSetStokes;

namespace dg
{
#ifdef FELSPA_ENALBE_LDG_REINIT
  template <int dim, typename NumberType>
  using LevelSetStokesSimulator =
    LevelSetStokes<dg::AdvectSimulator<dim, NumberType>,
                   dg::ReinitSimulatorLDG<dim, NumberType>>;
#else
  template <int dim, typename NumberType>
  using LevelSetStokesSimulator =
    LevelSetStokes<dg::AdvectSimulator<dim, NumberType>,
                   dg::ReinitSimulator<dim, NumberType>>;
#endif
}  // namespace dg

namespace internal
{
  template <int dim, typename NumberType,
            typename QuadratureType = QEvenlySpaced<dim>>
  class LevelSetStokesParticles;
}  // namespace internal


/* ************************************************** */
/**
 * A coupled simulator with \c StokesSimulator modelling
 * the velocity field (the "flow") and a collection
 * of \c MaterialCompositor describing material interfaces.
 * The usage of this class:
 * -# Call constructor
 * -# Add boundary conditions for \c StokesSimulator
 *  and \c MaterialCompositor.
 * -# Setup materials and LevelSet geometry. Level set simulators
 *  will be constructed at this time.
 * -# Attach a mesh refiner if adaptive refinement is desired
 * -# Call \c initialize() which will setup background material for
 *  \c MaterialCompositor and gravity model for \c StokesSimlator.
 *  \c StokesSimulator will be initialized here.
 */
/* ************************************************** */
template <typename AdvectType, typename ReinitType>
class LevelSetStokes : TimedPhysicsObject<typename AdvectType::value_type>
{
 public:
  class Control;
  struct Signals;

  using value_type = typename AdvectType::value_type;
  using control_type = Control;
  using time_step_type = typename AdvectType::time_step_type;
  using level_set_type = ls::LevelSetSurface<AdvectType, ReinitType>;
  using vector_type = typename AdvectType::vector_type;
  constexpr static int dim = AdvectType::dimension, dimension = dim;
  using Particles = internal::LevelSetStokesParticles<dim, value_type>;

  friend class internal::LevelSetStokesParticles<dim, value_type>;


  /**
   * Constructor.
   * By default we construct linear level set and Taylor-Hood element.
   */
  LevelSetStokes(Mesh<dim, value_type>& mesh,
                 unsigned int level_set_degree = 1,
                 unsigned int stokes_degree_velocity = 2,
                 unsigned int stokes_degree_pressure = 1,
                 const std::string& label = "LevelSetStokes");


  /**
   * Copy constructor - deleted.
   * \todo Maybe we should implement this by a shallow copy and define a
   * separate RHS.
   */
  LevelSetStokes(const LevelSetStokes<AdvectType, ReinitType>&) = delete;


  /**
   * Copy assignment - deleted.
   */
  LevelSetStokes<AdvectType, ReinitType>& operator=(
    const LevelSetStokes<AdvectType, ReinitType>&) = delete;


  /**
   * Append boundary condition for Stokes solver.
   */
  void set_bcs_stokes(
    const std::vector<std::weak_ptr<BCFunctionBase<dim, value_type>>>&);


  /**
   * Append level set boundary conditions.
   */
  void set_bcs_level_set(
    const std::vector<std::weak_ptr<BCFunction<dim, value_type>>>&);


  /**
   * Attach mesh refiner to all level set simulators as well as the
   * StokesSimulator. After this is done, the simulator will label
   * the mesh for coarsening and refinement after each cycle.
   */
  void attach_mesh_refiner(
    const std::shared_ptr<MeshRefiner<dim, value_type>>& p_mesh_refiner,
    bool stokes_compute_refinement = true);


  /**
   * Adding material and its enclosed domain to the coupled simulator.
   * If a shared_ptr to a mesh refiner object is also passed in,
   * then the level set simulator will utilize the mesh refiner
   */
  void add_material_domain(
    const std::shared_ptr<MaterialBase<dim, value_type>>& p_material,
    const std::shared_ptr<const ls::ICBase<dim, value_type>>& p_lvset_ic,
    bool participate_in_mesh_refinement = true);


  /**
   * Signal the object that all mateirals have been added.
   * The mesh will be stepwise refined.
   * -# each level set solver will mark the mesh for refinement.
   * -# Then the mesh refiner will refine the mesh altogether.
   * -# The initial conditions will be interpolated to the refined mesh.
   */
  void finalize_material_domains();


  /**
   * Initialize the simulator with appropriate.
   * -# Call \c initialize() on the Stokes simulator and set the gravity model.
   * -# Initialize the \c MaterialCompositor by setting the background material.
   * -# Propagate boundary conditions to all level set simulators
   */
  void initialize(const std::shared_ptr<MaterialBase<dim, value_type>>&
                    ptr_background_material,
                  const std::shared_ptr<TensorFunction<1, dim, value_type>>&
                    p_gravity_model = nullptr);


  /**
   * Setup particle handler for this solver
   */
  void setup_particles(const std::shared_ptr<Particles>& ptr_ptcl_handler);


  /**
   * Setup particle handler for this solver
   */
  void setup_particles(
    unsigned int nptcls_per_dim,
    bool automatic_replenish_particles,
    unsigned int level_set_id = constants::invalid_unsigned_int,
    bool generate_ptcls_near_interface = false,
    bool reinsert_periodic_particles = false);


  /**
   * Get control structure
   */
  Control& control();


  /**
   * Get the material stack.
   */
  const ls::MaterialStack<dim, value_type>& get_materials() const;


  /**
   * @brief Get the Stokes simulator
   */
  const StokesSimulator<dim, value_type>& get_stokes_simulator() const;


  /**
   * @brief Get the advection simulator associated with material n
   *
   */
  const level_set_type& get_level_set(unsigned int i) const;


  /**
   * Obtain the label string.
   */
  const std::string& get_label_string() const { return label_string; }


  /**
   * Test if the simulator is properly initialized.
   */
  bool is_initialized() const;


  /**
   * Test if the Stokes FEM solver and level set solvers
   * are synchronized.
   */
  bool is_synchronized() const;


  /**
   * Advance the time step by alternating updates on flow field and interface.
   * -# Under the current material config,
   *  compute velocity from \c StokesSimulator
   * -# Advect the material configuration with the computed velocity field.
   */
  void advance_time(time_step_type time_step) override;


  /**
   * Advance the time step using the same scheme above but using the
   * adaptive time step adjustment dictated by level set advection CFL.
   */
  time_step_type advance_time(time_step_type time_step,
                              bool compute_single_step);


  /**
   * Export the solution to output file.
   */
  void export_solution(const std::string& file_name_stem,
                       ExportFileFormat file_format) const;


  /**
   * Export multiple time steps.
   * When the simulation ends, a pvd file, containing a list of
   * corresponding vtu files and physical time points, will be exported.
   */
  void export_solutions(const std::string& path = constants::current_dir) const;


  /**
   * \name Constants related to the simulators.
   */
  //@{
  /** Polynomial degree for level set */
  const unsigned int level_set_degree;

  /** Polynomial degree for velocity */
  const unsigned int stokes_velocity_degree;

  /** Polynomial degree for pressure */
  const unsigned int stokes_pressure_degree;
  //@}


  /**
   * @brief Signal slots.
   */
  mutable Signals signals;


 private:
  /**
   * Pointer to the mesh object.
   */
  const dealii::SmartPointer<Mesh<dim, value_type>,
                             LevelSetStokes<AdvectType, ReinitType>>
    ptr_mesh;


  /**
   * Cell quadrature used by level_set solver
   */
  const dealii::QGauss<dim> level_set_quadrature;


  /**
   * Cell quadrature used by Stokes flow solver
   */
  const dealii::QGauss<dim> stokes_quadrature;


  /**
   * Pointers to boundary conditions.
   */
  std::vector<std::weak_ptr<BCFunction<dim, value_type>>> ptrs_bc;


  /**
   * Shared pointer to the mesh refiner
   */
  std::shared_ptr<MeshRefiner<dim, value_type>> ptr_mesh_refiner = nullptr;


  /**
   * Pointer to the Stokes solver.
   * This will be constructed in the constructor.
   */
  const std::shared_ptr<StokesSimulator<dim, value_type>> ptr_stokes;


  /**
   * Pointer to material solver.
   * This will be constructed in the \c initialize() function.
   */
  const std::shared_ptr<ls::MaterialStack<dim, value_type>> ptr_materials;


  /**
   * List of all level set simulators.
   * This object will not manage the resources of
   * the level set simulators
   */
  std::vector<std::weak_ptr<level_set_type>> ptrs_level_set;


  /**
   * Temporary storing the initial conditions
   */
  std::vector<std::shared_ptr<const ls::ICBase<dim, value_type>>> ptrs_lvset_ic;


  /**
   * Pointer to the control struct.
   */
  const std::shared_ptr<control_type> ptr_control;


  /**
   *  Pointer to a particle handler object
   */
  std::shared_ptr<Particles> ptr_particles = nullptr;


  /**
   * Flag if the coupled simulator is initialized.
   */
  bool initialized = false;


  /**
   * When this is set to \c true, the simulator will
   * no more accept materials.
   */
  bool material_domain_finalized = false;


  /**
   * A label string for description.
   */
  const std::string label_string;
};


/* ************************************************** */
/**
 * Control parameters for the coupled simulator.
 */
/* ************************************************** */
template <typename AdvectType, typename ReinitType>
class LevelSetStokes<AdvectType, ReinitType>::Control
{
 public:
  using value_type = typename AdvectType::value_type;
  constexpr static int dim = AdvectType::dimension;
  using StokesControl = typename StokesSimulator<dim, value_type>::Control;

  /**
   * Constructor
   */
  Control()
    : ptr_level_set(
        std::make_shared<ls::LevelSetControl<AdvectType, ReinitType>>()),
      ptr_stokes(std::make_shared<StokesControl>())
  {
    set_refine_reinit_interval(10);
  };


  /**
   * This is time step interval for level set reinitialization
   * and mesh adjustments.
   */
  void set_refine_reinit_interval(unsigned int interval);


  /**
   * Set coarsening and refinement limit.
   */
  void set_coarsen_refine_limit(int coarsen_limit, int refine_limit);


  /**
   * Pointer to the level set control.
   */
  std::shared_ptr<ls::LevelSetControl<AdvectType, ReinitType>> ptr_level_set;


  /**control().ptr_level_set->
   * Pointer to Stokes simulator.
   */
  std::shared_ptr<StokesControl> ptr_stokes;
};


template <typename AdvectType, typename ReinitType>
struct LevelSetStokes<AdvectType, ReinitType>::Signals
{
  /**
   * @brief Functions that will be called
   */
  boost::signals2::signal<void()> post_advance_full_timestep;


  /**
   * @brief
   *
   */
  boost::signals2::signal<void()> post_advance_sub_timestep;


  /**
   * @brief Functions that will be called after solution is exported
   * This can be used if additional exports are desired
   */
  boost::signals2::signal<void()> post_export;
};


/* ------------------- */
namespace internal
/* ------------------- */
{
  /* ************************************************** */
  /**
   *  Tracer particles
   * Memory layout for PorpertyPool:
   * SIZE      | PHYSICAL PARAMETER
   * 1         | spawn time, or time of creation
   * 1         | density
   * 1         | viscosity
   * 1         | flinn_slope
   * 1         | Wk, or kinematic vorticity
   * dim       | velocity
   * dim x dim | deformation gradient tensor
   * dim x dim | velocity gradient tensor
   * 1         | level set values of the reference level set
   * dim       | normal vector to the reference level set
   */
  /* ************************************************** */
  template <int dim, typename NumberType, typename QuadratureType>
  class LevelSetStokesParticles : public TimedPhysicsObject<NumberType>
  {
   public:
    using value_type = NumberType;
    using time_step_type = value_type;
    using cell_iterator_type =
      typename dealii::DoFHandler<dim>::active_cell_iterator;
    using this_type = LevelSetStokesParticles<dim, NumberType, QuadratureType>;
    using size_type = types::ParticleIndex;

    static_assert(
      std::is_base_of<dealii::Quadrature<dim>, QuadratureType>::value,
      "QuadratureType must be derived from dealii::Quadrature<dim>.");


    /**
     * Construct a new Stokes Particles object.
     */
    template <typename Advect, typename Reinit>
    LevelSetStokesParticles(
      const LevelSetStokes<Advect, Reinit>& levelset_stokes_sim,
      unsigned int nptcls_per_dim,
      bool automatic_replenish_particles = true,
      const std::shared_ptr<ls::LevelSetBase<dim, value_type>>&
        p_ref_level_set = nullptr,
      bool generate_ptcls_near_interface = false,
      bool reinsert_periodic_particles = false);


    /**
     *  Do not allow copy construction
     */
    LevelSetStokesParticles(
      const LevelSetStokesParticles<dim, value_type, QuadratureType>&) = delete;


    /**
     * Do not allow copy assignment.
     */
    LevelSetStokesParticles<dim, NumberType, QuadratureType>& operator=(
      const LevelSetStokesParticles<dim, NumberType, QuadratureType>&) = delete;


    /**
     * This will move the particles and update all kinematic parameters
     */
    void advance_time(time_step_type dt) override;


    /**
     * When a particle leaves the domain through the periodic boundary,
     * we create a new ptcl on the other side of the periodic boundary
     * so that it can be reinserted later in the particle_handler.
     */
    void collect_escaped_periodic_particles(
      const typename dealii::Particles::ParticleIterator<dim>& particle,
      const typename dealii::Triangulation<dim>::active_cell_iterator& cell);


    /**
     * Output particles
     */
    void export_solution(ExportFile& file) const;


    /**
     *  Output particles for this step and also append this record to the pvd.
     */
    void export_solutions(const std::string& path) const;


    /**
     * Get the label string of this object
     */
    const std::string& get_label_string() const { return label; }


   protected:
    /**
     * Number of property entries for each particle.
     */
    constexpr static unsigned int n_property_entries =
      6 + 2 * dim + 2 * dim * dim;

    constexpr static value_type invalid_Wk = -1.0;


    /**
     * An enum for the each entry of the property and
     * its location in the \c PropertyPool array.
     */
    enum Property : unsigned int
    {
      spawn_time = 0,
      density = 1,
      viscosity = 2,
      flinn_slope = 3,
      Wk = 4,
      velocity = 5,
      velocity_gradient = 5 + dim,
      deformation_gradient = 5 + dim + dim * dim,
      level_set = 5 + dim + 2 * dim * dim,
      level_set_normal = 6 + dim + 2 * dim * dim,
    };


    /**
     * @brief Generate particles near the interface.
     * Initialize the material parameter (density, viscosity...) by
     * what is defined in the level set material stack.
     */
    template <typename Iterator>
    void generate_particles(const dealii::IteratorRange<Iterator>& it_range);


    /**
     * @brief Actions to be performed on each cell
     * for generating particles
     */
    template <typename CellIterator>
    void generate_particles_in_cell(const CellIterator& cell);


    /**
     * In some cases the expansion of the surface will result in
     * cells with no particles. This function will replenish particles
     * into these cells.
     */
    void replenish_particles();


    /**
     * This function will
     * -# move the particles to the up-to-date location
     * -# update the deformation gradient tensor
     * -# through polar decomposition, find max stretching/compression axis
     * @note the material parameter is not updated.
     * It can be done, but we need an efficient way to implement that.
     */
    void update_cell_kinematics(const cell_iterator_type& cell,
                                const dealii::Quadrature<dim>& quad,
                                time_step_type dt);


    /**
     * Update the function values and function normals
     */
    void update_cell_level_set(const cell_iterator_type& cell,
                               const dealii::Quadrature<dim>& quad);


    /**
     * Pointer to a triangulation object
     */
    dealii::SmartPointer<Mesh<dim, value_type>, this_type> ptr_mesh;


    /**
     * @brief Weak pointer to the StokesSimulator
     */
    std::weak_ptr<const StokesSimulator<dim, value_type>> ptr_stokes;


    /**
     * @brief Number of particles in each dimension
     */
    const unsigned int nptcls_per_dim;


    /**
     * @brief  Weak pointer to MaterialStack
     */
    std::weak_ptr<MaterialBase<dim, value_type>> ptr_materials;


    /**
     * Particle handler
     */
    dealii::Particles::ParticleHandler<dim> particle_handler;


    /**
     *  Label for the this particle object
     */
    std::string label;


    /**
     * Counter of the current particle number.
     */
    size_type counter = 0;


   private:
    /**
     * @brief Compute the objective kinematic vorticity.
     * This is done by first estimating the frame rotation from eigenvectors
     * of L and take that away from the vorticity tensor.
     * @note if D in the previous step is [0],
     * return -1.0 as a signal for invalid value.
     * @param L_prev  velocity gradient tensor of the previous time step
     * @param L_current  velocity gradient tensor at current configuration
     * @param dt time step
     * @return objective kinematic vorticity Wk
     */
    static value_type objective_kinematic_vorticity(
      const dealii::Tensor<2, dim, value_type>& L_prev,
      const dealii::Tensor<2, dim, value_type>& L_current, value_type dt);

    /**
     * \f$ F^{n+1} = (I + L\delta{t}) F^{n} $\f
     * @param[out] deformation_gradient
     * @param[in] velocity_gradient
     * @param[in] dt
     */
    void integrate_deformation_gradient(
      value_type* deformation_gradient,
      const dealii::Tensor<2, dim, value_type>& l, time_step_type dt) const;


    /**
     * Compute the maximum stretching and compression axis, which is
     * the eigenvectors of \f$ F^{T}F $\f corresponding to its max/min
     * eigenvalues.
     */
    void principle_finite_strain_axis(value_type* max_stretch,
                                      value_type* max_compression,
                                      const value_type* F) const;


    /**
     * Particles whose deformation gradient tensor with
     * condition number greater than threshold will be reset
     * to identity F.
     */
    size_type reset_stretched_particles(
      value_type threshold_ratio = 1.0e2) const;


    /**
     * Unrolling indices of the 2d-array
     */
    int idx(int i, int j) const { return i * dim + j; }


    /**
     * Cache for paticles that escaped from the periodic boundary
     */
    std::multimap<typename dealii::Triangulation<dim>::active_cell_iterator,
                  dealii::Particles::Particle<dim>>
      ptcls_to_reinsert;


    /**
     * Shared pointer to the level set,
     * where the level set values the normals will be computed.
     */
    const std::shared_ptr<ls::LevelSetBase<dim, value_type>>
      ptr_reference_level_set;


    dealii::Particles::PropertyPool<dim> property_pool_reinsert;


    /**
     * An object recording the vtu files and at destruction time
     * spit out a pvd file for animation.
     * The record will be written if \c export_solutions() is called.
     */
    mutable PVDCollector<time_step_type> pvd_collector;


    bool automatic_replenish_particles;
  };

}  // namespace internal


FELSPA_NAMESPACE_CLOSE

/* --------- IMPLEMENTATIONS ----------- */
#include "src/level_set_stokes.implement.h"
/* ------------------------------------- */

#endif  // _FELPA_COUPLED_LEVEL_SET_STOKES_H_ //
