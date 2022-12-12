#ifndef _FELPA_COUPLED_LEVEL_SET_STOKES_IMPLEMENT_H_
#define _FELPA_COUPLED_LEVEL_SET_STOKES_IMPLEMENT_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/numerics/data_component_interpretation.h>
#include <felspa/base/exceptions.h>
#include <felspa/base/log.h>
#include <felspa/base/src/exception_classes.h>
#include <felspa/coupled/level_set_stokes.h>


FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/*                  LevelSetStokes                    */
/* ************************************************** */
template <typename AdvectType, typename ReinitType>
LevelSetStokes<AdvectType, ReinitType>::LevelSetStokes(
  Mesh<dim, value_type>& mesh, unsigned int level_set_degree_,
  unsigned int stokes_degree_velocity_, unsigned int stokes_degree_pressure_,
  const std::string& label)
  : level_set_degree(level_set_degree_),
    stokes_velocity_degree(stokes_degree_velocity_),
    stokes_pressure_degree(stokes_degree_pressure_),
    ptr_mesh(&mesh),
    level_set_quadrature(level_set_degree + 2),
    stokes_quadrature(stokes_velocity_degree + 1),
    ptr_stokes(std::make_shared<StokesSimulator<dim, value_type>>(
      mesh, dealii::FE_Q<dim>(stokes_velocity_degree),
      dealii::FE_Q<dim>(stokes_pressure_degree), label + "Stokes")),
    ptr_materials(std::make_shared<ls::MaterialStack<dim, value_type>>(
      label + "MaterialStack")),
    ptr_control(std::make_shared<control_type>()),
    label_string(label)
{
  ptr_stokes->attach_control(ptr_control->ptr_stokes);
  ptr_stokes->set_quadrature(stokes_quadrature);

  // forbid synchronization check from level set solvers
  ptr_stokes->set_allow_passive_update(false);
  ptr_materials->set_allow_passive_update(false);

  ASSERT(level_set_degree_ > 0, ExcArgumentCheckFail());
  ASSERT(stokes_degree_pressure_ > 0, ExcArgumentCheckFail());
}


template <typename AdvectType, typename ReinitType>
void LevelSetStokes<AdvectType, ReinitType>::initialize(
  const std::shared_ptr<MaterialBase<dim, value_type>>& ptr_background_material,
  const std::shared_ptr<TensorFunction<1, dim, value_type>>& p_gravity_model)
{
  ASSERT(this->initialized == false,
         EXCEPT_MSG("Initialization cannot be called twice."));
  ptr_materials->set_background_material(ptr_background_material);
  ASSERT(ptr_stokes.get(), ExcNullPointer());

  ptr_stokes->initialize(ptr_materials, p_gravity_model);

  this->initialized = true;
}


template <typename AdvectType, typename ReinitType>
void LevelSetStokes<AdvectType, ReinitType>::setup_particles(
  const std::shared_ptr<Particles>& ptr_ptcl_handler)
{
  ASSERT(is_initialized(), ExcSimulatorNotInitialized());
  ASSERT(ptr_ptcl_handler != nullptr, ExcNullPointer());
  ptr_particles = ptr_ptcl_handler;
}


template <typename AdvectType, typename ReinitType>
void LevelSetStokes<AdvectType, ReinitType>::setup_particles(
  unsigned int nptcls_per_dim, bool automatic_replenish_particles,
  unsigned int level_set_id, bool generate_ptcls_near_interface,
  bool reinsert_periodic_particles)
{
  std::shared_ptr<level_set_type> sp = nullptr;
  if (level_set_id != constants::invalid_unsigned_int) {
    sp = ptrs_level_set[level_set_id].lock();
    ASSERT(sp != nullptr, ExcExpiredPointer());
  }
  ptr_particles = std::make_shared<Particles>(
    *this, nptcls_per_dim, automatic_replenish_particles, sp,
    generate_ptcls_near_interface, reinsert_periodic_particles);
}


template <typename AdvectType, typename ReinitType>
bool LevelSetStokes<AdvectType, ReinitType>::is_initialized() const
{
  return initialized && material_domain_finalized && ptr_stokes != nullptr &&
         ptr_materials != nullptr && ptr_stokes->is_initialized() &&
         ptr_materials->has_background_material();
}


template <typename AdvectType, typename ReinitType>
bool LevelSetStokes<AdvectType, ReinitType>::is_synchronized() const
{
  return this->is_synced_with(*ptr_stokes) &&
         this->is_synced_with(*ptr_materials) &&
         (ptr_particles ? this->is_synced_with(*ptr_particles) : true);
}


template <typename AdvectType, typename ReinitType>
void LevelSetStokes<AdvectType, ReinitType>::set_bcs_stokes(
  const std::vector<std::weak_ptr<BCFunctionBase<dim, value_type>>>& pbcs)
{
  ASSERT(initialized == false,
         EXCEPT_MSG("BC must be added before initialization."));
  for (const auto& pbc : pbcs) {
    auto spbc = pbc.lock();
    ptr_stokes->append_boundary_condition(spbc);
  }
}


template <typename AdvectType, typename ReinitType>
FELSPA_FORCE_INLINE auto LevelSetStokes<AdvectType, ReinitType>::get_materials()
  const -> const ls::MaterialStack<dim, value_type>&
{
  ASSERT(ptr_materials != nullptr, ExcNullPointer());
  return *ptr_materials;
}


template <typename AdvectType, typename ReinitType>
FELSPA_FORCE_INLINE auto LevelSetStokes<AdvectType, ReinitType>::get_level_set(
  unsigned int i) const -> const level_set_type&
{
  auto sp = ptrs_level_set[i].lock();
  ASSERT(sp != nullptr, ExcExpiredPointer());
  return *sp;
}


template <typename AdvectType, typename ReinitType>
FELSPA_FORCE_INLINE auto
LevelSetStokes<AdvectType, ReinitType>::get_stokes_simulator() const
  -> const StokesSimulator<dim, value_type>&
{
  ASSERT(ptr_stokes != nullptr, ExcNullPointer());
  return *ptr_stokes;
}


template <typename AdvectType, typename ReinitType>
FELSPA_FORCE_INLINE auto LevelSetStokes<AdvectType, ReinitType>::control()
  -> Control&
{
  ASSERT(ptr_control, ExcNullPointer());
  return *ptr_control;
}


template <typename AdvectType, typename ReinitType>
void LevelSetStokes<AdvectType, ReinitType>::set_bcs_level_set(
  const std::vector<std::weak_ptr<BCFunction<dim, value_type>>>& pbc)
{
  ASSERT(ptrs_bc.empty(), EXCEPT_MSG("BC can only be added once"));
  ASSERT(initialized == false,
         EXCEPT_MSG("BC must be added before initialization."));
  ptrs_bc = pbc;
}


template <typename AdvectType, typename ReinitType>
void LevelSetStokes<AdvectType, ReinitType>::add_material_domain(
  const std::shared_ptr<MaterialBase<dim, value_type>>& p_material,
  const std::shared_ptr<const ls::ICBase<dim, value_type>>& p_lvset_ic,
  bool participate_in_mesh_refinement)
{
  ASSERT(!material_domain_finalized,
         EXCEPT_MSG("Materials are already finalized and cannot be added."));

  // allocate the level set simulator,
  // using material name as solver name.
  auto p_lvset = std::make_shared<level_set_type>(
    *ptr_mesh, level_set_degree,
    this->get_label_string() + p_material->get_label_string() + "LS");

  // attach the control structure to level set.
  p_lvset->attach_control(ptr_control->ptr_level_set);
  ptrs_level_set.push_back(p_lvset);  // append to ptrs_level_set

  // set the appropriate level set quadrature
  p_lvset->set_quadrature(level_set_quadrature);

  // propagate the boundary conditions
  for (auto& pbc : ptrs_bc) p_lvset->append_boundary_condition(pbc);

  // mesh refinement object
  if (participate_in_mesh_refinement) {
    ASSERT(ptr_mesh_refiner != nullptr, EXCEPT_MSG("Mesh refiner undefined."));
    p_lvset->attach_mesh_refiner(ptr_mesh_refiner);
  }
  else
    ptr_mesh_refiner->append(*p_lvset);

  // initialize the level set advect simulator,
  // but don't do mesh refinement just yet.
  ptrs_lvset_ic.push_back(p_lvset_ic);
  p_lvset->initialize(*p_lvset_ic, ptr_stokes, false);

  // add this material-interface pair into the material compositor
  ptr_materials->push(p_material, p_lvset);
}


template <typename AdvectType, typename ReinitType>
void LevelSetStokes<AdvectType, ReinitType>::finalize_material_domains()
{
  ASSERT(!material_domain_finalized,
         EXCEPT_MSG("Materials cannot be finalized twice."));
  ASSERT_SAME_SIZE(ptrs_level_set, ptrs_lvset_ic);

  MeshControl<value_type>& mesh_ctrl = *ptr_control->ptr_level_set->ptr_mesh;

  if (ptr_mesh_refiner) {
    for (int ilevel = ptr_mesh->get_info().min_level;
         ilevel <= mesh_ctrl.max_level;
         ++ilevel) {
      // flag cells close to the interface for refinement
      for (auto& p_lvset : ptrs_level_set) {
        auto sp_ls = p_lvset.lock();
        ASSERT(sp_ls != nullptr, ExcExpiredPointer());
        sp_ls->flag_mesh_for_coarsen_and_refine(mesh_ctrl, true);
      }

      // execute mesh coarsening and refinement for this level
      ptr_mesh_refiner->run_coarsen_and_refine(true);

      // interpolate initial values on new mesh
      auto p_ic = ptrs_lvset_ic.begin();
      for (auto& p_lvset : ptrs_level_set) {
        auto sp_ls = p_lvset.lock();
        ASSERT(sp_ls != nullptr, ExcExpiredPointer());
        sp_ls->discretize_function_to_solution(**p_ic++);
      }
    }  // ilevel-loop
  }

  // if needed, run reinit
  auto pp_lvset = ptrs_level_set.begin();
  for (const auto& p_ic : ptrs_lvset_ic) {
    if (!p_ic->exact) {
      auto sp_ls = (*pp_lvset).lock();
      ASSERT(sp_ls != nullptr, ExcExpiredPointer());

      sp_ls->reinit_simulator.initialize(sp_ls->get_solution());
      sp_ls->reinit_simulator.run_iteration();
    }
    ++pp_lvset;
  }

  ptrs_lvset_ic.clear();
  this->material_domain_finalized = true;
}


template <typename AdvectType, typename ReinitType>
void LevelSetStokes<AdvectType, ReinitType>::attach_mesh_refiner(
  const std::shared_ptr<MeshRefiner<dim, value_type>>& p_mesh_refiner,
  bool stokes_compute_refinement)
{
  ASSERT(p_mesh_refiner != nullptr, ExcNullPointer());
  ptr_mesh_refiner = p_mesh_refiner;
  stokes_compute_refinement ? ptr_stokes->attach_mesh_refiner(p_mesh_refiner)
                            : ptr_mesh_refiner->append(*ptr_stokes);
}


template <typename AdvectType, typename ReinitType>
void LevelSetStokes<AdvectType, ReinitType>::advance_time(
  time_step_type time_step)
{
  ASSERT(is_synchronized(), ExcNotSynchronized());
  if (numerics::is_zero(time_step)) return;  // nothing to do

  LOG_PREFIX("LevelSetStokes");

  // clear all user flags set on the mesh
  for (const auto& cell :
       this->ptr_stokes->get_dof_handler().active_cell_iterators())
    cell->clear_user_flag();

  // call StokesSimulator to get the velocity field.
  felspa_log << "Advancing time for StokesSimulator ..." << std::endl;
  ptr_stokes->advance_time(time_step);

  // evolve material compositor forward, force single cycle
  felspa_log << "Advancing time for LevelSet ..." << std::endl;
  ptr_materials->advance_time(time_step);

  // also evlove the particles...
  if (ptr_particles) ptr_particles->advance_time(time_step);

  this->phsx_time += time_step;

  // now make sure two major components are synchronized
  ASSERT(is_synchronized(), ExcNotSynchronized());

  felspa_log << "End of time step, simulation time = " << this->phsx_time
             << std::endl;
}


template <typename AdvectType, typename ReinitType>
auto LevelSetStokes<AdvectType, ReinitType>::advance_time(
  time_step_type time_step, bool compute_single_step) -> time_step_type
{
  ASSERT(is_synchronized(), ExcNotSynchronized());
  if (numerics::is_zero(time_step)) return 0.0;

  LOG_PREFIX("LevelSetStokes");

  time_step_type cumulative_time = 0.0;

  do {
    // clear all user flags set on the mesh
    for (const auto& cell :
         this->ptr_stokes->get_dof_handler().active_cell_iterators())
      cell->clear_user_flag();

    // call StokesSimulator to get the velocity field.
    felspa_log << "Updating velocity field from StokesSimulator ..."
               << std::endl;

    ptr_stokes->try_advance_time();

    // evolve material compositor forward, force single cycle
    felspa_log << "Updating level set ..." << std::endl;
    time_step_type time_substep = ptr_materials->advance_time(time_step, true);

    felspa_log << "Updating tracer particles ..." << std::endl;
    if (ptr_particles) ptr_particles->advance_time(time_substep);

    ptr_stokes->finalize_time_step(time_substep);

    this->phsx_time += time_substep;
    time_step -= time_substep;
    cumulative_time += time_substep;

    ASSERT(is_synchronized(), ExcNotSynchronized());

    felspa_log << "End of time step, simulation time = " << this->phsx_time
               << std::endl;

  } while (!numerics::is_zero(time_step) && !compute_single_step);

  return cumulative_time;
}


template <typename AdvectType, typename ReinitType>
void LevelSetStokes<AdvectType, ReinitType>::export_solution(
  const std::string& file_name_stem, ExportFileFormat format) const
{
  // Solution for Stokes system
  {
    std::stringstream stokes_ss;
    stokes_ss << file_name_stem << "Stokes" << '.' << format;
    ExportFile stokes_file(stokes_ss.str());
    // export the Stokes solution
    ptr_stokes->export_solution(stokes_file);
  }

  // Solution for level set
  {
    // export the solution from level set sovlers
    const auto n_level_sets = this->get_materials().n_materials();

    for (typename std::remove_const<decltype(n_level_sets)>::type ils = 1;
         ils < n_level_sets;
         ++ils) {
      const std::string& matname =
        this->get_materials()[ils].first.get_label_string();
      std::stringstream lvset_ss;
      lvset_ss << file_name_stem << matname << '.' << format;
      ExportFile lvset_file(lvset_ss.str());
      this->get_materials()[ils].second.export_solution(lvset_file);
    }
  }

  if (ptr_particles != nullptr) {
    std::stringstream ptcl_ss;
    ptcl_ss << file_name_stem << "Ptcls" << '.' << format;
    ExportFile ptcl_file(ptcl_ss.str());
    ptr_particles->export_solution(ptcl_file);
  }

  signals.post_export();
}


template <typename AdvectType, typename ReinitType>
void LevelSetStokes<AdvectType, ReinitType>::export_solutions(
  const std::string& path) const
{
  // Stokes System
  ptr_stokes->export_solutions(path);

  // Level set
  const auto n_level_sets = this->get_materials().n_materials();
  for (typename std::remove_const<decltype(n_level_sets)>::type ils = 1;
       ils < n_level_sets;
       ++ils)
    this->get_materials()[ils].second.export_solutions(path);

  // Particles
  if (ptr_particles != nullptr) ptr_particles->export_solutions(path);

  signals.post_export();
}


/* ************************************************** */
/*           LevelSetStokes::Control                  */
/* ************************************************** */

template <typename AdvectType, typename ReinitType>
void LevelSetStokes<AdvectType, ReinitType>::Control::
  set_refine_reinit_interval(unsigned int interval)
{
  ptr_level_set->set_refine_reinit_interval(interval);
  ptr_stokes->ptr_mesh->refinement_interval = interval;
}


template <typename AdvectType, typename ReinitType>
void LevelSetStokes<AdvectType, ReinitType>::Control::set_coarsen_refine_limit(
  int coarsen_limit, int refine_limit)
{
  ptr_level_set->ptr_mesh->set_coarsen_refine_limit(coarsen_limit,
                                                    refine_limit);
  ptr_stokes->ptr_mesh->set_coarsen_refine_limit(coarsen_limit, refine_limit);
}


/* ------------------- */
namespace internal
/* ------------------- */
{
  /* ************************************************** */
  /*            LevelSetStokes::Particles               */
  /* ************************************************** */
  template <int dim, typename NumberType, typename QuadratureType>
  template <typename Advect, typename Reinit>
  LevelSetStokesParticles<dim, NumberType, QuadratureType>::
    LevelSetStokesParticles(
      const LevelSetStokes<Advect, Reinit>& sim,
      unsigned int nptcls_per_dim_,
      bool automatic_replenish_particles_,
      const std::shared_ptr<ls::LevelSetBase<dim, value_type>>&
        ptr_ref_level_set_,
      bool generate_ptcls_near_interface,
      bool reinsert_periodic_particles)
    : ptr_mesh(&(*sim.ptr_mesh)),
      ptr_stokes(sim.ptr_stokes),
      nptcls_per_dim(nptcls_per_dim_),
      ptr_materials(sim.ptr_materials),
      particle_handler(sim.ptr_stokes->get_mesh(),
                       sim.ptr_stokes->get_mapping(), n_property_entries),
      label(sim.get_label_string() + "Ptcls"),
      counter(0),
      ptr_reference_level_set(ptr_ref_level_set_),
      property_pool_reinsert(n_property_entries),
      pvd_collector(label),
      automatic_replenish_particles(automatic_replenish_particles_)
      
  {
    using namespace dealii;

    // upon grid coarsening and refinement, store the particles and reload
#ifdef FELSPA_HAS_MPI
    ptr_mesh->signals.pre_distributed_refinement.connect(
      [&]() { particle_handler.register_store_callback_function(); });
    ptr_mesh->signals.post_distributed_refinement.connect(
      [&]() { particle_handler.register_load_callback_function(false); });
#else
    ptr_mesh->signals.pre_refinement.connect(
      [&]() { particle_handler.register_store_callback_function(); });
    ptr_mesh->signals.post_refinement.connect(
      [&]() { particle_handler.register_load_callback_function(false); });
#endif

    if (reinsert_periodic_particles)
      particle_handler.signals.particle_lost.connect(
        [this](const typename Particles::ParticleIterator<dim>& ptcl,
               const typename Triangulation<dim>::active_cell_iterator& cell) {
          this->collect_escaped_periodic_particles(ptcl, cell);
        });

    if (generate_ptcls_near_interface) {
      for (auto plvset : sim.ptrs_level_set) {
        auto sp_lvset = plvset.lock();
        ASSERT(sp_lvset != nullptr, ExcExpiredPointer());
        // FEValues<dim> feval(sp_lvset->get_mapping(), sp_lvset->get_fe(),
        //                     quadrature, update_flags);

        value_type threshold =
          sp_lvset->get_control().refinement_width_coeff *
          sp_lvset->get_mesh().get_info().min_diameter;

        auto interface_cells = sp_lvset->cells_near_interface(
          sp_lvset->get_dof_handler().active_cell_iterators(), threshold);

        generate_particles(interface_cells);
      }
    } else {
      // generate particles all over the whole domain
      auto pstokes = ptr_stokes.lock();
      ASSERT(pstokes != nullptr, ExcExpiredPointer());
      generate_particles(pstokes->get_dof_handler().active_cell_iterators());
    }

    particle_handler.update_cached_numbers();
  }


  template <int dim, typename NumberType, typename QuadratureType>
  template <typename Iterator>
  void
  LevelSetStokesParticles<dim, NumberType, QuadratureType>::generate_particles(
    const dealii::IteratorRange<Iterator>& cell_range)
  {
    // TODO: parallelize multithread
    for (const auto& cell : cell_range) generate_particles_in_cell(cell);
  }


  template <int dim, typename NumberType, typename QuadratureType>
  void LevelSetStokesParticles<dim, NumberType,
                               QuadratureType>::replenish_particles()
  {
    auto sp_stokes = ptr_stokes.lock();
    ASSERT(sp_stokes != nullptr, ExcExpiredPointer());

    for (const auto& cell :
         sp_stokes->get_dof_handler().active_cell_iterators()) {
      // no particles in this cell
      auto ptcl_range = particle_handler.particles_in_cell(cell);
      if (ptcl_range.begin() == ptcl_range.end())
        generate_particles_in_cell(cell);
    }

    particle_handler.update_cached_numbers();
    particle_handler.sort_particles_into_subdomains_and_cells();
  }


  template <int dim, typename NumberType, typename QuadratureType>
  template <typename CellIterator>
  void LevelSetStokesParticles<dim, NumberType, QuadratureType>::
    generate_particles_in_cell(const CellIterator& cell)
  {
    using namespace dealii;

    QuadratureType quadrature(nptcls_per_dim);

    // construct material accessor for this cell and reinit
    auto sp_material = ptr_materials.lock();
    ASSERT(sp_material != nullptr, ExcExpiredPointer());
    auto p_material_accessor = sp_material->generate_accessor(quadrature);
    p_material_accessor->reinit(cell);

    // build FEValues using the fe and mapping from Stokes solver
    const static UpdateFlags update_flags =
      update_values | update_quadrature_points;
    const auto sp_stokes = ptr_stokes.lock();
    ASSERT(sp_stokes != nullptr, ExcExpiredPointer());
    const DoFHandler<dim>& dofh = cell->get_dof_handler();
    FEValues<dim> fevals(sp_stokes->get_mapping(), dofh.get_fe(), quadrature,
                         update_flags);
    fevals.reinit(cell);

    // obtain material paramters on the particles
    std::vector<value_type> density_qpt(fevals.n_quadrature_points),
      viscosity_qpt(fevals.n_quadrature_points);
    PointsField<dim, value_type> pts_field(fevals.n_quadrature_points);
    auto qpts = fevals.get_quadrature_points();
    pts_field.ptr_pts = &qpts;
    p_material_accessor->eval_scalars(MaterialParameter::density, pts_field,
                                      density_qpt);
    p_material_accessor->eval_scalars(MaterialParameter::viscosity, pts_field,
                                      viscosity_qpt);
    auto it_density = density_qpt.begin();
    auto it_viscosity = viscosity_qpt.begin();

    // set particle locations
    for (const auto& qpt : qpts) {
      dealii::Particles::Particle<dim> new_ptcl;
      new_ptcl.set_location(qpt);
      new_ptcl.set_reference_location(
        fevals.get_mapping().transform_real_to_unit_cell(cell, qpt));
      new_ptcl.set_id(++counter);

      auto it_ptcl = particle_handler.insert_particle(new_ptcl, cell);

      // initialize the particle properties
      auto ptcl_property = it_ptcl->get_properties();
      std::fill(ptcl_property.begin(), ptcl_property.end(), 0.0);
      ptcl_property[spawn_time] = this->get_time();
      ptcl_property[viscosity] = *it_viscosity++;
      ptcl_property[density] = *it_density++;

      // the initial deformation gradient is an identity matrix
      for (int idim = 0; idim < dim; ++idim)
        for (int jdim = 0; jdim < dim; ++jdim)
          ptcl_property[deformation_gradient + idx(idim, jdim)] =
            static_cast<value_type>(idim == jdim);
      ptcl_property[flinn_slope] = 1.0;
      ptcl_property[Wk] = invalid_Wk;

      // write the property to ptcl
      it_ptcl->set_properties(ptcl_property);
    }  // qpt loop
  }


  template <int dim, typename NumberType, typename QuadratureType>
  void LevelSetStokesParticles<dim, NumberType, QuadratureType>::advance_time(
    time_step_type dt)
  {
    using dealii::Point;
    using dealii::Quadrature;
    using namespace dealii::Particles;

    const auto p_stokes = ptr_stokes.lock();

    ASSERT(p_stokes != nullptr, ExcExpiredPointer());
    ASSERT(dt >= 0.0, ExcArgumentCheckFail());
    LOG_PREFIX("LevelSetStokesParticles");

    auto cell_stokes = p_stokes->get_dof_handler().begin_active();
    for (const auto& cell_lvset :
         ptr_reference_level_set->active_cell_iterators()) {
      // get reference positions of the particles
      std::vector<Point<dim, value_type>> ptcl_ref_positions;
      const typename ParticleHandler<dim>::particle_iterator_range
        ptcls_in_cell = particle_handler.particles_in_cell(cell_lvset);
      std::transform(ptcls_in_cell.begin(), ptcls_in_cell.end(),
                     std::back_inserter(ptcl_ref_positions),
                     [](const ParticleAccessor<dim>& ptcl) {
                       return ptcl.get_reference_location();
                     });

      // construct a quadrature rule based on the particles
      const Quadrature<dim> ptcl_quad(ptcl_ref_positions);

      // update cell kinematics with Stokes simulator
      update_cell_kinematics(cell_stokes++, ptcl_quad, dt);

      // update interface geometry with reference level set simulator
      if (ptr_reference_level_set != nullptr)
        update_cell_level_set(cell_lvset, ptcl_quad);
    }

    size_type n_distorted = reset_stretched_particles();
    felspa_log << n_distorted << " stretched particles have been reset."
               << std::endl;

    this->phsx_time += dt;

    // Re-sort particles. This will implicitly collect the particles
    // that have escaped from periodic bdry into
    // ptcls_to_reinsert
    particle_handler.sort_particles_into_subdomains_and_cells();

    // now reinsert the particles
    particle_handler.insert_particles(ptcls_to_reinsert);

    // sort the particles again to account for
    // ptcls that escaped from periodic bdry and then reinserted
    if (!ptcls_to_reinsert.empty()) {
      particle_handler.sort_particles_into_subdomains_and_cells();
      ptcls_to_reinsert.clear();
    }

    // now we should not have any particles escaping from the periodic bc
    ASSERT(ptcls_to_reinsert.empty(), ExcInternalErr());

    // If some cells are devoid of particles, add them
    if (automatic_replenish_particles)
      replenish_particles();
  }


  template <int dim, typename NumberType, typename QuadratureType>
  void
  LevelSetStokesParticles<dim, NumberType, QuadratureType>::export_solution(
    ExportFile& file) const
  {
    using namespace dealii;
    namespace dci = dealii::DataComponentInterpretation;
    Particles::DataOut<dim, dim> ptcl_out;

#if FELSPA_DEAL_II_VERSION_GTE(9, 3, 0)
    std::vector<std::string> labels;
    std::vector<dci::DataComponentInterpretation> data_component_interpretation;

    labels.emplace_back("spawn_time");
    data_component_interpretation.push_back(dci::component_is_scalar);

    labels.emplace_back("density");
    data_component_interpretation.push_back(dci::component_is_scalar);

    labels.emplace_back("viscosity");
    data_component_interpretation.push_back(dci::component_is_scalar);

    labels.emplace_back("flinn_slope");
    data_component_interpretation.push_back(dci::component_is_scalar);

    labels.emplace_back("Wk");
    data_component_interpretation.push_back(dci::component_is_scalar);

    for (int idim = 0; idim < dim; ++idim) {
      labels.emplace_back("velocity");
      data_component_interpretation.push_back(dci::component_is_part_of_vector);
    }

    // velocity gradient tensor
    for (int idim = 0; idim < dim * dim; ++idim) {
      labels.emplace_back("velocity_gradient");
      data_component_interpretation.push_back(dci::component_is_part_of_tensor);
    }

    // deformation gradient tensor
    for (int idim = 0; idim < dim * dim; ++idim) {
      labels.emplace_back("deformation_gradient");
      data_component_interpretation.push_back(dci::component_is_part_of_tensor);
    }


    if (ptr_reference_level_set) {
      labels.emplace_back("level_set");
      data_component_interpretation.push_back(dci::component_is_scalar);

      for (int idim = 0; idim < dim; ++idim) {
        labels.emplace_back("level_set_normal");
        data_component_interpretation.push_back(
          dci::component_is_part_of_vector);
      }
    }

    ptcl_out.build_patches(this->particle_handler, labels,
                           data_component_interpretation);
#else
    // Multiple component export is only available after deal.II 9.3.0
    ptcl_out.build_patches(this->particle_handler);
#endif  // deal.II >= 9.3.0


    switch (file.get_format()) {
      case ExportFileFormat::vtk:
        ptcl_out.write_vtk(file.access_stream());
        break;
      case ExportFileFormat::vtu:
        ptcl_out.write_vtu(file.access_stream());
        break;
      default:
        THROW(ExcNotImplementedInFileFormat(file.get_file_extension()));
    }
  }


  template <int dim, typename NumberType, typename QuadratureType>
  void LevelSetStokesParticles<dim, NumberType, QuadratureType>::
    collect_escaped_periodic_particles(
      const typename dealii::Particles::ParticleIterator<dim>& ptcl,
      const typename dealii::Triangulation<dim>::active_cell_iterator& cell)
  {
    using namespace dealii;
    using TriaActiveIt =
      typename dealii::Triangulation<dim>::active_cell_iterator;
    using PointType = dealii::Point<dim, value_type>;

    PointType ptcl_xyz = ptcl->get_location();

    const auto pstokes = ptr_stokes.lock();
    ASSERT(pstokes != nullptr, ExcExpiredPointer());


    for (unsigned int face_no = 0;
         face_no < dealii::GeometryInfo<dim>::faces_per_cell;
         ++face_no) {
      // check if we have periodic boundary defined at this cell
      if (cell->has_periodic_neighbor(face_no)) {
        const unsigned int dimension = face_no / 2;
        const unsigned int neighbor_face_no =
          cell->periodic_neighbor_of_periodic_neighbor(face_no);

        TriaActiveIt neighbor = cell->periodic_neighbor(face_no);
        const auto& face = cell->face(face_no);
        const auto& neighbor_face = neighbor->face(neighbor_face_no);

        const dealii::Tensor<1, dim, value_type> face_center_to_ptcl =
          ptcl_xyz - face->center();

        FEFaceValues<dim> fe_face_values(
          pstokes->get_mapping(), pstokes->get_fe(), QMidpoint<dim - 1>(),
          update_normal_vectors);
        fe_face_values.reinit(cell, face_no);

        // Now we make sure the particle indeed
        // left through the periodic boundary
        if (scalar_product(face_center_to_ptcl,
                           fe_face_values.get_normal_vectors()[0]) > 0.0) {
          dealii::Point<dim, value_type> new_ptcl_xyz(ptcl_xyz);
          new_ptcl_xyz[dimension] -= face->center()[dimension];
          new_ptcl_xyz[dimension] += neighbor_face->center()[dimension];

          // create a new particle on the other side
          dealii::Particles::Particle<dim> new_ptcl;
          new_ptcl.set_reference_location(ptcl->get_reference_location());
          new_ptcl.set_id(ptcl->get_id());
          new_ptcl.set_location(new_ptcl_xyz);
          new_ptcl.set_property_pool(property_pool_reinsert);
          new_ptcl.set_properties(ptcl->get_properties());

          ptcls_to_reinsert.insert(std::make_pair(neighbor, new_ptcl));
        }
      }
    }
  }


  template <int dim, typename NumberType, typename QuadratureType>
  void
  LevelSetStokesParticles<dim, NumberType, QuadratureType>::export_solutions(
    const std::string& path) const
  {
    using dealii::Utilities::int_to_string;

    pvd_collector.set_file_path(path);
    auto counter = pvd_collector.get_file_count() + 1;
    std::string master_file_name =
      this->get_label_string() + '_' +
      int_to_string(counter, constants::max_export_numeric_digits);
    std::string vtu_file_name = path + master_file_name + ".vtu";

    // Export this time step
    ExportFile vtu_file(vtu_file_name);
    export_solution(vtu_file);

    // append the record to the collector.
    pvd_collector.append_record(this->get_time(), master_file_name + ".vtu");
  }


  template <int dim, typename NumberType, typename QuadratureType>
  void LevelSetStokesParticles<dim, NumberType, QuadratureType>::
    update_cell_level_set(const cell_iterator_type& cell,
                          const dealii::Quadrature<dim>& ptcl_quad)
  {
    using dealii::FEValues;
    using dealii::Tensor;
    using namespace dealii::Particles;

    if (particle_handler.n_particles_in_cell(cell) == 0) return;

    // construct FEvalues
    std::unique_ptr<FEValues<dim>> p_fevals(ptr_reference_level_set->fe_values(
      ptcl_quad, dealii::update_values | dealii::update_gradients));
    p_fevals->reinit(cell);

    // get the particle iterators
    auto ptcl_range = particle_handler.particles_in_cell(cell);

    {
      // get level set values
      std::vector<value_type> ptcl_lvset(ptcl_quad.size());
      ptr_reference_level_set->extract_level_set_values(*p_fevals, ptcl_lvset);
      auto itr_ptcl_lvset = ptcl_lvset.cbegin();
      for (auto& p : ptcl_range)
        p.get_properties()[level_set] = *itr_ptcl_lvset++;
    }

    {
      // get level set gradients
      std::vector<Tensor<1, dim, value_type>> ptcl_lvset_grads(
        ptcl_quad.size());
      ptr_reference_level_set->extract_level_set_gradients(*p_fevals,
                                                           ptcl_lvset_grads);
      auto itr_ptcl_lvset_grads = ptcl_lvset_grads.begin();
      for (auto& p : ptcl_range) {
        value_type* prop = &p.get_properties()[level_set_normal];
        *itr_ptcl_lvset_grads /= itr_ptcl_lvset_grads->norm();
        for (int idim = 0; idim < dim; ++idim)
          prop[idim] = (*itr_ptcl_lvset_grads)[idim];
      }
      ++itr_ptcl_lvset_grads;
    }
  }


  template <int dim, typename NumberType, typename QuadratureType>
  void LevelSetStokesParticles<dim, NumberType, QuadratureType>::
    update_cell_kinematics(const cell_iterator_type& cell,
                           const dealii::Quadrature<dim>& ptcl_quad,
                           time_step_type dt)
  {
    using namespace dealii;
    using namespace dealii::Particles;
    const auto p_stokes = ptr_stokes.lock();
    ASSERT(p_stokes != nullptr, ExcExpiredPointer());

    if (particle_handler.n_particles_in_cell(cell) == 0) return;

    FEValues<dim> fevals(p_stokes->get_mapping(), p_stokes->get_fe(), ptcl_quad,
                         update_values | update_gradients);
    fevals.reinit(cell);

    // get velocities and velocity gradient tensor
    const dealii::FEValuesExtractors::Vector velo(0);
    std::vector<Tensor<1, dim, value_type>> ptcl_velocities(ptcl_quad.size());
    std::vector<Tensor<2, dim, value_type>> ptcl_velocity_gradients(
      ptcl_quad.size());
    fevals[velo].get_function_values(p_stokes->get_solution_vector(),
                                     ptcl_velocities);
    fevals[velo].get_function_gradients(p_stokes->get_solution_vector(),
                                        ptcl_velocity_gradients);

    auto it_v = ptcl_velocities.cbegin();
    auto it_v_grad = ptcl_velocity_gradients.cbegin();
    auto ptcls_in_cell = particle_handler.particles_in_cell(cell);

    for (auto it_ptcl = ptcls_in_cell.begin(); it_ptcl != ptcls_in_cell.end();
         ++it_ptcl) {
      auto prop = it_ptcl->get_properties();

      // load velocity
      for (int idim = 0; idim < dim; ++idim)
        prop[velocity + idim] = (*it_v)[idim];

      // move the particle position
      const Point<dim> new_location = it_ptcl->get_location() + dt * (*it_v);
      it_ptcl->set_location(new_location);

      // update the kinematic vorticity
      dealii::Tensor<2, dim, value_type> L_old;
      const value_type* it_L_in_property = &prop[velocity_gradient];
      for (auto it = L_old.begin_raw(); it != L_old.end_raw(); ++it)
        *it = *it_L_in_property++;
      prop[Wk] = objective_kinematic_vorticity(L_old, *it_v_grad, dt);

      // store the velocity gradient tensor
      std::copy(it_v_grad->begin_raw(), it_v_grad->end_raw(),
                &prop[velocity_gradient]);

      // update deformation gradient tensor
      integrate_deformation_gradient(&prop[deformation_gradient], *it_v_grad,
                                     dt);

      // move the iterator to the next particle
      ++it_v;
      ++it_v_grad;
    }
  }


  template <int dim, typename NumberType, typename QuadratureType>
  auto LevelSetStokesParticles<dim, NumberType, QuadratureType>::
    objective_kinematic_vorticity(const dealii::Tensor<2, dim, value_type>& L0,
                                  const dealii::Tensor<2, dim, value_type>& L1,
                                  value_type dt) -> value_type
  {
    ASSERT(dt > 0.0, ExcArgumentCheckFail());
    using dealii::ArrayView;
    using dealii::SymmetricTensor;
    using dealii::Tensor;
    using numerics::is_zero;

    SymmetricTensor<2, dim, value_type> D1(0.5 * (L1 + transpose(L1)));

    if (is_zero(L0.norm()) || is_zero(D1.norm())) return invalid_Wk;

    Tensor<2, dim, value_type> H1;
    Tensor<2, dim, value_type> H0t;

    SymmetricTensor<2, dim, value_type> D0(0.5 * (L0 + transpose(L0)));
    auto eigpair0 = eigenvectors(D0);
    auto eigpair1 = eigenvectors(D1);

    // conversion matrix defined by H1 * H0^T
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j) H1[j][i] = eigpair1[i].second[j];
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j) H0t[i][j] = eigpair0[i].second[j];

    // Omega = (H1 * H0t - I)/dt, 1st order approx
    Tensor<2, dim, value_type> Theta = H1 * H0t;
    Tensor<2, dim, value_type> Omega = (Theta - transpose(Theta)) / 2.0 / dt;

    // objective vorticity = vorticity - Omega
    Tensor<2, dim, value_type> W = 0.5 * (L1 - transpose(L1));
    Tensor<2, dim, value_type> W_obj = W - Omega;

    return std::min(W_obj.norm(), W.norm()) / D1.norm();
  }


  template <int dim, typename NumberType, typename QuadratureType>
  void LevelSetStokesParticles<dim, NumberType, QuadratureType>::
    principle_finite_strain_axis(value_type* max_stretch,
                                 value_type* max_compression,
                                 const value_type* F) const
  {
    dealii::SymmetricTensor<2, dim, value_type> FFt;
    for (int i = 0; i < dim; ++i)
      for (int j = i; j < dim; ++j)
        for (int k = 0; k < dim; ++k) FFt[i][j] += F[idx(i, k)] * F[idx(j, k)];

    auto eigen_pair = eigenvectors(FFt);

    ASSERT(eigen_pair[0].first > eigen_pair[dim].first, ExcInternalErr());
    ASSERT(numerics::is_nearly_equal(eigen_pair[0].second.norm(), 1.0),
           ExcUnexpectedValue<double>(eigen_pair[0].second.norm()));
    ASSERT(numerics::is_nearly_equal(eigen_pair[dim - 1].second.norm(), 1.0),
           ExcUnexpectedValue<double>(eigen_pair[dim - 1].second.norm()));

    // scale the maximum stretch/compression
    std::for_each(
      eigen_pair[0].second.begin_raw(),
      eigen_pair[0].second.end_raw(),
      [&](value_type& val) { val *= std::sqrt(eigen_pair[0].first); });
    std::for_each(
      eigen_pair[dim - 1].second.begin_raw(),
      eigen_pair[dim - 1].second.end_raw(),
      [&](value_type& val) { val *= std::sqrt(eigen_pair[dim - 1].first); });

    std::copy(eigen_pair[0].second.begin_raw(), eigen_pair[0].second.end_raw(),
              max_stretch);
    std::copy(eigen_pair[dim - 1].second.begin_raw(),
              eigen_pair[dim - 1].second.end_raw(),
              max_compression);
  }


  template <int dim, typename NumberType, typename QuadratureType>
  void LevelSetStokesParticles<dim, NumberType, QuadratureType>::
    integrate_deformation_gradient(value_type* F,
                                   const dealii::Tensor<2, dim, value_type>& l,
                                   time_step_type dt) const
  {
    value_type temp[dim * dim];
    std::fill(std::begin(temp), std::end(temp), 0.0);

    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
        for (int k = 0; k < dim; ++k)
          temp[idx(i, j)] +=
            (dt * l[i][k] + static_cast<value_type>(i == k)) * F[idx(k, j)];

    std::copy(std::begin(temp), std::end(temp), F);
  }


  template <int dim, typename NumberType, typename QuadratureType>
  auto LevelSetStokesParticles<dim, NumberType, QuadratureType>::
    reset_stretched_particles(NumberType threshold_ratio) const -> size_type
  {
    // this only makes sense for 3-dimension objects
    if constexpr (dim == 3) {
      size_type counter = 0;

      for (auto ptcl_itr = particle_handler.begin();
           ptcl_itr != particle_handler.end();
           ++ptcl_itr) {
        // properties
        value_type* F = &ptcl_itr->get_properties()[deformation_gradient];
        value_type& ptcl_spawn_time = ptcl_itr->get_properties()[spawn_time];
        value_type& ptcl_flinn_slope = ptcl_itr->get_properties()[flinn_slope];

        // left Cauchy-Green Tensor
        dealii::SymmetricTensor<2, dim, NumberType> FFt;
        for (int i = 0; i < dim; ++i)
          for (int j = i; j < dim; ++j)
            for (int k = 0; k < dim; ++k)
              FFt[i][j] += F[idx(i, k)] * F[idx(j, k)];

        std::array<NumberType, dim> eigvals = eigenvalues(FFt);

        ASSERT(eigvals[dim - 1] > 0.0,
               ExcUnexpectedValue<NumberType>(eigvals[dim - 1]));

        value_type relative_stretching = std::sqrt(eigvals[0] / eigvals[2]);

        if (relative_stretching > threshold_ratio) {
          // set the deformation gradient to identity
          for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j)
              F[idx(i, j)] = static_cast<value_type>(i == j);
          // set spawn time to current time
          ptcl_spawn_time = this->phsx_time;
          // reset flinn slope to unity
          ptcl_flinn_slope = 1.0;
          // increment counter
          ++counter;
        } else {
          // update the flinn slope
          ptcl_flinn_slope =
            std::sqrt(eigvals[0] * eigvals[2] / eigvals[1] / eigvals[1]);
        }
      }

      return counter;
    }
    return 0;
  }
}  // namespace internal

FELSPA_NAMESPACE_CLOSE
#endif  // _FELPA_COUPLED_LEVEL_SET_STOKES_IMPLEMENT_H_ //
