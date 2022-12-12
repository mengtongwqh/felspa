#ifndef _FELSPA_LEVEL_SET_LEVEL_SET_IMPLEMENT_H_
#define _FELSPA_LEVEL_SET_LEVEL_SET_IMPLEMENT_H_

#include <deal.II/base/quadrature_lib.h>
#include <felspa/level_set/level_set.h>

#include "felspa/base/felspa_config.h"
#include "felspa/base/src/exception_classes.h"

FELSPA_NAMESPACE_OPEN

/* -------------- */
namespace ls
/* -------------- */
{
  /* ************************************************** */
  /**                 LevelSetBase                      */
  /* ************************************************** */
  template <int dim, typename NumberType>
  FELSPA_FORCE_INLINE dealii::FEValues<dim>*
  LevelSetBase<dim, NumberType>::fe_values(
    const dealii::Quadrature<dim>& quad, dealii::UpdateFlags update_flags) const
  {
    auto pb = dynamic_cast<const SimulatorBase<dim, NumberType>*>(this);
    ASSERT(pb != nullptr, ExcNullPointer());
    return pb->fe_values(quad, update_flags);
  }


  template <int dim, typename NumberType>
  FELSPA_FORCE_INLINE auto
  LevelSetBase<dim, NumberType>::active_cell_iterators() const
  {
    auto pb = dynamic_cast<const SimulatorBase<dim, NumberType>*>(this);
    ASSERT(pb != nullptr, ExcNullPointer());
    return pb->get_dof_handler().active_cell_iterators();
  }


  /* ************************************************** */
  /**               LevelSetControl                     */
  /* ************************************************** */

  template <typename AdvectType, typename ReinitType>
  LevelSetControl<AdvectType, ReinitType>::LevelSetControl()
    : ptr_reinit(std::make_shared<reinit_control_t>())
  {
    ptr_reinit->ptr_mesh = this->ptr_mesh;
    set_refine_reinit_interval(10);
  }


  template <typename AdvectType, typename ReinitType>
  void LevelSetControl<AdvectType, ReinitType>::set_artificial_viscosity(
    value_type viscosity)
  {
    ASSERT(viscosity >= 0.0, ExcArgumentCheckFail());
    ptr_reinit->ptr_ldg->ptr_assembler->viscosity = viscosity;
  }

  template <typename AdvectType, typename ReinitType>
  void LevelSetControl<AdvectType, ReinitType>::set_refine_reinit_interval(
    unsigned int interval)
  {
    this->ptr_mesh->refinement_interval = interval;
    reinit_frequency = interval;
  }


  /* ************************************************** */
  /**               LevelSetSurface                     */
  /* ************************************************** */

  template <typename Advect, typename Reinit>
  template <typename Iterator>
  dealii::IteratorRange<dealii::FilteredIterator<Iterator>>
  LevelSetSurface<Advect, Reinit>::cells_near_interface(
    dealii::IteratorRange<Iterator> it_range, value_type threshold) const
  {
    InterfaceCellFilter predicate(*this);
    predicate.set_threshold(threshold);
    return filter_iterators(it_range, predicate);
  }


  template <typename Advect, typename Reinit>
  LevelSetSurface<Advect, Reinit>::LevelSetSurface(Mesh<dim>& triag,
                                                   unsigned int pdegree,
                                                   const std::string& label)
    : Advect(triag, pdegree, label),
      reinit_simulator(triag, pdegree, label + "Reinit"),
      n_steps_without_reinit(0),
      ptr_control(std::make_shared<control_type>())
  {
    check_consistency();
    Advect::attach_control(ptr_control);
    reinit_simulator.attach_control(ptr_control->ptr_reinit);
  }


  template <typename Advect, typename Reinit>
  void LevelSetSurface<Advect, Reinit>::attach_control(
    const std::shared_ptr<control_type>& p_control)
  {
    ptr_control = p_control;
    Advect::attach_control(ptr_control);
    reinit_simulator.attach_control(ptr_control->ptr_reinit);
  }

  template <typename Advect, typename Reinit>
  auto LevelSetSurface<Advect, Reinit>::get_control() const
    -> const control_type&
  {
    ASSERT(ptr_control != nullptr, ExcNullPointer());
    return *ptr_control;
  }


  template <typename Advect, typename Reinit>
  template <typename VeloFcnType>
  void LevelSetSurface<Advect, Reinit>::initialize(
    const ICBase<dim, value_type>& initial_condition,
    const std::shared_ptr<VeloFcnType>& p_vfield, bool refine_mesh)
  {
    n_steps_without_reinit = 0;
    // force mesh refinement
    Advect::initialize(initial_condition, p_vfield, nullptr, refine_mesh,
                       false);

    // if the initial condition is not exact, run reinit
    if (!initial_condition.exact) {
      reinit_simulator.initialize(this->get_solution());
      if (!this->ptr_control->ptr_reinit->additional_control.global_solve) {
        this->ptr_control->ptr_reinit->additional_control.global_solve = true;
        reinit_simulator.run_iteration();
        this->ptr_control->ptr_reinit->additional_control.global_solve = false;

      } else {
        reinit_simulator.run_iteration();
      }
    }
  }


  template <typename Advect, typename Reinit>
  template <typename QuadratureType>
  void LevelSetSurface<Advect, Reinit>::set_quadrature(
    const QuadratureType& quad)
  {
    Advect::set_quadrature(quad);
    reinit_simulator.set_quadrature(quad);
  }


  template <typename Advect, typename Reinit>
  void LevelSetSurface<Advect, Reinit>::advance_time(time_step_type time_step)
  {
    // Simply forward the time step. No questions asked.
    Advect::advance_time(time_step);

    ++n_steps_without_reinit;

    if (ptr_control->execute_reinit &&
        n_steps_without_reinit == ptr_control->reinit_frequency) {
      reinit_simulator.initialize(this->get_solution());
      if (unsigned int n_iter = reinit_simulator.run_iteration()) {
        n_steps_without_reinit = 0;
        felspa_log << "Level set reinited after " << n_iter << " iterations."
                   << std::endl;
      } else {
        THROW(EXCEPT_MSG("Reinit iteration is never run."));
      }
    } else {
      felspa_log << "Level set reinit idling: " << n_steps_without_reinit << '/'
                 << ptr_control->reinit_frequency << std::endl;
    }
  }


  template <typename Advect, typename Reinit>
  auto LevelSetSurface<Advect, Reinit>::advance_time(time_step_type time_step,
                                                     bool compute_single_step)
    -> time_step_type
  {
    time_step_type cumulative_time = 0.0;

    do {
      time_step_type time_substep = Advect::advance_time(time_step, true);

      ++n_steps_without_reinit;

      if (ptr_control->execute_reinit &&
          n_steps_without_reinit == ptr_control->reinit_frequency) {
        reinit_simulator.initialize(this->get_solution());
        if (reinit_simulator.run_iteration()) n_steps_without_reinit = 0;
      }

      cumulative_time += time_substep;
      time_step -= time_substep;
      ASSERT(time_step >= 0.0, ExcInternalErr());

    } while (!numerics::is_zero(time_step) && !compute_single_step);

    return cumulative_time;
  }


  template <typename Advect, typename Reinit>
  bool LevelSetSurface<Advect, Reinit>::is_synchronized() const
  {
    return Advect::is_synchronized();
  }


  template <typename Advect, typename Reinit>
  auto LevelSetSurface<Advect, Reinit>::compute_mass_error(
    const ScalarFunction<dim, value_type>& analytical_soln,
    const dealii::Quadrature<dim>& quadrature,
    bool compute_relative_error) const -> value_type
  {
    using namespace dealii;

    value_type mass_analytical = 0.0;
    value_type mass_numerical = 0.0;

    FEValues<dim> fe(
      this->get_fe(),
      quadrature,
      update_values | update_quadrature_points | update_JxW_values);
    using nqpt_type =
      typename std::remove_const<decltype(fe.n_quadrature_points)>::type;

    std::vector<value_type> lvset_at_qpt(fe.n_quadrature_points);

    for (const auto& cell : this->get_dof_handler().active_cell_iterators()) {
      fe.reinit(cell);

      const std::vector<Point<dim>>& qpts = fe.get_quadrature_points();
      fe.get_function_values(this->get_solution_vector(), lvset_at_qpt);

      for (nqpt_type iqpt = 0; iqpt < fe.n_quadrature_points; ++iqpt) {
        mass_analytical +=
          domain_identity(analytical_soln(qpts[iqpt])) * fe.JxW(iqpt);
        mass_numerical += domain_identity(lvset_at_qpt[iqpt]) * fe.JxW(iqpt);
      }
    }  // cell-loop

#ifdef DEBUG
    LOG_PREFIX("LevelSetSurface");
    felspa_log << "Numerical Mass / Analytical Mass =  " << mass_numerical
               << " / " << mass_analytical << std::endl;
#endif

    value_type error = mass_numerical - mass_analytical;
    return compute_relative_error ? error / mass_analytical : error;
  }


  template <typename Advect, typename Reinit>
  auto LevelSetSurface<Advect, Reinit>::compute_mass_error(
    const ScalarFunction<dim, value_type>& analytical_soln,
    bool compute_relative_error) const -> value_type
  {
    return compute_mass_error(analytical_soln,
                              dealii::QGauss<dim>(this->fe_degree() + 1),
                              compute_relative_error);
  }


  template <typename Advect, typename Reinit>
  auto LevelSetSurface<Advect, Reinit>::integrate_mass(
    const dealii::Quadrature<dim>& quadrature) const -> value_type
  {
    using namespace dealii;

    value_type mass = 0.0;

    FEValues<dim> fe(
      this->get_fe(), quadrature,
      update_values | update_quadrature_points | update_JxW_values);

    using nqpt_type =
      typename std::remove_const<decltype(fe.n_quadrature_points)>::type;

    std::vector<value_type> lvset_at_qpt(fe.n_quadrature_points);

    for (const auto& cell : this->get_dof_handler().active_cell_iterators()) {
      fe.reinit(cell);
      fe.get_function_values(this->get_solution_vector(), lvset_at_qpt);
      for (nqpt_type iqpt = 0; iqpt < fe.n_quadrature_points; ++iqpt)
        mass += domain_identity(lvset_at_qpt[iqpt]) * fe.JxW(iqpt);
    }  // cell-loop

    return mass;
  }

  template <typename Advect, typename Reinit>
  auto LevelSetSurface<Advect, Reinit>::estimate_curvature() const
    -> const vector_type&
  {
    // allocate the curvature estimator
    if (ptr_curvature_estimator == nullptr)
      ptr_curvature_estimator = std::make_unique<CurvatureEstimator>(*this);
    // run estimation
    return ptr_curvature_estimator->estimate();
  }


  template <typename Advect, typename Reinit>
  void LevelSetSurface<Advect, Reinit>::export_solution(ExportFile& file) const
  {
#ifdef DEBUG
    LOG_PREFIX("LevelSetSurface");
    felspa_log << "Writing solution to file " << file.get_file_name()
               << std::endl;
#endif  // DEBUG //

    // export the level set values
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(this->get_dof_handler());
    data_out.add_data_vector(*this->solution, this->get_label_string());

    // export curvature, if computed
    if (ptr_curvature_estimator != nullptr &&
        this->is_synced_with(ptr_curvature_estimator->get_curvature()))
      data_out.add_data_vector(
        this->ptr_curvature_estimator->get_curvature_vector(),
        this->get_label_string() + "Curvature");

    data_out.build_patches();

    switch (file.get_format()) {
      case ExportFileFormat::vtk:
        data_out.write_vtk(file.access_stream());
        break;
      case ExportFileFormat::vtu:
        data_out.write_vtu(file.access_stream());
        break;
      default:
        THROW(ExcNotImplementedInFileFormat(file.get_file_extension()));
    }
  }


  template <typename Advect, typename Reinit>
  void LevelSetSurface<Advect, Reinit>::check_consistency() const
  {
    ASSERT(Advect::spacedim == Reinit::spacedim,
           EXCEPT_MSG("Spatial dimension of advect/reinit solver must agree"));
    static_assert(std::is_same<typename Advect::vector_type,
                               typename Reinit::vector_type>::value,
                  "vector_type of Advect and Reinit Simulator must agree.");
  }


  template <typename Advect, typename Reinit>
  FELSPA_FORCE_INLINE auto LevelSetSurface<Advect, Reinit>::domain_identity(
    value_type x) const -> value_type
  {
    value_type eps = 1.5 * this->mesh().get_info().min_diameter;

    switch (ptr_control->heaviside_smoothing) {
      case HeavisideSmoothing::none: {
        HeavisideFunction<HeavisideSmoothing::none, value_type> heaviside_fcn;
        return heaviside_fcn(-x);
      }
      case HeavisideSmoothing::linear: {
        HeavisideFunction<HeavisideSmoothing::linear, value_type> heaviside_fcn(
          eps);
        return heaviside_fcn(-x);
      }
      case HeavisideSmoothing::sine: {
        HeavisideFunction<HeavisideSmoothing::sine, value_type> heaviside_fcn(
          eps);
        return heaviside_fcn(-x);
      }
      default:
        THROW(ExcNotImplemented());
    }
  }


  template <typename Advect, typename Reinit>
  void LevelSetSurface<Advect, Reinit>::append_boundary_condition(
    const std::weak_ptr<BCFunction<dim, value_type>>& pbc)
  {
    auto sp_bc = pbc.lock();
    ASSERT(sp_bc != nullptr, ExcExpiredPointer());
    // this->append_boundary_condition(sp_bc);
    advect_solver_type::append_boundary_condition(sp_bc);
    reinit_simulator.append_boundary_condition(sp_bc);
  }


  template <typename Advect, typename Reinit>
  void LevelSetSurface<Advect, Reinit>::extract_level_set_values(
    const dealii::FEValuesBase<dim>& feval,
    std::vector<value_type>& lvset_values) const
  {
    feval.get_function_values(this->get_solution_vector(), lvset_values);
  }


  template <typename Advect, typename Reinit>
  void LevelSetSurface<Advect, Reinit>::extract_level_set_gradients(
    const dealii::FEValuesBase<dim>& feval,
    std::vector<dealii::Tensor<1, dim, value_type>>& lvset_grads) const
  {
    feval.get_function_gradients(this->get_solution_vector(), lvset_grads);
  }


  template <typename Advect, typename Reinit>
  void LevelSetSurface<Advect, Reinit>::do_flag_mesh_for_coarsen_and_refine(
    const MeshControl<value_type>& mesh_control) const
  {
    LOG_PREFIX("LevelSetSurface");
    felspa_log << "Labelling mesh for carsening and refinement..." << std::endl;


    auto threshold = this->ptr_control->refinement_width_coeff *
                     this->mesh().get_info().min_diameter;
    const auto max_level = mesh_control.max_level;
    const auto min_level = mesh_control.min_level;

    auto dofs_per_cell = this->get_fe().dofs_per_cell;
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);

    dealii::QTrapezoid<dim> quadrature;
    dealii::FEValues<dim> fe_values(this->get_mapping(), this->get_fe(),
                                    quadrature, dealii::update_values);

    // *** flagging for refinement/coarsening *** //
    // Refine the cell if the any support point has level set value
    // below the threshold or has a sign flip.
    // If none of the support points are below threshold, mark for coarsening
    for (const auto& cell : this->dof_handler().active_cell_iterators()) {
      // indices of local dof
      cell->get_dof_indices(local_dof_indices);
      fe_values.reinit(cell);

      std::vector<value_type> lvset_at_vertices(quadrature.size());

      fe_values.get_function_values(this->get_solution_vector(),
                                    lvset_at_vertices);

      bool is_interface_cell = false;
      for (auto lvset : lvset_at_vertices) {
        if (std::abs(lvset) < threshold ||
            lvset_at_vertices[0] * lvset <= 0.0) {
          is_interface_cell = true;
          cell->set_user_flag();
          if (cell->level() < max_level) {
            if (cell->coarsen_flag_set()) cell->clear_coarsen_flag();
            cell->set_refine_flag();
          }
          break;
        }
      }

      if (cell->user_flag_set() && cell->coarsen_flag_set())
        cell->clear_coarsen_flag();

      if (!is_interface_cell && cell->level() > min_level &&
          !cell->refine_flag_set() && !cell->user_flag_set())
        cell->set_coarsen_flag();
    }  // cell-loop


    // walk all cells again to make sure the max/min levels are respected
    for (const auto& cell : this->dof_handler().active_cell_iterators()) {
      if (cell->level() == max_level && cell->refine_flag_set())
        cell->clear_refine_flag();
      if (cell->level() == min_level && cell->coarsen_flag_set())
        cell->clear_coarsen_flag();
    }
  }

  /* --------------------------------------- */
  /*  LevelSetSurface :: InterfaceCellFilter */
  /* --------------------------------------- */
  template <typename Advect, typename Reinit>
  FELSPA_FORCE_INLINE
  LevelSetSurface<Advect, Reinit>::InterfaceCellFilter::InterfaceCellFilter(
    const LevelSetSurface<Advect, Reinit>& level_set_sim)
    : ptr_fevals(std::make_shared<dealii::FEValues<dim>>(
        level_set_sim.get_mapping(), level_set_sim.get_fe(),
        dealii::QTrapez<dim>(), dealii::update_values)),
      ptr_soln_vector(&level_set_sim.get_solution_vector())
  {}


  template <typename Advect, typename Reinit>
  FELSPA_FORCE_INLINE void
  LevelSetSurface<Advect, Reinit>::InterfaceCellFilter::set_threshold(
    value_type thres)
  {
    ASSERT(threshold > 0.0, ExcArgumentCheckFail());
    threshold = thres;
  }


  template <typename Advect, typename Reinit>
  template <typename Iterator>
  FELSPA_FORCE_INLINE bool
  LevelSetSurface<Advect, Reinit>::InterfaceCellFilter::operator()(
    const Iterator& it) const
  {
    ptr_fevals->reinit(it);
    std::vector<value_type> lvset_values_cell(dealii::QTrapez<dim>().size());
    ptr_fevals->get_function_values(*ptr_soln_vector, lvset_values_cell);

    for (const auto val : lvset_values_cell)
      if (std::abs(val) <= threshold) return true;
    return false;
  }


  /* --------------------------------------- */
  /*  LevelSetSurface ::  DipAngleCellFilter */
  /* --------------------------------------- */
  template <typename Advect, typename Reinit>
  LevelSetSurface<Advect, Reinit>::DipAngleCellFilter::DipAngleCellFilter(
    const LevelSetSurface<Advect, Reinit>& level_set_sim)
    : ptr_fevals(std::make_unique<dealii::FEValues<dim>>(
        level_set_sim.get_mapping(), level_set_sim.get_fe(),
        dealii::QMidpoint<dim>(), dealii::update_gradients)),
      ptr_soln_vector(&level_set_sim.get_solution_vector())
  {}


  template <typename Advect, typename Reinit>
  FELSPA_FORCE_INLINE void
  LevelSetSurface<Advect, Reinit>::DipAngleCellFilter::set_threshold(
    value_type lower_bound_, value_type upper_bound_,
    bool convert_degree_to_radian)
  {
    ASSERT(lower_bound_ < upper_bound_, ExcArgumentCheckFail());
    using constants::PI;

    if (convert_degree_to_radian) {
      ASSERT(lower_bound_ >= 0.0 && lower_bound_ < 90.0,
             ExcUnexpectedValue<value_type>(lower_bound_));
      ASSERT(upper_bound_ > 0.0 && upper_bound <= 90.0,
             ExcUnexpectedValue<value_type>(upper_bound_));

      lower_bound = lower_bound_ / 180.0 * PI;
      upper_bound = upper_bound_ / 180.0 * PI;

    } else {
      ASSERT(lower_bound_ >= 0.0 && lower_bound_ < 0.5 * PI,
             ExcUnexpectedValue<value_type>(lower_bound_));
      ASSERT(upper_bound_ > 0.0 && upper_bound <= 0.5 * PI,
             ExcUnexpectedValue<value_type>(upper_bound_));
      lower_bound = lower_bound_;
      upper_bound = upper_bound_;
    }
  }


  template <typename Advect, typename Reinit>
  template <typename Iterator>
  FELSPA_FORCE_INLINE bool
  LevelSetSurface<Advect, Reinit>::DipAngleCellFilter::operator()(
    const Iterator& it) const
  {
    ASSERT(
      lower_bound > 0.0 && upper_bound > 0.0,
      EXCEPT_MSG("The threshold of upper and lower limit has not been set."));
    ptr_fevals->reinit(it);
    std::vector<dealii::Tensor<1, dim, value_type>> cell_gradient_vec(1);
    // obtain the normal to the level set
    ptr_fevals->get_function_gradients(*ptr_soln_vector, cell_gradient_vec);
    dealii::Tensor<1, dim, value_type> cell_gradient =
      cell_gradient_vec[0] / cell_gradient_vec[0].norm();

    if (std::abs(cell_gradient[dim]) <= std::cos(lower_bound) &&
        std::abs(cell_gradient[dim] >= std::cos(upper_bound)))
      return true;
    else
      return false;
  }


  /* -------------------------------------- */
  /*  LevelSetSurface :: CurvatureEstimator */
  /* -------------------------------------- */
  template <typename Advect, typename Reinit>
  LevelSetSurface<Advect, Reinit>::CurvatureEstimator::CurvatureEstimator(
    const LevelSetSurface<Advect, Reinit>& lvset)
    : ptr_dof_handler(&lvset.get_dof_handler()),
      ptr_lvset_soln(&lvset.get_solution()),
      ptr_linear_system(&lvset.get_linear_system()),
      curvature(std::make_shared<vector_type>())
  {}


  template <typename Advect, typename Reinit>
  auto LevelSetSurface<Advect, Reinit>::CurvatureEstimator::estimate()
    -> const vector_type&
  {
    // allocate space for curvature solution and the RHS
    curvature->reinit((*ptr_lvset_soln)->size());
    rhs.reinit((*ptr_lvset_soln)->size());
    curvature.set_time(ptr_lvset_soln->get_time());

    // assemble the RHS
    dealii::QGauss<dim> qgauss(ptr_dof_handler->get_fe().degree + 1);
    assemble_rhs(qgauss);

    // solve the linear system
    dealii::SolverControl solver_control;
    this->ptr_linear_system->solve(*curvature, rhs, solver_control);

    std::for_each(curvature->begin(), curvature->end(),
                  [](value_type& x) { x = std::abs(x); });

    return *curvature;
  }


  template <typename Advect, typename Reinit>
  FELSPA_FORCE_INLINE auto
  LevelSetSurface<Advect, Reinit>::CurvatureEstimator::get_curvature_vector()
    const -> const vector_type&
  {
    return *curvature;
  }


  template <typename Advect, typename Reinit>
  FELSPA_FORCE_INLINE auto
  LevelSetSurface<Advect, Reinit>::CurvatureEstimator::get_curvature() const
    -> const TimedVector&
  {
    return curvature;
  }


  template <typename Advect, typename Reinit>
  template <typename QuadratureType>
  auto LevelSetSurface<Advect, Reinit>::CurvatureEstimator::assemble_rhs(
    const QuadratureType& quadrature) ->
    typename std::enable_if<
      std::is_base_of<dealii::Quadrature<dim>, QuadratureType>::value,
      void>::type

  {
    using namespace dealii;
    const auto& dofh = *this->ptr_dof_handler;

    const UpdateFlags update_flags =
      update_values | update_gradients | update_JxW_values;
    ScratchDataBox scratch_box(ptr_linear_system->get_mapping(), dofh.get_fe(),
                               quadrature, update_flags, **ptr_lvset_soln);
    CopyDataBox copy_box(dofh.get_fe().dofs_per_cell);

    rhs = 0.0;

    WorkStream::run(dofh.begin_active(), dofh.end(), *this,
                    &this_type::local_assembly,
                    &this_type::copy_local_to_global, scratch_box, copy_box);
  }


  template <typename Advect, typename Reinit>
  void LevelSetSurface<Advect, Reinit>::CurvatureEstimator::local_assembly(
    const CellIterator& cell, ScratchDataBox& scratch_box,
    CopyDataBox& copy_box)
  {
    copy_box.reset();

    // CELL INTEGRATION //
    copy_box.cell.reinit(cell);
    cell_assembly(scratch_box.reinit_cell(cell), copy_box.cell);
    copy_box.cell.set_active();

    for (unsigned int face_no = 0;
         face_no < dealii::GeometryInfo<dim>::faces_per_cell;
         ++face_no) {
      // BOUNDARY FACE INTEGRATION //
      if (cell->at_boundary(face_no)) {
        const auto& bdry_scratch = scratch_box.reinit_face(cell, face_no);
        CopyData& copy = copy_box.interior_faces[bdry_scratch.get_face_no()];
        copy.reinit(cell);
        boundary_assembly(bdry_scratch, copy);
      }

      else {
        // INTERNAL FACE INTEGRATION //
        const CellIterator neighbor = cell->neighbor(face_no);
        const bool neighbor_is_coarser = cell->neighbor_is_coarser(face_no);

        if (neighbor_is_coarser) {
          ASSERT(!cell->has_children(), ExcInternalErr());
          ASSERT(!neighbor->has_children(), ExcInternalErr());
          ASSERT(cell->level() < neighbor->level(), ExcInternalErr());

          const std::pair<unsigned int, unsigned int> neighbor_face_subface_no =
            cell->neighbor_of_coarser_neighbor(face_no);

          const auto face_scratch_pair = scratch_box.reinit_faces(
            cell, face_no, neighbor, neighbor_face_subface_no.first,
            neighbor_face_subface_no.second);

          const ScratchData& scratch_in = face_scratch_pair.first;
          const ScratchData& scratch_ex = face_scratch_pair.second;
          CopyData& copy_in = copy_box.interior_faces[scratch_in.get_face_no()];
          CopyData& copy_ex = copy_box.exterior_faces[scratch_ex.get_face_no()];

          copy_in.reinit(cell);
          copy_ex.reinit(neighbor);
          face_assembly(scratch_in, scratch_ex, copy_in, copy_ex);
        }

        else if (!neighbor->has_children() && cell->id() < neighbor->id()) {
          ASSERT(cell->level() == neighbor->level(), ExcInternalErr());
          ASSERT(!cell->has_children(), ExcInternalErr());
          ASSERT(!neighbor->has_children(), ExcInternalErr());

          const unsigned int neighbor_face_no =
            cell->neighbor_of_neighbor(face_no);

          const std::pair<ScratchData&, ScratchData&> face_scratch_pair =
            scratch_box.reinit_faces(cell, face_no, neighbor, neighbor_face_no);

          const ScratchData& scratch_in = face_scratch_pair.first;
          const ScratchData& scratch_ex = face_scratch_pair.second;
          CopyData& copy_in = copy_box.interior_faces[scratch_in.get_face_no()];
          CopyData& copy_ex = copy_box.exterior_faces[scratch_ex.get_face_no()];

          copy_in.reinit(cell);
          copy_ex.reinit(cell);
          face_assembly(scratch_in, scratch_ex, copy_in, copy_ex);
        }

        // Current cell is coarser the neighbor cell.
        // This face will be assembled by the neighbor cell.
        // Skip and go to the next face.
        else
          continue;
      }  // if internal face
    }    // face_no-loop
  }


  template <typename Advect, typename Reinit>
  void LevelSetSurface<Advect, Reinit>::CurvatureEstimator::cell_assembly(
    const ScratchData& s, CopyData& c)
  {
    using namespace dealii;
    const FEValuesBase<dim>& fe = s.fe_values();
    const auto ndof = fe.dofs_per_cell;
    const auto nqpt = fe.n_quadrature_points;
    auto& local_rhs = c.vector();
    const std::vector<value_type>& JxW = fe.get_JxW_values();

    // weak curvature term
    for (unsigned int idof = 0; idof < ndof; ++idof) {
      for (unsigned int iqpt = 0; iqpt < nqpt; ++iqpt) {
        local_rhs[idof] -= fe.shape_grad(idof, iqpt) * s.soln_grad_qpt[iqpt] /
                           s.soln_grad_qpt[iqpt].norm() * JxW[iqpt];
      }  // iqpt-loop
    }    // idof-loop
  }


  template <typename Advect, typename Reinit>
  void LevelSetSurface<Advect, Reinit>::CurvatureEstimator::face_assembly(
    const ScratchData& s_in, const ScratchData& s_ex, CopyData& c_in,
    CopyData& c_ex)
  {
    using namespace dealii;

    const FEValuesBase<dim>& fe_in = s_in.fe_values();
    const FEValuesBase<dim>& fe_ex = s_ex.fe_values();

    ASSERT(fe_in.n_quadrature_points == fe_ex.n_quadrature_points,
           ExcInternalErr());

    const auto ndof_in = fe_in.dofs_per_cell;
    const auto ndof_ex = fe_ex.dofs_per_cell;
    const auto nqpt = fe_in.n_quadrature_points;

    auto& local_rhs_in = c_in.vector();
    auto& local_rhs_ex = c_ex.vector();

    // outward pointing normals from the internal face.
    const std::vector<Tensor<1, dim, value_type>>& normals =
      fe_in.get_normal_vectors();


    for (unsigned int iqpt = 0; iqpt < nqpt; ++iqpt) {
      // do this dimension-by-dimension
      dealii::Tensor<1, dim, value_type> face_flux;

      value_type soln_grad_norm_in = s_in.soln_grad_qpt[iqpt].norm();
      value_type soln_grad_norm_ex = s_ex.soln_grad_qpt[iqpt].norm();

      for (int idim = 0; idim < dim; ++idim) {
        value_type shock_speed =
          (s_in.soln_grad_qpt[iqpt][idim] / soln_grad_norm_in -
           s_ex.soln_grad_qpt[iqpt][idim] / soln_grad_norm_ex) /
          (s_in.soln_qpt[iqpt] - s_ex.soln_qpt[iqpt]);

        if (shock_speed > 0.0)
          // shock travels to the right
          face_flux[idim] =
            normals[iqpt][idim] > 0
              ? s_in.soln_grad_qpt[iqpt][idim] / soln_grad_norm_in
              : s_ex.soln_grad_qpt[iqpt][idim] / soln_grad_norm_ex;
        else if (shock_speed < 0.0)
          // shock travels to the left
          face_flux[idim] =
            normals[iqpt][idim] > 0
              ? s_ex.soln_grad_qpt[iqpt][idim] / soln_grad_norm_ex
              : s_in.soln_grad_qpt[iqpt][idim] / soln_grad_norm_in;
        else
          face_flux[idim] = 0.0;
      }

      // internal cell face //
      for (unsigned int idof = 0; idof < ndof_in; ++idof)
        local_rhs_in[idof] += fe_in.shape_value(idof, iqpt) * face_flux *
                              normals[iqpt] * fe_in.JxW(iqpt);

      // external cell face //
      for (unsigned int idof = 0; idof < ndof_ex; ++idof)
        local_rhs_ex[idof] -= fe_ex.shape_value(idof, iqpt) * face_flux *
                              normals[iqpt] * fe_ex.JxW(iqpt);
    }  // iqpt-loop
  }


  template <typename Advect, typename Reinit>
  void LevelSetSurface<Advect, Reinit>::CurvatureEstimator::boundary_assembly(
    const ScratchData& s, CopyData& c)
  {
    using namespace dealii;
    const FEValuesBase<dim>& fe = s.fe_values();
    const auto nqpt = fe.n_quadrature_points;
    const auto ndof = fe.dofs_per_cell;

    auto& local_rhs = c.vector();

    for (unsigned int iqpt = 0; iqpt < nqpt; ++iqpt) {
      const std::vector<Tensor<1, dim, value_type>>& normals =
        fe.get_normal_vectors();
      for (unsigned int idof = 0; idof < ndof; ++idof)
        local_rhs[idof] += s.soln_grad_qpt[iqpt] /
                           s.soln_grad_qpt[iqpt].norm() * normals[iqpt] *
                           fe.shape_value(idof, iqpt) * fe.JxW(iqpt);
    }  // iqpt-loop
  }


  template <typename Advect, typename Reinit>
  void
  LevelSetSurface<Advect, Reinit>::CurvatureEstimator::copy_local_to_global(
    const CopyDataBox& copy_box)
  {
    // assemble RHS of the current cell
    copy_box.assemble(ptr_linear_system->get_constraints(), rhs);
  }


  /* ------------------------------------------------------ */
  /*  LevelSetSurface :: CurvatureEstimator :: ScratchData  */
  /* ------------------------------------------------------ */
  template <typename Advect, typename Reinit>
  LevelSetSurface<Advect, Reinit>::CurvatureEstimator::ScratchData::ScratchData(
    FEValuesEnum fevalenum, const vector_type& soln)
    : base_type(fevalenum), ptr_lvset_soln(&soln)
  {}


  template <typename Advect, typename Reinit>
  void
  LevelSetSurface<Advect, Reinit>::CurvatureEstimator::ScratchData::allocate()
  {
    const auto nqpt = this->fe_values().n_quadrature_points;
    soln_qpt.resize(nqpt);
    soln_grad_qpt.resize(nqpt);
  }


  template <typename Advect, typename Reinit>
  void LevelSetSurface<
    Advect, Reinit>::CurvatureEstimator::ScratchData::reinit_local_solution()
  {
    // interpolate solution to quadrature points
    this->fe_values().get_function_values(*ptr_lvset_soln, soln_qpt);
    // solution gradient at quadrature points
    this->fe_values().get_function_gradients(*ptr_lvset_soln, soln_grad_qpt);
  }


  template <typename Advect, typename Reinit>
  FELSPA_FORCE_INLINE void
  LevelSetSurface<Advect, Reinit>::CurvatureEstimator::ScratchData::reinit(
    const CellIterator& cell)
  {
    base_type::reinit(cell);
    reinit_local_solution();
  }


  template <typename Advect, typename Reinit>
  FELSPA_FORCE_INLINE void
  LevelSetSurface<Advect, Reinit>::CurvatureEstimator::ScratchData::reinit(
    const CellIterator& cell, unsigned int face_no)
  {
    base_type::reinit(cell, face_no);
    reinit_local_solution();
  }


  template <typename Advect, typename Reinit>
  FELSPA_FORCE_INLINE void
  LevelSetSurface<Advect, Reinit>::CurvatureEstimator::ScratchData::reinit(
    const CellIterator& cell, unsigned int face_no, unsigned int subface_no)
  {
    base_type::reinit(cell, face_no, subface_no);
    reinit_local_solution();
  }


  /* --------------------------------------------------------- */
  /*  LevelSetSurface :: CurvatureEstimator :: ScratchDataBox  */
  /* --------------------------------------------------------- */
  template <typename Advect, typename Reinit>
  template <typename QuadratureType>
  FELSPA_FORCE_INLINE LevelSetSurface<Advect, Reinit>::CurvatureEstimator::
    ScratchDataBox::ScratchDataBox(const dealii::Mapping<dim>& mapping,
                                   const dealii::FiniteElement<dim>& fe,
                                   const QuadratureType& quad,
                                   const dealii::UpdateFlags update_flags,
                                   const vector_type& lvset_soln)
    : base_type(lvset_soln)
  {
    this->add(mapping, fe, quad, update_flags);
    this->cell.allocate();
    this->face_in.allocate();
    this->face_ex.allocate();
    this->subface.allocate();
  }


  /* ---------------------------------------------- */
  /* Utilities functions for level set computations */
  /* ---------------------------------------------- */

  /* ************************************ */
  /** HeavisideFunction, linear smoothing */
  /* ************************************ */
  template <typename NumberType>
  auto HeavisideFunction<HeavisideSmoothing::linear, NumberType>::operator()(
    value_type x) const -> value_type
  {
    const auto eps = this->get_smoothing_width();
    if (x > eps)
      return 1.0;
    else if (x < -eps)
      return 0.0;
    else
      return 0.5 * (x / eps + 1);
  }

  /* ********************************** */
  /** HeavisideFunction, sine smoothing */
  /* ********************************** */
  template <typename NumberType>
  auto HeavisideFunction<HeavisideSmoothing::sine, NumberType>::operator()(
    value_type x) const -> value_type
  {
    using constants::PI;
    const auto eps = this->get_smoothing_width();

    if (x > eps)
      return 1.0;
    else if (x < -eps)
      return 0.0;
    else {
      auto val = 0.5 * (1.0 + x / eps + sin(x * PI / eps) / PI);
      if (std::abs(val) < std::numeric_limits<value_type>::epsilon())
        return 0.0;
      else if (std::abs(val - 1.0) < std::numeric_limits<value_type>::epsilon())
        return 1.0;
      else
        return val;
    }
  }

}  // namespace ls

FELSPA_NAMESPACE_CLOSE

#endif  // _FELSPA_LEVEL_SET_LEVEL_SET_IMPLEMENT_H_ //
