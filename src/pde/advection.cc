#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/numerics/matrix_tools.h>
#include <felspa/pde/advection.h>

#include <cmath>
#include <functional>
#include <string>

#ifdef FELSPA_CXX_PARALLEL_ALGORITHM
#include <execution>
#endif

FELSPA_NAMESPACE_OPEN

/* -------------------------------------------*/
namespace dg
/* -------------------------------------------*/
{
  /* ************************************************** */
  /** \class AdvectSimulator */
  /* ************************************************** */
  template <int dim, typename NumberType>
  AdvectSimulator<dim, NumberType>::AdvectSimulator(
    Mesh<dim, value_type>& triag,
    unsigned int fe_degree,
    const std::string& label)
    : FESimulator<dim, FEDGType<dim>, DGLinearSystem<dim, NumberType>,
                  TempoIntegrator<NumberType>>(triag, fe_degree, label),
      ptr_control(std::make_shared<AdvectControl<NumberType>>())
  {
    // the default quadrature method is Gaussian quadrature
    this->set_quadrature(dealii::QGauss<dim>(fe_degree + 2));

    this->tempo_integrator.attach_control(ptr_control->ptr_tempo);
    this->ptr_solution_transfer = std::make_shared<AdvectSolutionTransfer>(
      this->ptr_fe, this->ptr_dof_handler, this->solution);
  }


  template <int dim, typename NumberType>
  AdvectSimulator<dim, NumberType>::AdvectSimulator(
    const AdvectSimulator<dim, NumberType>& that)
    : FESimulator<dim, FEDGType<dim>, DGLinearSystem<dim, NumberType>,
                  TempoIntegrator<NumberType>>(that),
      ptr_control(
        std::make_shared<AdvectControl<NumberType>>(*(that.ptr_control)))
  {
    // re-directing control object for tempo_integrator
    this->tempo_integrator.attach_control(ptr_control->ptr_tempo);
    ptr_source_term = nullptr;
  }


  template <int dim, typename NumberType>
  AdvectSimulator<dim, NumberType>& AdvectSimulator<dim, NumberType>::operator=(
    const AdvectSimulator<dim, NumberType>& that)
  {
    if (this == &that) return *this;
    this->FESimulator<dim, FEDGType<dim>, DGLinearSystem<dim, NumberType>,
                      TempoIntegrator<NumberType>>::operator=(that);
    ptr_control =
      std::make_shared<AdvectControl<NumberType>>(*(that.ptr_control));
    ptr_velocity_field.reset();
    this->tempo_integrator.attach_control(ptr_control->ptr_tempo);
    ptr_source_term = nullptr;
    return *this;
  }


  template <int dim, typename NumberType>
  void AdvectSimulator<dim, NumberType>::attach_control(
    const std::shared_ptr<AdvectControl<value_type>>& pcontrol)
  {
    ptr_control = pcontrol;
    this->tempo_integrator.attach_control(ptr_control->ptr_tempo);
  }


  template <int dim, typename NumberType>
  void AdvectSimulator<dim, NumberType>::allocate_assemble_system()
  {
    ASSERT(this->dof_handler().has_active_dofs(), ExcDoFHandlerNotInit());

    // If this is not the primary simulator,
    // linear system update will be done the primary simulator
    if (!this->primary_simulator) return;

    // allocate space in linear system object
    this->linear_system().populate_system_from_dofs();
    // reassemble the mass matrix
    this->assemble_mass_matrix();

    this->mesh_update_detected = false;

#ifdef DEBUG
    ASSERT(this->linear_system().size() == this->dof_handler().n_dofs(),
           ExcInternalErr());
    felspa_log << "Linear system setup completed and mass matrix assembled."
               << std::endl;
#endif  // DEBUG //
  }


  template <int dim, typename NumberType>
  bool AdvectSimulator<dim, NumberType>::is_initialized() const
  {
    return ptr_control && this->initialized &&
           ptr_velocity_field.lock() != nullptr;
  }


  template <int dim, typename NumberType>
  void AdvectSimulator<dim, NumberType>::assemble_mass_matrix()
  {
    // Assert linear_system is properly allocated
    ASSERT(this->linear_system().is_populated(),
           EXCEPT_MSG("Linear system is not populated/allocated prior "
                      "to assembly. Call populate_system_from_dofs()."));
    MassMatrixAssembler<dim, value_type, LinearSystem> system_assembler(
      this->linear_system());
    system_assembler.assemble();
  }


  template <int dim, typename NumberType>
  void AdvectSimulator<dim, NumberType>::assemble_rhs(
    const vector_type& soln_prev_step)
  {
    dealii::TimerOutput::Scope t(this->simulation_timer, "assemble_rhs");
    // Assert linear_system is properly allocated
    ASSERT(this->linear_system().is_populated(),
           EXCEPT_MSG("Linear system is not populated/allocated prior "
                      "to assembly. Call populate_system_from_dofs()."));

    ptr_rhs_assembler->assemble(*this->ptr_quadrature, soln_prev_step,
                                this->bcs, ptr_source_term);
  }


  template <int dim, typename NumberType>
  void AdvectSimulator<dim, NumberType>::discretize_function_to_vector(
    const ScalarFunction<dim, value_type>& function_field,
    vector_type& vec) const
  {
    const auto& dealii_fcn =
      static_cast<const dealii::Function<dim, value_type>&>(function_field);
    ASSERT(vec.size() == this->get_dof_handler().n_dofs(),
           ExcSizeMismatch(vec.size(), this->get_dof_handler().n_dofs()));

    if (this->get_fe().has_generalized_support_points())
      dealii::VectorTools::interpolate(this->get_dof_handler(), dealii_fcn,
                                       vec);
    else
      dealii::VectorTools::project(this->get_dof_handler(), this->constraints(),
                                   this->get_quadrature(), dealii_fcn, vec);
  }


  template <int dim, typename NumberType>
  void AdvectSimulator<dim, NumberType>::discretize_function_to_solution(
    const ScalarFunction<dim, value_type>& function_field)
  {
    discretize_function_to_vector(function_field, *this->solution);
  }


  template <int dim, typename NumberType>
  void AdvectSimulator<dim, NumberType>::upon_mesh_update()
  {
    base_type::upon_mesh_update();
    if (this->initialized) {
      ASSERT(this->dof_handler().has_active_dofs(), ExcDoFHandlerNotInit());
      this->constraints().clear();
      this->constraints().close();
    }
  }


  template <int dim, typename NumberType>
  void AdvectSimulator<dim, NumberType>::do_flag_mesh_for_coarsen_and_refine(
    const MeshControl<value_type>& mesh_control) const
  {
    using namespace dealii;

    // Put this assertion here because otherwise
    // the DoFHandler will not be re-distributed
    ASSERT(this->is_initialized(), ExcSimulatorNotInitialized());
    LOG_PREFIX("AdvectSimulator");
    felspa_log << "Labelling mesh for carsening and refinement..." << std::endl;

    MeshFlagOperator<dim, value_type> flag_op(this->mesh());

    // store the flags
    const auto n_cells = this->mesh().n_active_cells();
    std::vector<bool> old_refine_flags(n_cells), old_coarse_flags(n_cells);
    Vector<float> grad_indicator(n_cells);
    this->mesh().save_refine_flags(old_refine_flags);
    this->mesh().save_coarsen_flags(old_coarse_flags);

    std::for_each(this->mesh().active_cell_iterators().begin(),
                  this->mesh().active_cell_iterators().end(),
                  [](const TriaActiveIterator<CellAccessor<dim, dim>>& cell) {
                    cell->clear_refine_flag();
                    cell->clear_coarsen_flag();
                  });


    // use gradient approximation
    // and they will be cell-wise scaled by the factor $h^{1+d/2}$
    DerivativeApproximation::approximate_gradient(
      this->get_dof_handler(), this->get_solution_vector(), grad_indicator);

#ifdef FELSPA_CXX_PARALLEL_ALGORITHM
    {
      // dealii::TimerOutput::Scope(this->simulation_timer,
      // "parallel_transform"); auto active_cell_range =
      // this->get_mesh().active_cell_iterators();
      const auto& const_grad_indicator = grad_indicator;
      std::transform(std::execution::par,
                     const_grad_indicator.begin(),
                     const_grad_indicator.end(),
                     this->get_dof_handler().begin_active(),
                     grad_indicator.begin(),
                     [](float grad_entry,
                        const typename DoFHandler<dim>::cell_accessor& cell) {
                       return grad_entry *
                              std::pow(cell.diameter(), 1.0 + dim / 2.0);
                     });
    }
#else
    {
      // dealii::TimerOutput::Scope(this->simulation_timer, "serial_transform");
      auto it_grad = grad_indicator.begin();
      for (const auto& cell : this->mesh().active_cell_iterators())
        *(it_grad++) *= std::pow(cell->diameter(), 1.0 + dim / 2.0);
    }
#endif  // FELSPA_CXX_PARALLEL_ALGORITHM


    GridRefinement::refine_and_coarsen_fixed_number(
      this->mesh(), grad_indicator, mesh_control.refine_top_fraction,
      mesh_control.coarsen_bottom_fraction);


    flag_op.prioritize_refinement();
    flag_op.limit_level(mesh_control.min_level, mesh_control.max_level);
#ifdef DEBUG
    flag_op.print_info(felspa_log);
#endif  // DEBUG //
  }


  template <int dim, typename NumberType>
  void AdvectSimulator<dim, NumberType>::solve_linear_system(
    vector_type& soln, const vector_type& rhs)
  {
    ASSERT_SAME_SIZE(soln, this->linear_system());
    ASSERT_SAME_SIZE(rhs, this->linear_system());
    dealii::TimerOutput::Scope t(this->simulation_timer, "solve_linear_system");
    this->linear_system().solve(soln, rhs, *(ptr_control->ptr_solver));
  }


  template <int dim, typename NumberType>
  typename AdvectSimulator<dim, NumberType>::vector_type
  AdvectSimulator<dim, NumberType>::explicit_time_derivative(
    time_step_type current_time, const vector_type& soln_prev_step)
  {
    LOG_PREFIX("AdvectSimulator");
    ASSERT(this->initialized, ExcSimulatorNotInitialized());

    // advance passive members forward to current_time
    this->set_time_temporal_passive_members(current_time);
    if (this->mesh_update_detected) this->allocate_assemble_system();

    ASSERT(this->mesh_update_detected == false, ExcMeshUpdateUnprocessed());
    ASSERT(this->linear_system().is_populated(), ExcLinSysNotInit());

    // assemble and solve the linear system
    ASSERT(
      this->dof_handler().n_dofs() == this->solution->size(),
      ExcSizeMismatch(this->dof_handler().n_dofs(), this->solution->size()));
    ASSERT(this->dof_handler().n_dofs() == this->linear_system().size(),
           ExcSizeMismatch(this->dof_handler().n_dofs(),
                           this->linear_system().size()));

    assemble_rhs(soln_prev_step);
    vector_type phidot(*(this->solution));
    this->solve_linear_system(phidot, this->linear_system().get_rhs());
    return phidot;
  }


  template <int dim, typename NumberType>
  auto AdvectSimulator<dim, NumberType>::max_velocity_over_diameter(
    time_step_type current_time) const -> value_type
  {
    /**
     *  This CFL estimation is computed by
     *  (largest velocity l2-norm among all vertices of the cell) /
     *  (diameter of the cell)
     */
    UNUSED_VARIABLE(current_time);

    value_type velo_diam;

    if (ptr_cfl_estimator->is_simulator)
      velo_diam = ptr_cfl_estimator->estimate(
        this->get_dof_handler().begin_active(), this->get_dof_handler().end());
    else
      velo_diam = ptr_cfl_estimator->estimate(
        this->get_dof_handler().begin_active(), this->get_dof_handler().end(),
        this->get_mapping(), this->get_fe());

    // multiply the dg_scaling
    return velo_diam * cfl_scaling(this->fe_degree());
  }


  template <int dim, typename NumberType>
  void AdvectSimulator<dim, NumberType>::advance_time(time_step_type time_step)
  {
    const auto method_info = ptr_control->ptr_tempo->query_method();

    ASSERT(time_step >= 0.0, ExcBackwardSync(time_step));
    ASSERT(is_initialized(), ExcSimulatorNotInitialized());
    ASSERT(method_info.second == TempoCategory::exp,
           EXCEPT_MSG("AdvectionSimulator only allow explicit time stepping."));

    switch (method_info.first) {
      case TempoMethod::rktvd1:
        this->tempo_integrator
          .template advance_time_step<RungeKuttaTVD<1>, TempoCategory::exp>(
            *this, time_step);
        break;
      case TempoMethod::rktvd2:
        this->tempo_integrator
          .template advance_time_step<RungeKuttaTVD<2>, TempoCategory::exp>(
            *this, time_step);
        break;
      case TempoMethod::rktvd3:
        this->tempo_integrator
          .template advance_time_step<RungeKuttaTVD<3>, TempoCategory::exp>(
            *this, time_step);
        break;
      default:
        THROW(ExcNotImplemented());
    }

    this->flag_mesh_for_coarsen_and_refine(*this->control().ptr_mesh);
  }


  template <int dim, typename NumberType>
  auto AdvectSimulator<dim, NumberType>::advance_time(time_step_type time_step,
                                                      bool compute_single_cycle)
    -> time_step_type
  {
    const auto method_info = ptr_control->ptr_tempo->query_method();

    ASSERT(time_step >= 0.0, ExcBackwardSync(time_step));
    ASSERT(is_initialized(), ExcSimulatorNotInitialized());
    ASSERT(method_info.second == TempoCategory::exp,
           EXCEPT_MSG("AdvectionSimulator only allow explicit time stepping."));

    time_step_type (TempoIntegrator<time_step_type>::*integrator)(
      AdvectSimulator<dim, NumberType>&, time_step_type);
    time_step_type cumulative_time = 0.0;

    switch (method_info.first) {
      case TempoMethod::rktvd1:
        integrator =
          &TempoIntegrator<time_step_type>::template advance_time_step<
            RungeKuttaTVD<1>, TempoCategory::exp>;
        break;
      case TempoMethod::rktvd2:
        integrator =
          &TempoIntegrator<time_step_type>::template advance_time_step<
            RungeKuttaTVD<2>, TempoCategory::exp>;
        break;
      case TempoMethod::rktvd3:
        integrator =
          &TempoIntegrator<time_step_type>::template advance_time_step<
            RungeKuttaTVD<3>, TempoCategory::exp>;
        break;
      default:
        THROW(ExcNotImplemented());
    }  // switch get_tempo_method() //

    // forward the solution for this time step
    if (ptr_control->ptr_tempo->defined_auto_adjust()) {
      value_type max_cfl = this->ptr_control->ptr_tempo->get_cfl().second;
      ASSERT(max_cfl > 0.0, EXCEPT_MSG("Max CFL must be positive."));

      do {
        value_type suggest_time_step =
          max_cfl / this->max_velocity_over_diameter(this->get_time());
        value_type time_substep =
          std::min(suggest_time_step, time_step - cumulative_time);

        ASSERT(time_step - cumulative_time > 0, ExcInternalErr());

        (this->tempo_integrator.*integrator)(*this, time_substep);
        cumulative_time += time_substep;

        this->flag_mesh_for_coarsen_and_refine(*this->control().ptr_mesh);

        if (compute_single_cycle) break;

      } while (!numerics::is_zero(cumulative_time - time_step));
    }

    else {
      ASSERT(compute_single_cycle, ExcArgumentCheckFail());
      (this->tempo_integrator.*integrator)(*this, time_step);
      this->flag_mesh_for_coarsen_and_refine(*this->control().ptr_mesh);
      cumulative_time += time_step;
    }

    return cumulative_time;
  }


  template <int dim, typename NumberType>
  auto AdvectSimulator<dim, NumberType>::advance_time() -> time_step_type
  {
    ASSERT(ptr_control->ptr_tempo->defined_auto_adjust(),
           EXCEPT_MSG("Auto adjust flag must be set"));
    time_step_type time_step = estimate_max_time_step();
    advance_time(time_step);
    return time_step;
  }


  template <int dim, typename NumberType>
  auto AdvectSimulator<dim, NumberType>::estimate_max_time_step() const
    -> time_step_type
  {
    value_type max_velo_diam =
      this->max_velocity_over_diameter(this->get_time());
    value_type max_cfl = this->ptr_control->ptr_tempo->get_cfl().second;
    ASSERT(max_cfl > 0.0, EXCEPT_MSG("Max CFL must be positive."));
    return max_cfl / max_velo_diam;
  }


  template <int dim, typename NumberType>
  auto AdvectSimulator<dim, NumberType>::compute_error(
    const ScalarFunction<dim, value_type>& analytical_soln,
    dealii::VectorTools::NormType norm_type) const -> value_type
  {
    dealii::Vector<value_type> difference_per_cell(
      this->mesh().n_active_cells());

    dealii::VectorTools::integrate_difference(
      this->get_dof_handler(), *this->solution,
      static_cast<const dealii::Function<dim, NumberType>&>(analytical_soln),
      difference_per_cell, dealii::QGauss<dim>(this->fe_degree() + 1),
      norm_type);

    return dealii::VectorTools::compute_global_error(
      this->mesh(), difference_per_cell, norm_type);
  }


  template <int dim, typename NumberType>
  auto AdvectSimulator<dim, NumberType>::compute_error(
    const vector_type& discrete_soln) const -> value_type
  {
    ASSERT_SAME_SIZE(discrete_soln, this->get_solution_vector());

    value_type diff{value_type()};
    for (auto it1 = discrete_soln.begin(),
              it2 = this->get_solution_vector().begin();
         it1 != discrete_soln.end(); ++it1, ++it2)
      diff += std::abs(*it2 - *it1);
    
    return diff / discrete_soln.size();
  }


  template <int dim, typename NumberType>
  auto AdvectSimulator<dim, NumberType>::cfl_scaling(const unsigned int degree)
    -> time_step_type
  {
    using value_type = typename TempoControl<NumberType>::value_type;
    value_type p = static_cast<value_type>(degree);
    value_type cfl_coeff = 4.0 / (2.0 + p) / (2.0 + p) + 1.0;
    cfl_coeff *= (2.0 * p + 1.0);
    return cfl_coeff;
  }


  /* --------------------------------------------------*/
  /** \class AdvectSimulator::AdvectSolutionTransfer   */
  /* --------------------------------------------------*/

  template <int dim, typename NumberType>
  AdvectSimulator<dim, NumberType>::AdvectSolutionTransfer::
    AdvectSolutionTransfer(
      const std::shared_ptr<const fe_type>& pfe,
      const std::shared_ptr<dealii::DoFHandler<dim>>& pdofh,
      TimedSolutionVector<vector_type, time_step_type>& solution)
    : ptr_fe(pfe),
      ptr_dof_handler(pdofh),
      ptr_soln(&solution),
      soln_transfer(*pdofh)
  {
    ASSERT(pdofh.get(), ExcNullPointer());
  }


  template <int dim, typename NumberType>
  void AdvectSimulator<dim, NumberType>::AdvectSolutionTransfer::
    prepare_for_coarsening_and_refinement()
  {
    // no point in doing solution transfer
    // if it is managed by another simulator
    if (!ptr_soln->is_independent()) return;

#ifndef FELSPA_HAS_MPI
    soln_transfer.clear();
#endif
    soln_transfer.prepare_for_coarsening_and_refinement(**ptr_soln);
    felspa_log << "SolutionTransfer prepared for coarsening and refinement."
               << std::endl;
  }


  template <int dim, typename NumberType>
  void AdvectSimulator<dim, NumberType>::AdvectSolutionTransfer::interpolate()
  {
    ASSERT(ptr_dof_handler->n_dofs() > 0, EXCEPT_MSG("DoFHandler is empty."));

    // no point in doing solution transfer
    // if it is managed by another simulator
    if (!ptr_soln->is_independent()) return;

    vector_type interpolated_soln(ptr_dof_handler->n_dofs());

    felspa_log << "SolutionTransfer working, now have "
               << ptr_dof_handler->get_triangulation().n_active_cells()
               << " active cells and " << ptr_dof_handler->n_dofs()
               << " dofs...";
#ifdef FELSPA_HAS_MPI
    soln_transfer.interpolate(interpolated_soln);
#else
    soln_transfer.interpolate(**ptr_soln, interpolated_soln);
#endif
    (*ptr_soln)->swap(interpolated_soln);

    felspa_log << "done." << std::endl;
  }


}  // namespace dg

/* -------- Explicit Instantiations ----------*/
#include "advection.inst"
/* -------------------------------------------*/
FELSPA_NAMESPACE_CLOSE
