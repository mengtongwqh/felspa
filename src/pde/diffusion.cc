#include <deal.II/numerics/vector_tools.h>
#include <felspa/pde/diffusion.h>

FELSPA_NAMESPACE_OPEN

/* -------------------------------------------*/
namespace dg
/* -------------------------------------------*/
{
  template <int dim, typename NumberType>
  DiffusionSimulator<dim, NumberType>::DiffusionSimulator(
    Mesh<dim, value_type>& mesh,
    unsigned int fe_degree,
    const std::string& label)
    : base_type(mesh, fe_degree, label),
      grad_fe(this->get_fe(), dim),
      grad_dof_handler(mesh),
      ptr_grad_linear_system(
        std::make_shared<LDGGradientLinearSystem<dim, value_type>>(
          grad_dof_handler)),
      ptr_control(std::make_shared<DiffusionControl<dim, value_type>>())
  {
    this->tempo_integrator.attach_control(ptr_control->ptr_tempo);
  }


  template <int dim, typename NumberType>
  void DiffusionSimulator<dim, NumberType>::attach_control(
    const std::shared_ptr<control_type>& pcontrol)
  {
    ptr_control = pcontrol;
    this->tempo_integrator.attach_control(ptr_control->ptr_tempo);
  }


  template <int dim, typename NumberType>
  void DiffusionSimulator<dim, NumberType>::initialize(
    const initial_condition_type& initial_condition,
    bool use_independent_solution)
  {
    do_initialize(use_independent_solution);

    const auto& ic =
      static_cast<const dealii::Function<dim, value_type>&>(initial_condition);

    ptr_initial_condition = &initial_condition;
    dealii::VectorTools::interpolate(this->get_dof_handler(), ic,
                                     *(this->solution));

    this->initialized = true;
  }


  template <int dim, typename NumberType>
  void DiffusionSimulator<dim, NumberType>::do_initialize(
    bool use_independent_solution)
  {
    this->reset_time();
    allocate_assemble_system();

    // allocate solution
    if (!this->primary_simulator && use_independent_solution)
      this->solution.reinit(std::make_shared<vector_type>(), 0.0);
    if (this->solution.is_independent())
      this->solution->reinit(this->dof_handler().n_dofs());

    ASSERT(
      this->dof_handler().n_dofs() == this->solution->size(),
      ExcSizeMismatch(this->dof_handler().n_dofs(), this->solution->size()));
  }


  template <int dim, typename NumberType>
  bool DiffusionSimulator<dim, NumberType>::is_initialized() const
  {
    return this->initialized && ptr_initial_condition;
  }


  template <int dim, typename NumberType>
  void DiffusionSimulator<dim, NumberType>::upon_mesh_update()
  {
    base_type::upon_mesh_update();
    if (is_initialized()) {
      ASSERT(this->dof_handler().has_active_dofs(), ExcDoFHandlerNotInit());
      this->constraints().clear();
      this->constraints().close();
    }
  }


  template <int dim, typename NumberType>
  void DiffusionSimulator<dim, NumberType>::allocate_assemble_system()
  {
    // Diffusion system
    if (this->primary_simulator) {
      this->dof_handler().distribute_dofs(this->fe());
      this->linear_system().populate_system_from_dofs();
      assemble_mass_matrix();
    }

    // Gradient system
    grad_dof_handler.distribute_dofs(this->grad_fe);
    this->ptr_grad_linear_system->count_dofs();
    dealii::DoFRenumbering::component_wise(grad_dof_handler);
    ptr_grad_linear_system->populate_system_from_dofs();
    assemble_gradient_mass_matrix();

    // gradient solution
    this->solution_gradient.reinit(dim);
    const auto& ndofs_component = ptr_grad_linear_system->get_component_ndofs();
    for (int idim = 0; idim < dim; ++idim)
      solution_gradient.block(idim).reinit(ndofs_component[idim]);
    solution_gradient.collect_sizes();

    // reallocate solution
    // The proper way is to do this by solution transfer
    this->solution->reinit(this->dof_handler().n_dofs());

    this->mesh_update_detected = false;
  }


  template <int dim, typename NumberType>
  void DiffusionSimulator<dim, NumberType>::assemble_mass_matrix()
  {
    ASSERT(this->linear_system().is_populated(),
           EXCEPT_MSG("Linear system is not populated/allocated prior "
                      "to assembly. Call populate_system_from_dofs()."));
    MassMatrixAssembler<dim, value_type, LinearSystem> system_assembler(
      this->linear_system());
    system_assembler.assemble();
  }


  template <int dim, typename NumberType>
  void DiffusionSimulator<dim, NumberType>::assemble_gradient_mass_matrix()
  {
    ASSERT(this->ptr_grad_linear_system->is_populated(),
           EXCEPT_MSG("Gradient linear system is not populated/allocated prior "
                      "to assembly. Call populate_system_from_dofs()."));
    MassMatrixAssembler<dim, value_type, BlockLinearSystem> mass_assembler(
      *ptr_grad_linear_system);
    mass_assembler.assemble();
  }


  template <int dim, typename NumberType>
  void DiffusionSimulator<dim, NumberType>::assemble_gradient_rhs()
  {
    grad_dof_handler.initialize_local_block_info();
    dealii::BlockVector<value_type> solution_repeated(
      this->grad_dof_handler.block_info().global());
    solution_repeated.block(0) = *(this->solution);
    solution_repeated.collect_sizes();

    auto& pctrl = ptr_control->ptr_ldg;

    for (int idim = 0; idim < dim; ++idim)
      pctrl->ptr_assembler->beta[idim] = 1.0;

    // assemble the rhs
    LDGGradientAssembler<LDGGradientLinearSystem<dim, value_type>>
      grad_assembler(*ptr_grad_linear_system);
    grad_assembler.attach_control(pctrl->ptr_assembler);
    grad_assembler.template assemble<LDGFluxEnum::alternating>(
      solution_repeated, bcs, true);
  }


  template <int dim, typename NumberType>
  void DiffusionSimulator<dim, NumberType>::assemble_rhs()
  {
    LDGDiffusionAssembler<DGLinearSystem<dim, value_type>> diffusion_assembler(
      this->linear_system());
    diffusion_assembler.attach_control(ptr_control->ptr_ldg->ptr_assembler);
    diffusion_assembler.template assemble<LDGFluxEnum::alternating>(
      solution_gradient, bcs, true);
  }


  template <int dim, typename NumberType>
  void DiffusionSimulator<dim, NumberType>::solve_linear_system(
    vector_type& soln, const vector_type& rhs)
  {
    ASSERT_SAME_SIZE(soln, this->linear_system());
    ASSERT_SAME_SIZE(rhs, this->linear_system());
    this->linear_system().solve(soln, rhs, *ptr_control->ptr_solver);
  }


  template <int dim, typename NumberType>
  auto DiffusionSimulator<dim, NumberType>::explicit_time_derivative(
    time_step_type current_time, const vector_type& soln_prev_step)
    -> vector_type
  {
    UNUSED_VARIABLE(soln_prev_step);
    ASSERT(this->mesh_update_detected == false, ExcMeshUpdateUnprocessed());
    ASSERT(this->dof_handler().n_dofs() == this->solution->size(),
           ExcInternalErr());
    ASSERT(this->grad_dof_handler.n_dofs() == solution_gradient.size(),
           ExcInternalErr());
    ASSERT(this->ptr_control, ExcNullPointer());
    ASSERT(this->ptr_grad_linear_system->is_populated(), ExcLinSysNotInit());

    // solve this gradient system
    this->set_time_temporal_passive_members(current_time);

    if (this->mesh_update_detected) allocate_assemble_system();
    assemble_gradient_rhs();
    ptr_grad_linear_system->solve(solution_gradient,
                                  ptr_grad_linear_system->get_rhs(),
                                  *(ptr_control->ptr_ldg->ptr_solver));

    // asemble rhs to solve for diffusion
    assemble_rhs();
    vector_type phidot(*(this->solution));
    this->solve_linear_system(phidot, this->linear_system().get_rhs());
    return phidot;
  }


  template <int dim, typename NumberType>
  auto DiffusionSimulator<dim, NumberType>::estimate_max_time_step() const
    -> time_step_type
  {
    const auto diam = this->mesh().get_info().min_diameter;
    const auto viscosity = ptr_control->ptr_ldg->ptr_assembler->viscosity;
    const auto p = this->fe_degree();
    return diam * diam / viscosity / std::pow(p + 1, 4);
  }


  template <int dim, typename NumberType>
  auto DiffusionSimulator<dim, NumberType>::advance_time(
    time_step_type time_step, bool compute_single_cycle) -> time_step_type
  {
    const auto& tempo_control = *ptr_control->ptr_tempo;
    ASSERT(time_step > 0.0, ExcBackwardSync(time_step));
    ASSERT(is_initialized(), ExcSimulatorNotInitialized());
    ASSERT(tempo_control.query_method().second == TempoCategory::exp,
           ExcNotImplemented());

    time_step_type (TempoIntegrator<time_step_type>::*integrator)(
      DiffusionSimulator<dim, NumberType>&, time_step_type);
    time_step_type cumulative_time = 0.0;


    switch (tempo_control.query_method().first) {
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

    if (tempo_control.defined_auto_adjust()) {
      const auto max_cfl = tempo_control.get_cfl().second;

      do {
        ASSERT(time_step - cumulative_time > 0, ExcInternalErr());

        value_type suggest_time_step = max_cfl * estimate_max_time_step();
        value_type time_substep =
          std::min(suggest_time_step, time_step - cumulative_time);

        (this->tempo_integrator.*integrator)(*this, time_substep);
        cumulative_time += time_substep;

        if (compute_single_cycle) break;

      } while (!numerics::is_zero(cumulative_time - time_step));
    }

    else {
      ASSERT(compute_single_cycle, ExcArgumentCheckFail());
      (this->tempo_integrator.*integrator)(*this, time_step);
      cumulative_time += time_step;
    }

    return cumulative_time;
  }


  template <int dim, typename NumberType>
  void DiffusionSimulator<dim, NumberType>::export_solution(
    ExportFile& file) const
  {
    using namespace dealii;

    DataOut<dim> data_out;
    data_out.attach_dof_handler(this->get_dof_handler());
    const std::string& label = this->get_label_string();

#ifdef DEBUG
    std::vector<std::string> gradient_names(dim, label + "_gradient");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim,
                     DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector(grad_dof_handler, solution_gradient,
                             gradient_names, interpretation);
    data_out.clear_data_vectors();
#endif  // DEBUG //

    data_out.add_data_vector(*(this->solution), label);
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

}  // namespace dg

/* -------- Explicit Instantiations ----------*/
template class dg::DiffusionSimulator<1, types::DoubleType>;
template class dg::DiffusionSimulator<2, types::DoubleType>;
template class dg::DiffusionSimulator<3, types::DoubleType>;
/* -------------------------------------------*/
FELSPA_NAMESPACE_CLOSE
