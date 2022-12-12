
#ifndef _FELSPA_PDE_STOKES_COMMON_IMPLEMENT_H_
#define _FELSPA_PDE_STOKES_COMMON_IMPLEMENT_H_

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/numerics/error_estimator.h>
#include <felspa/pde/stokes_common.h>

FELSPA_NAMESPACE_OPEN

namespace internal
{
  template <typename MatrixType>
  void StokesScalingOperator<MatrixType>::initialize(const matrix_type& matrix)
  {
    size_type v_block_size = matrix.block(0, 0).m();
    size_type p_block_size = matrix.block(1, 1).m();

    // scaling factor for velocity block
    for (size_type irow = 0; irow != v_block_size; ++irow) {
      value_type max_abs_value = 0.0;
      for (auto it = matrix.block(0, 0).begin(irow);
           it != matrix.block(0, 0).end(irow);
           ++it)
        max_abs_value = std::max(std::abs(it->value()), max_abs_value);

      ASSERT(!::FELSPA_NAMESPACE::numerics::is_zero(max_abs_value),
             ExcInternalErr());
      scaling_coeffs.block(0)[irow] = std::sqrt(max_abs_value);
    }

    auto g_hat = scaling_coeffs.block(0);

    //  extract maximum value for the pressure block
    for (size_type irow = 0; irow != v_block_size; ++irow) {
      value_type max_abs_value = 0.0;
      for (auto it = matrix.block(0, 1).begin(irow);
           it != matrix.block(0, 1).end(irow);
           ++it)
        max_abs_value = std::max(std::abs(it->value()), max_abs_value);
      g_hat[irow] = max_abs_value;
    }

    // pressure scaling value
    value_type p_scale_val = 0.0;
    for (auto val : scaling_coeffs.block(0)) {
      ASSERT(!::FELSPA_NAMESPACE::numerics::is_zero(val), ExcDividedByZero());
      p_scale_val += 1.0 / val / val;
    }

    p_scale_val *= std::sqrt(p_scale_val * g_hat.norm_sqr()) / p_block_size;
    scaling_coeffs.block(1) = p_scale_val;
  }


  template <typename MatrixType>
  void StokesScalingOperator<MatrixType>::apply_to_matrix(
    matrix_type& matrix) const
  {
    ASSERT(
      scaling_coeffs.block(0).size() == matrix.block(0, 0).m(),
      ExcSizeMismatch(scaling_coeffs.block(0).size(), matrix.block(0, 0).m()));
    ASSERT(
      scaling_coeffs.block(1).size() == matrix.block(1, 1).m(),
      ExcSizeMismatch(scaling_coeffs.block(1).size(), matrix.block(1, 1).m()));

    // Block(0,0)
    for (auto accessor : matrix.block(0, 0)) {
      accessor.value() /= scaling_coeffs.block(0)[accessor.row()] *
                          scaling_coeffs.block(0)[accessor.column()];
    }
    // Block(1,0)
    for (auto accessor : matrix.block(0, 1)) {
      accessor.value() /= scaling_coeffs.block(0)[accessor.row()] *
                          scaling_coeffs.block(1)[accessor.column()];
    }
    // Block(0,1)
    for (auto accessor : matrix.block(1, 0)) {
      accessor.value() /= scaling_coeffs.block(1)[accessor.row()] *
                          scaling_coeffs.block(0)[accessor.column()];
    }
  }


  template <typename MatrixType>
  void StokesScalingOperator<MatrixType>::apply_to_preconditioner(
    matrix_block_type& precond_matrix) const
  {
    ASSERT(scaling_coeffs.block(1).size() == precond_matrix.m(),
           ExcSizeMismatch(scaling_coeffs.block(0).size(), precond_matrix.m()));
    for (auto accessor : precond_matrix) {
      accessor.value() /= scaling_coeffs.block(1)[accessor.row()] *
                          scaling_coeffs.block(1)[accessor.column()];
    }
  }

  template <typename MatrixType>
  void StokesScalingOperator<MatrixType>::apply_to_vector(vector_type& rhs,
                                                          int component) const
  {
    ASSERT_SAME_SIZE(scaling_coeffs.block(0), rhs.block(0));
    ASSERT_SAME_SIZE(scaling_coeffs.block(1), rhs.block(1));

    if (component == -1) {
      apply_to_vector(rhs, 0);
      apply_to_vector(rhs, 1);

    } else if (component == 0 || component == 1) {
      auto it_coeffs = scaling_coeffs.block(component).begin();
      auto it_rhs = rhs.block(component).begin();
      for (; it_rhs != rhs.block(component).end(); ++it_rhs, ++it_coeffs) {
        ASSERT(!numerics::is_zero(*it_coeffs), ExcDividedByZero());
        *it_rhs /= *it_coeffs;
      }

    } else
      THROW(ExcUnexpectedValue<int>(component));
  }

  template <typename MatrixType>
  void StokesScalingOperator<MatrixType>::apply_inverse_to_vector(
    vector_type& soln, int component) const
  {
    ASSERT_SAME_SIZE(scaling_coeffs.block(0), soln.block(0));
    ASSERT_SAME_SIZE(scaling_coeffs.block(1), soln.block(1));

    if (component == -1) {
      apply_inverse_to_vector(soln, 0);
      apply_inverse_to_vector(soln, 1);

    } else if (component == 0 || component == 1) {
      auto it_coeffs = scaling_coeffs.block(component).begin();
      auto it_soln = soln.block(component).begin();
      for (; it_soln != soln.block(component).end(); ++it_soln, ++it_coeffs) {
        *it_soln *= *it_coeffs;
      }

    } else
      THROW(ExcUnexpectedValue<int>(component));
  }

}  // namespace internal


/* ************************************************** */
/*               StokesSimulatorBase                  */
/* ************************************************** */

template <int dim, typename NumberType, typename LinsysType>
StokesSimulatorBase<dim, NumberType, LinsysType>::StokesSimulatorBase(
  Mesh<dim, value_type>& mesh, unsigned int degree_v, unsigned int degree_p,
  const std::string& label)
  : StokesSimulatorBase(mesh, dealii::FE_Q<dim>(degree_v),
                        dealii::FE_Q<dim>(degree_p), label)
{
  ASSERT(
    degree_v > degree_p,
    EXCEPT_MSG("velocity FE degree must be greater than pressure FE degree."));
}


template <int dim, typename NumberType, typename LinsysType>
StokesSimulatorBase<dim, NumberType, LinsysType>::StokesSimulatorBase(
  Mesh<dim, value_type>& mesh, const dealii::FiniteElement<dim>& fe_v,
  const dealii::FiniteElement<dim>& fe_p, const std::string& label)
  : base_type(mesh, fe_v, dim, fe_p, 1, label)
{
  this->set_quadrature(dealii::QGauss<dim>(fe_v.degree + 1));
  this->ptr_solution_transfer = std::make_shared<StokesSolutionTransfer>(*this);
}


template <int dim, typename NumberType, typename LinsysType>
void StokesSimulatorBase<dim, NumberType, LinsysType>::attach_control(
  const std::shared_ptr<StokesControlBase<value_type>>& pcontrol)
{
  ASSERT(pcontrol, ExcNullPointer());
  ptr_control = pcontrol;
}


template <int dim, typename NumberType, typename LinsysType>
StokesControlBase<NumberType>&
StokesSimulatorBase<dim, NumberType, LinsysType>::control()
{
  ASSERT(this->ptr_control, ExcNullPointer());
  return *this->ptr_control;
}


template <int dim, typename NumberType, typename LinsysType>
const StokesControlBase<NumberType>&
StokesSimulatorBase<dim, NumberType, LinsysType>::get_control() const
{
  return control();
}


template <int dim, typename NumberType, typename LinsysType>
template <typename MaterialType>
void StokesSimulatorBase<dim, NumberType, LinsysType>::initialize(
  const std::shared_ptr<MaterialType>& p_material,
  const std::shared_ptr<TensorFunction<1, dim, value_type>>& p_gravity_model,
  bool use_independent_solution)
{
  ASSERT(p_material != nullptr, ExcNullPointer());

  ptr_material = p_material;
  ptr_momentum_source = p_gravity_model;
  this->add_temporal_passive_member(ptr_material.lock());

  if (p_gravity_model != nullptr)
    this->add_temporal_passive_member(p_gravity_model);

  // set all object members to time 0
  this->reset_time();

  // set initialization flag here so that upon_mesh_update
  // will distribute dofs and count dofs.
  this->initialized = true;

  // distribute_dofs and count_dofs
  upon_mesh_update();

  // allocate solution object again if this is not a primary simulator
  // but still wants to have its own solution object.
  if (!this->primary_simulator && use_independent_solution)
    this->solution.reinit(std::make_shared<vector_type>(), 0.0);

  // allocate the solution vector in the solution object
  if (this->solution.is_independent())
    allocate_solution_vector(*this->solution);

  this->mesh_update_detected = true;  // force reallocation

  ptr_assembler =
    std::make_shared<StokesAssembler<linsys_type>>(this->linear_system());

  const auto sp_material = ptr_material.lock();
  ASSERT(sp_material != nullptr, ExcExpiredPointer());
  ptr_assembler->initialize(this->bcs, sp_material, ptr_momentum_source);

#if 0
  if (!this->get_control().material_parameters_to_export.empty())
    this->ptr_material_parameter_exporter = std::make_unique<
      internal::StokesMaterialParameterExporter<dim, value_type>>(*this);
#endif
}


template <int dim, typename NumberType, typename LinsysType>
FELSPA_FORCE_INLINE unsigned int
StokesSimulatorBase<dim, NumberType, LinsysType>::fe_degree(
  unsigned int component) const
{
  return this->ptr_fe->base_element(component).degree;
}


template <int dim, typename NumberType, typename LinsysType>
FELSPA_FORCE_INLINE unsigned int
StokesSimulatorBase<dim, NumberType, LinsysType>::fe_degree(
  SolutionComponent component) const
{
  return fe_degree(static_cast<unsigned int>(component));
}


template <int dim, typename NumberType, typename LinsysType>
FELSPA_FORCE_INLINE bool
StokesSimulatorBase<dim, NumberType, LinsysType>::is_initialized() const
{
  return this->initialized && ptr_material.lock() != nullptr;
}


template <int dim, typename NumberType, typename LinsysType>
void StokesSimulatorBase<dim, NumberType, LinsysType>::advance_time(
  time_step_type time_step)
{
  try_advance_time();

#ifdef EXPORT_MATRIX
  std::ofstream os("StokesLHSMatrix.dat");
  this->linear_system().get_matrix().print_formatted(os);
  os.close();
#endif

  finalize_time_step(time_step);
}


template <int dim, typename NumberType, typename LinsysType>
void StokesSimulatorBase<dim, NumberType, LinsysType>::try_advance_time()
{
  ASSERT(!requires_finalizing_time_step, ExcTimeStepNotFinalized());
  ASSERT(is_initialized(), ExcSimulatorNotInitialized());
  ASSERT(this->is_synchronized(), ExcNotSynchronized());

  if (this->mesh_update_detected) allocate_system();
  assemble_system();
  solve_linear_system(*this->solution);
  requires_finalizing_time_step = true;
}


template <int dim, typename NumberType, typename LinsysType>
void StokesSimulatorBase<dim, NumberType, LinsysType>::finalize_time_step(
  time_step_type time_step)
{
  time_step_type current_time = time_step + this->get_time();
  this->solution.set_time(current_time);
  this->phsx_time = current_time;
  this->set_time_temporal_passive_members(current_time);
  requires_finalizing_time_step = false;

  if (this->ptr_mesh_refiner)
    this->flag_mesh_for_coarsen_and_refine(*this->ptr_control->ptr_mesh);
}


template <int dim, typename NumberType, typename LinsysType>
void StokesSimulatorBase<dim, NumberType, LinsysType>::upon_mesh_update()
{
  // implies distribute dof
  base_type::upon_mesh_update();
#if 0
  if (ptr_material_parameter_exporter != nullptr)
    ptr_material_parameter_exporter->mesh_updated();
#endif

  if (this->initialized && this->primary_simulator) {
    ASSERT(this->dof_handler().has_active_dofs(), ExcDoFHandlerNotInit());
    this->linear_system().upon_mesh_update();

    dealii::DoFRenumbering::Cuthill_McKee(this->dof_handler());
    std::vector<types::SizeType> block_component(dim + 1, 0);
    block_component[dim] = 1;
    dealii::DoFRenumbering::component_wise(this->dof_handler(),
                                           block_component);

    felspa_log << "Stokes simulator has " << this->mesh().n_active_cells()
               << " active cells" << std::endl;
  }
}


template <int dim, typename NumberType, typename LinsysType>
void StokesSimulatorBase<dim, NumberType, LinsysType>::allocate_system()
{
  dealii::TimerOutput::Scope t(this->simulation_timer, "allocate_system");
  ASSERT(this->dof_handler().has_active_dofs(), ExcDoFHandlerNotInit());

  // only primary simulator is responsible for
  // dof_handler update and linsys allocation
  if (!this->primary_simulator) return;

  // compute DoF renumbering
  // std::vector<types::SizeType> block_component(dim + 1, 0);
  // block_component[dim] = 1;
  // dealii::DoFRenumbering::boost::king_ordering(this->dof_handler());
  // dealii::DoFRenumbering::boost::Cuthill_McKee(this->dof_handler());
  // dealii::DoFRenumbering::component_wise(this->dof_handler(),
  // block_component);

  // allocate the linear system
  // We don't need to reallocate the solution vector
  // since it is taken care of in the mesh_refine routines.
  this->linear_system().setup_constraints_and_system(this->bcs);

  // alternative, do this serially:
  // this->linear_system().setup_constraints(this->bcs);
  // this->linear_system().populate_system_from_dofs();

  // falsify mesh_update_detected flag
  this->mesh_update_detected = false;
}


template <int dim, typename NumberType, typename LinsysType>
void StokesSimulatorBase<dim, NumberType, LinsysType>::assemble_system()
{
  LOG_PREFIX("StokesSimulatorBase");
  dealii::TimerOutput::Scope t(this->simulation_timer,
                               "assemble_stokes_system");
  ptr_assembler->assemble(this->get_quadrature());

  // apply pressure scaling
  // this->linear_system().apply_pressure_scaling(ptr_control->reference_viscosity,
  //                                              ptr_control->reference_length);

#ifdef DEBUG
  for (const auto& i : this->linear_system().get_matrix())
    ASSERT(!std::isnan(i.value()) && std::isfinite(i.value()),
           ExcUnexpectedValue<value_type>(i.value()));

  for (const auto& i :
       this->linear_system().get_preconditioner_matrix().block(0, 0))
    ASSERT(!std::isnan(i.value()) && std::isfinite(i.value()),
           ExcUnexpectedValue<value_type>(i.value()));

  for (const auto& i :
       this->linear_system().get_preconditioner_matrix().block(1, 1))
    ASSERT(!std::isnan(i.value()) && std::isfinite(i.value()),
           ExcUnexpectedValue<value_type>(i.value()));

  for (const auto& i : this->linear_system().get_rhs())
    ASSERT(!std::isnan(i) && std::isfinite(i),
           ExcUnexpectedValue<value_type>(i));
#endif

  felspa_log << "Stokes linear system assembly completed and the pressure "
                "scaling coeff = "
             << ptr_control->reference_viscosity << " / "
             << ptr_control->reference_length << std::endl;
}


template <int dim, typename NumberType, typename LinsysType>
void StokesSimulatorBase<dim, NumberType, LinsysType>::solve_linear_system(
  vector_type& soln, vector_type& rhs)
{
  dealii::TimerOutput::Scope t(this->simulation_timer, "solve_stokes_system");

  // const value_type pressure_scaling =
  //   ptr_control->reference_viscosity / ptr_control->reference_length;

  // soln.block(1) /= pressure_scaling;
  this->linear_system().solve(soln, rhs, *ptr_control->ptr_solver);
  // soln.block(1) *= pressure_scaling;
}


template <int dim, typename NumberType, typename LinsysType>
void StokesSimulatorBase<dim, NumberType, LinsysType>::solve_linear_system(
  vector_type& soln)
{
  dealii::TimerOutput::Scope t(this->simulation_timer, "solve_stokes_system");

  // const value_type pressure_scaling =
  //   ptr_control->reference_viscosity / ptr_control->reference_length;

  // soln.block(1) /= pressure_scaling;
  this->linear_system().solve(soln, *ptr_control->ptr_solver);
  // soln.block(1) *= pressure_scaling;
}


template <int dim, typename NumberType, typename LinsysType>
void StokesSimulatorBase<dim, NumberType, LinsysType>::
  do_flag_mesh_for_coarsen_and_refine(
    const MeshControl<value_type>& mesh_control) const
{
  using namespace dealii;
  LOG_PREFIX("StokesSimulator");

  MeshFlagOperator<dim, value_type> flag_op(this->mesh());

  // We use a refinement scheme that is based on pressure.
  Vector<float> error_per_cell(this->mesh().n_active_cells());

  FEValuesExtractors::Scalar pressure(dim);

  KellyErrorEstimator<dim>::estimate(
    this->dof_handler(),
    QGauss<dim - 1>(this->fe_degree(SolutionComponent::pressure) + 1),
    std::map<dealii::types::boundary_id, const Function<dim>*>(),
    this->get_solution_vector(), error_per_cell,
    this->get_fe().component_mask(pressure));

  felspa_log << "Refinement fraction = " << mesh_control.refine_top_fraction
             << "; coarsen fraction = " << mesh_control.coarsen_bottom_fraction
             << "; using quadrature degree "
             << this->fe_degree(SolutionComponent::pressure) + 1 << std::endl;

  GridRefinement::refine_and_coarsen_fixed_number(
    this->mesh(), error_per_cell, mesh_control.refine_top_fraction,
    mesh_control.coarsen_bottom_fraction);

  flag_op.prioritize_refinement();
  flag_op.limit_level(mesh_control.min_level, mesh_control.max_level);
#ifdef DEBUG
  flag_op.print_info(felspa_log);
#endif  // DEBUG //
}


template <int dim, typename NumberType, typename LinsysType>
void StokesSimulatorBase<dim, NumberType, LinsysType>::export_solution(
  ExportFile& file) const
{
  using namespace dealii;
  const std::string v_label = this->get_label_string() + "Velo";
  const std::string p_label = this->get_label_string() + "Pres";

  std::vector<std::string> solution_names(dim, v_label);
  solution_names.push_back(p_label);

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(this->dof_handler());
  data_out.add_data_vector(*(this->solution), solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);
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

// reassemble the material parameter output
#if 0
  if (ptr_material_parameter_exporter != nullptr) {
    ptr_material_parameter_exporter->update_parameters(this->get_quadrature());
    ptr_material_parameter_exporter->export_parameters();
  }
#endif
}


/* ************************************************** */
/**             StokesSolutionTransfer                */
/* ************************************************** */
template <int dim, typename NumberType, typename LinsysType>
StokesSimulatorBase<dim, NumberType, LinsysType>::StokesSolutionTransfer::
  StokesSolutionTransfer(
    StokesSimulatorBase<dim, NumberType, LinsysType>& stokes)
  : ptr_simulator(&stokes),
    ptr_dof_handler(&stokes.get_dof_handler()),
    ptr_soln(&stokes.solution),
    soln_transfer(stokes.get_dof_handler())
{}


template <int dim, typename NumberType, typename LinsysType>
void StokesSimulatorBase<dim, NumberType, LinsysType>::StokesSolutionTransfer::
  prepare_for_coarsening_and_refinement()
{
  if (!ptr_simulator->is_initialized() || !ptr_soln->is_independent()) return;
#if !defined(FELSPA_HAS_MPI)
  soln_transfer.clear();
#endif
  soln_transfer.prepare_for_coarsening_and_refinement(**ptr_soln);
}


template <int dim, typename NumberType, typename LinsysType>
void StokesSimulatorBase<dim, NumberType,
                         LinsysType>::StokesSolutionTransfer::interpolate()
{
  if (!ptr_simulator->is_initialized() || !ptr_soln->is_independent()) return;

  // allocate the solution vector
  vector_type interpolated_soln;
  ptr_simulator->allocate_solution_vector(interpolated_soln);

#ifdef FELSPA_HAS_MPI
  soln_transfer.interpolate(interpolated_soln);
#else
  soln_transfer.interpolate(**ptr_soln, interpolated_soln);
#endif

  (*ptr_soln)->swap(interpolated_soln);
}


/* **************************************************** */
/** VelocityExtractor<StokesSimulator<dim, NumberType>> */
/* **************************************************** */
template <int dim, typename NumberType, typename LinsysType>
void VelocityExtractor<StokesSimulator<dim, NumberType, LinsysType>,
                       false>::extract(const simulator_type& simulator,
                                       const dealii::FEValuesBase<dim>& feval,
                                       std::vector<tensor_type>& velocities)
  const
{
  const dealii::FEValuesExtractors::Vector velo(0);
  feval[velo].get_function_values(simulator.get_solution_vector(), velocities);
}


/* ************************************************** */
/**              StokesAssemblerBase                  */
/* ************************************************** */
template <typename LinsysType>
void StokesAssemblerBase<LinsysType>::initialize(
  const bcs_type& bcs,
  const std::shared_ptr<const MaterialBase<dim, value_type>>& p_material,
  const std::shared_ptr<source_type>& p_source_term)
{
  ptr_bcs = &bcs;
  ptr_material = p_material;
  ptr_momentum_source = p_source_term;
}


template <typename LinsysType>
FELSPA_FORCE_INLINE auto
StokesAssemblerBase<LinsysType>::preconditioner_matrix() -> matrix_type&
{
  return this->ptr_linear_system->preconditioner_matrix;
}


template <typename LinsysType>
FELSPA_FORCE_INLINE auto
StokesAssemblerBase<LinsysType>::preconditioner_matrix() const
  -> const matrix_type&
{
  return this->ptr_linear_system->preconditioner_matrix;
}


template <typename LinsysType>
void StokesAssemblerBase<LinsysType>::assemble(
  const dealii::Quadrature<dim>& quadrature)
{
  LOG_PREFIX("StokesAssembler");
  felspa_log << "Using " << quadrature.size() << "-point quadrature rule."
             << std::endl;
  felspa_log << "Material type is " << FELSPA_DEMANGLE(*ptr_material)
             << std::endl;

  this->ptr_linear_system->zero_out(true, true, true);

  ScratchData scratch_data(this->dof_handler().get_fe(), this->get_mapping(),
                           quadrature, default_update_flags,
                           this->ptr_material);

  CopyData copy_data(this->dof_handler().get_fe());
  try {
    dealii::WorkStream::run(
      this->dof_handler().begin_active(), this->dof_handler().end(), *this,
      &StokesAssemblerBase<LinsysType>::local_assembly,
      &StokesAssemblerBase<LinsysType>::copy_local_to_global, scratch_data,
      copy_data);
  }
  catch (const std::exception& exc) {
    std::string str("Exception in StokesAssembler: ");
    str += exc.what();
    THROW(EXCEPT_MSG(str));
  }
}


template <typename LinsysType>
void StokesAssemblerBase<LinsysType>::copy_local_to_global(const CopyData& copy)
{
  // assemble LHS and RHS
  this->constraints().distribute_local_to_global(
    copy.local_matrix, copy.local_rhs, copy.local_dof_indices, this->matrix(),
    this->rhs());
  // assemble the preconditioner matrix
  this->constraints().distribute_local_to_global(copy.local_preconditioner,
                                                 copy.local_dof_indices,
                                                 this->preconditioner_matrix());
}

FELSPA_NAMESPACE_CLOSE
#endif  // _FELSPA_PDE_STOKES_COMMON_IMPLEMENT_H_
