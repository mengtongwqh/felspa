#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <felspa/pde/stokes_ilu.h>

FELSPA_NAMESPACE_OPEN


template <int dim, typename NumberType>
BlockSchurPreconditioner<dim, NumberType>::BlockSchurPreconditioner(
  const matrix_type& stokes_matrix, const matrix_type& stokes_precond_matrix,
  A_preconditioner_type& A_precond, S_preconditioner_type& S_precond,
  const std::shared_ptr<A_preconditioner_control>& sp_A_control,
  const std::shared_ptr<S_preconditioner_control>& sp_S_control)
  : base_type(stokes_matrix, stokes_precond_matrix.block(1, 1), A_precond,
              S_precond),
    ptr_preconditioner_matrix(&stokes_precond_matrix),
    ptr_A_preconditioner(&A_precond),
    ptr_S_preconditioner(&S_precond),
    ptr_A_control(sp_A_control),
    ptr_S_control(sp_S_control)
{}


template <int dim, typename NumberType>
void BlockSchurPreconditioner<dim, NumberType>::reinitialize()
{
  // resize the tmp vector
  this->tmp.reinit(this->ptr_matrix->block(1, 1).m());
  base_type::initialize(ptr_preconditioner_matrix->block(1, 1),
                        *ptr_A_preconditioner, *ptr_S_preconditioner);
  ptr_A_preconditioner->initialize(this->ptr_matrix->block(0, 0),
                                   *ptr_A_control);
  ptr_S_preconditioner->initialize(this->ptr_preconditioner_matrix->block(1, 1),
                                   *ptr_S_control);
}

template <int dim, typename NumberType>
[[nodiscard]] auto
BlockSchurPreconditioner<dim, NumberType>::get_preconditioner_matrix() const
  -> const matrix_type&
{
  ASSERT(ptr_preconditioner_matrix != nullptr, ExcNullPointer());
  return *ptr_preconditioner_matrix;
}


/* ************************************************** *
 * StokesSimulator<dim, NumberType,
 *                 StokesLinearSystem>
 * ************************************************** */
template <int dim, typename NumberType>
StokesSimulator<dim, NumberType, StokesLinearSystem<dim, NumberType>>::
  StokesSimulator(Mesh<dim, NumberType>& mesh, unsigned int degree_v,
                  unsigned int degree_p, const std::string& label)
  : base_type(mesh, degree_v, degree_p, label)
{
  this->ptr_control = std::make_shared<Control>();
  this->linear_system().set_control_parameters(control().ptr_A_preconditioner,
                                               control().ptr_S_preconditioner,
                                               control().solution_method);
}


template <int dim, typename NumberType>
StokesSimulator<dim, NumberType, StokesLinearSystem<dim, NumberType>>::
  StokesSimulator(Mesh<dim, NumberType>& mesh,
                  const dealii::FiniteElement<dim>& fe_v,
                  const dealii::FiniteElement<dim>& fe_p,
                  const std::string& label)
  : base_type(mesh, fe_v, fe_p, label)
{
  this->ptr_control = std::make_shared<Control>();
  // Control& ctrl = static_cast<Control&>(*this->ptr_control);
  this->linear_system().set_control_parameters(control().ptr_A_preconditioner,
                                               control().ptr_S_preconditioner,
                                               control().solution_method);
}


template <int dim, typename NumberType>
FELSPA_FORCE_INLINE auto
StokesSimulator<dim, NumberType, StokesLinearSystem<dim, NumberType>>::control()
  -> Control&
{
  ASSERT(this->ptr_control, ExcNullPointer());
  return static_cast<Control&>(*this->ptr_control);
}

template <int dim, typename NumberType>
FELSPA_FORCE_INLINE auto StokesSimulator<
  dim, NumberType, StokesLinearSystem<dim, NumberType>>::get_control() const
  -> const Control&
{
  ASSERT(this->ptr_control, ExcNullPointer());
  return static_cast<const Control&>(*this->ptr_control);
}


template <int dim, typename NumberType>
FELSPA_FORCE_INLINE void
StokesSimulator<dim, NumberType, StokesLinearSystem<dim, NumberType>>::
  attach_control(const std::shared_ptr<Control>& sp_control)
{
  ASSERT(sp_control != nullptr, ExcNullPointer());
  this->ptr_control = sp_control;
  this->linear_system().set_control_parameters(control().ptr_A_preconditioner,
                                               control().ptr_S_preconditioner,
                                               sp_control->solution_method);
}


template <int dim, typename NumberType>
void StokesSimulator<dim, NumberType, StokesLinearSystem<dim, NumberType>>::
  allocate_solution_vector(vector_type& vect)
{
  // we make the assumption that
  // the dof count in the linear system is already recomputed.
  vect.reinit(2);
  vect.block(0).reinit(this->linear_system().get_block_ndofs(0));
  vect.block(1).reinit(this->linear_system().get_block_ndofs(1));
  vect.collect_sizes();
}


/* ************************************************** *
 * StokesSimulator<dim, NumberType,                   *
 *   StokesLinearSystem<dim, NumberType>>::Control    *
 * ************************************************** */
template <int dim, typename NumberType>
StokesSimulator<dim, NumberType,
                StokesLinearSystem<dim, NumberType>>::Control::Control()
  : ptr_A_preconditioner(std::make_shared<A_preconditioner_control_type>()),
    ptr_S_preconditioner(std::make_shared<S_preconditioner_control_type>())
{}


#ifdef FELSPA_STOKES_USE_CUSTOM_ILU
template <int dim, typename NumberType>
void StokesSimulator<dim, NumberType, StokesLinearSystem<dim, NumberType>>::
  Control::set_level_of_fill(unsigned int A_lof, unsigned int S_lof)
{
  ptr_A_preconditioner->max_level_of_fill = A_lof;
  ptr_S_preconditioner->max_level_of_fill = S_lof;
}
#endif  // FELSPA_STOKES_USE_CUSTOM_ILU //


/* ************************************************** */
/**               StokesLinearSystem                  */
/* ************************************************** */
template <int dim, typename NumberType>
StokesLinearSystem<dim, NumberType>::StokesLinearSystem(
  const dealii::DoFHandler<dim>& dofh)
  : base_type(dofh),
    ptr_A_preconditioner(std::make_unique<APreconditioner>()),
    ptr_S_preconditioner(std::make_unique<SPreconditioner>()),
    ptr_scaling_operator(std::make_unique<DiagonalScalingOp>())
{}


template <int dim, typename NumberType>
StokesLinearSystem<dim, NumberType>::StokesLinearSystem(
  const dealii::DoFHandler<dim>& dofh, const dealii::Mapping<dim>& mapping)
  : base_type(dofh, mapping),
    ptr_A_preconditioner(std::make_unique<APreconditioner>()),
    ptr_S_preconditioner(std::make_unique<SPreconditioner>()),
    ptr_scaling_operator(std::make_unique<DiagonalScalingOp>())
{}


template <int dim, typename NumberType>
FELSPA_FORCE_INLINE auto
StokesLinearSystem<dim, NumberType>::get_preconditioner_matrix() const
  -> const matrix_type&
{
  return preconditioner_matrix;
}


template <int dim, typename NumberType>
void StokesLinearSystem<dim, NumberType>::populate_system_from_dofs()
{
  base_type::populate_system_from_dofs();
  // allocate space for outer preconditioner
  make_preconditioner_sparsity();
}


template <int dim, typename NumberType>
void StokesLinearSystem<dim, NumberType>::upon_mesh_update()
{
  count_dofs();
  this->flag_constraints_for_update();
  ptr_A_preconditioner_control->use_previous_sparsity = false;
  ptr_S_preconditioner_control->use_previous_sparsity = false;
}


template <int dim, typename NumberType>
void StokesLinearSystem<dim, NumberType>::count_dofs(bool count_component,
                                                     bool count_block)
{
  base_type::count_dofs(count_component, false);

  if (count_block) {
    std::vector<types::SizeType> block_component(dim + 1, 0);
    block_component[dim] = 1;
    this->ndofs_per_block = dealii::DoFTools::count_dofs_per_fe_block(
      this->get_dof_handler(), block_component);
  }
}


template <int dim, typename NumberType>
void StokesLinearSystem<dim, NumberType>::zero_out(bool zero_lhs,
                                                   bool zero_rhs,
                                                   bool zero_precond)
{
  base_type::zero_out(zero_lhs, zero_rhs);
  if (zero_precond) preconditioner_matrix.reinit(preconditioner_sparsity);
}


template <int dim, typename NumberType>
void StokesLinearSystem<dim, NumberType>::setup_constraints_and_system(
  const BCBookKeeper<dim, NumberType>& bcs)
{
  using namespace dealii;
  using this_type = StokesLinearSystem<dim, NumberType>;
  dealii::BlockDynamicSparsityPattern matrix_dsp(2, 2);
  dealii::BlockDynamicSparsityPattern preconditioner_dsp(2, 2);


  // setting up the constraints
  Threads::Task<void> task_setup_constraints =
    Threads::new_task(&this_type::setup_constraints, *this, std::cref(bcs));

  // allocate the solution vector
  this->rhs.reinit(2);
  this->rhs.block(0).reinit(this->ndofs_per_block[0]);
  this->rhs.block(1).reinit(this->ndofs_per_block[1]);
  this->rhs.collect_sizes();

  // sync constraints
  task_setup_constraints.join();

  // setting up the linear system sparsity
  Threads::Task<void> task_preconditioner_sparsity =
    Threads::new_task(&this_type::make_preconditioner_sparsity, *this);
  this->make_matrix_sparsity();
  task_preconditioner_sparsity.join();

  // set status
  this->populated = true;
}


template <int dim, typename NumberType>
void StokesLinearSystem<dim, NumberType>::make_preconditioner_sparsity()
{
  using namespace dealii;
  LOG_PREFIX("MixedVPLinearSystem");
  ASSERT(this->ptr_dof_handler->has_active_dofs(), ExcDoFHandlerNotInit());
  ASSERT(this->constraints_updated,
         EXCEPT_MSG("Constraints need to be updated before populating system. "
                    "Call setup_constraints()"));
  felspa_log << "Populating the preconditioner matrix ..." << std::endl;

  const auto ndof_u(this->ndofs_per_block[0]), ndof_p(this->ndofs_per_block[1]);

  dealii::BlockDynamicSparsityPattern dsp(2, 2);
  dsp.block(0, 0).reinit(ndof_u, ndof_u);
  dsp.block(0, 1).reinit(ndof_u, ndof_p);
  dsp.block(1, 0).reinit(ndof_p, ndof_u);
  dsp.block(1, 1).reinit(ndof_p, ndof_p);
  dsp.collect_sizes();

  Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
  for (unsigned int c = 0; c < dim + 1; ++c)
    for (unsigned int d = 0; d < dim + 1; ++d)
      if (c == dim && d == dim)
        coupling[c][d] = DoFTools::always;
      else
        coupling[c][d] = DoFTools::none;

  DoFTools::make_sparsity_pattern(this->get_dof_handler(), coupling, dsp,
                                  this->constraints, false);
  this->preconditioner_sparsity.copy_from(dsp);
  this->preconditioner_matrix.reinit(this->preconditioner_sparsity);
}


template <int dim, typename NumberType>
void StokesLinearSystem<dim, NumberType>::set_control_parameters(
  const std::shared_ptr<APreconditionerControl>& p_A_control,
  const std::shared_ptr<SPreconditionerControl>& p_S_control,
  const StokesSolutionMethod& solution_method)
{
  ASSERT(p_A_control != nullptr, ExcNullPointer());
  ASSERT(p_S_control != nullptr, ExcNullPointer());

  ptr_A_preconditioner_control = p_A_control;
  ptr_S_preconditioner_control = p_S_control;
  ptr_solution_method = &solution_method;
}


template <int dim, typename NumberType>
void StokesLinearSystem<dim, NumberType>::setup_bc_constraints(
  const BCBookKeeper<dim, value_type>& bcs)
{
  // dirichlet boundary conditions
  dealii::FEValuesExtractors::Vector velocities(0);

  // impose Dirichlet boundary condition for velocity
  if (bcs.has_category(BCCategory::dirichlet)) {
    const auto& drchlt_bcs = bcs(BCCategory::dirichlet);

    for (const auto& pbc : drchlt_bcs) {
      bool mask_test = false;
      for (int idim = 0; idim < dim; ++idim)
        mask_test |= pbc->get_component_mask()[idim];
      if (!mask_test) continue;

#ifdef DEBUG
      auto pbc_fcn = dynamic_cast<const BCFunction<dim, value_type>*>(pbc);
      ASSERT(pbc_fcn != nullptr, ExcNullPointer());
#else
      auto pbc_fcn = static_cast<const BCFunction<dim, value_type>*>(pbc);
#endif  // DEBUG //

      auto bdry_id = pbc_fcn->get_boundary_id();

      dealii::VectorTools::interpolate_boundary_values(
        this->get_dof_handler(),
        bdry_id,
        *pbc_fcn,
        this->constraints,
        pbc_fcn->get_component_mask());
    }
  }

  // periodic boundary conditions
  if (bcs.has_category(BCCategory::periodic)) {
    const auto& periodic_bcs = bcs(BCCategory::periodic);

    for (const auto& pbc : periodic_bcs) {
#ifdef DEBUG
      auto pbc_periodic =
        dynamic_cast<const PeriodicBCFunction<dim, value_type>*>(pbc);
      ASSERT(pbc_periodic != nullptr, ExcNullPointer());
#else
      auto pbc_periodic =
        static_cast<const PeriodicBCFunction<dim, value_type>*>(pbc);
#endif  // DEBUG //

      // get the periodicity vector
      auto periodicity_vector =
        pbc_periodic->collect_periodic_faces(this->get_dof_handler());

      // impose the periodic constraints
      dealii::DoFTools::make_periodicity_constraints<dim, dim, value_type>(
        periodicity_vector, this->constraints, pbc->get_component_mask());
    }  // loop over all periodic bcs
  }

  // no normal flux boundary constraints
  if (bcs.has_category(BCCategory::no_normal_flux)) {
    const auto& no_normal_flux_bcs = bcs(BCCategory::no_normal_flux);
    std::set<dealii::types::boundary_id> bdry_ids;

    for (const auto& pbc : no_normal_flux_bcs) {
#ifdef DEBUG
      auto pbc_fcn = dynamic_cast<const BCFunction<dim, value_type>*>(pbc);
      ASSERT(pbc_fcn != nullptr, ExcNullPointer());
#else
      auto pbc_fcn = static_cast<const BCFunction<dim, value_type>*>(pbc);
#endif  // DEBUG /

      if (auto [it, status] = bdry_ids.insert(pbc_fcn->get_boundary_id());
          status == false)
        THROW(EXCEPT_MSG("Boundary condition cannot be inserted twice"));
    }

    dealii::VectorTools::compute_no_normal_flux_constraints(
      this->get_dof_handler(), 0, bdry_ids, this->constraints,
      this->get_mapping());
  }


  // apply mean pressure constraints
  dealii::ComponentMask pressure_mask(dim + 1, false);
  pressure_mask.set(dim, true);
  dealii::IndexSet pdofs =
    dealii::DoFTools::extract_dofs(this->get_dof_handler(), pressure_mask);
  const auto first_pdof = pdofs.nth_index_in_set(0);
  this->constraints.add_line(first_pdof);
  // for (auto i : pdofs) {
  //   if (i != first_pdof) this->constraints.add_entry(first_pdof, i, -1);
  // }
}


template <int dim, typename NumberType>
void StokesLinearSystem<dim, NumberType>::apply_pressure_scaling(
  NumberType ref_viscosity, NumberType ref_length)
{
  const value_type pressure_scaling = ref_viscosity / ref_length;
  this->matrix.block(1, 0) *= ref_viscosity / ref_length;
  this->matrix.block(0, 1) *= ref_viscosity / ref_length;
  preconditioner_matrix.block(1, 1) *=
    (1.0 / ref_viscosity * pressure_scaling * pressure_scaling);
}


template <int dim, typename NumberType>
void StokesLinearSystem<dim, NumberType>::pre_solve_scaling(
  vector_type& soln, vector_type& rhs, bool apply_scaling_to_matrix)
{
  // scale the lhs matrix and the preconditioner matrix
  // ptr_scaling_operator->initialize(this->matrix);
  if (apply_scaling_to_matrix) {
    ptr_scaling_operator->apply_to_matrix(this->matrix);
    ptr_scaling_operator->apply_to_preconditioner(
      this->preconditioner_matrix.block(1, 1));
  }
  ptr_scaling_operator->apply_inverse_to_vector(soln);
  ptr_scaling_operator->apply_to_vector(rhs);
}


template <int dim, typename NumberType>
void StokesLinearSystem<dim, NumberType>::post_solve_scaling(vector_type& soln)
{
  ptr_scaling_operator->apply_to_vector(soln);
}


template <int dim, typename NumberType>
void StokesLinearSystem<dim, NumberType>::solve(
  vector_type& soln, vector_type& rhs, dealii::SolverControl& solver_control)
{
  ptr_scaling_operator->allocate(this->matrix);
  ptr_scaling_operator->initialize(this->matrix);
  switch (*ptr_solution_method) {
    case FC:
      solve_gmres(soln, rhs, solver_control);
      break;
    case SCR:
      solve_cg(soln, rhs, solver_control);
      break;
    case CompareTest:
      solve_compare_test(soln, rhs, solver_control);
      break;
    default:
      THROW(ExcNotImplemented());
  }
}


template <int dim, typename NumberType>
void StokesLinearSystem<dim, NumberType>::solve(
  vector_type& soln, dealii::SolverControl& solver_control)
{
  solve(soln, this->rhs, solver_control);
}


template <int dim, typename NumberType>
void StokesLinearSystem<dim, NumberType>::solve_compare_test(
  vector_type& soln, vector_type& rhs, dealii::SolverControl& solver_control)
{
  felspa_log << "Running CG and GMRES for comparison..." << std::endl;
  auto& control = dynamic_cast<StokesSolverControl&>(solver_control);

  // scale the matrix and preconditioner
  if (control.apply_diagonal_scaling) {
    ptr_scaling_operator->apply_to_matrix(this->matrix);
    ptr_scaling_operator->apply_to_preconditioner(
      this->preconditioner_matrix.block(1, 1));
  }

  vector_type soln_gmres = soln;

  felspa_log << std::endl;
  felspa_log << " >>> CG Schur Complement Reduction <<<" << std::endl;
  solve_cg(soln, rhs, solver_control, false);

  if (control.apply_diagonal_scaling)
    ptr_scaling_operator->apply_inverse_to_vector(rhs);

  felspa_log << std::endl;
  felspa_log << " >>> GMRES block solver <<<" << std::endl;
  solve_gmres(soln_gmres, rhs, solver_control, false);

  // compare solution from both solvers
  soln_gmres -= soln;

  control.soln_diff_l2.push_back(soln_gmres.l2_norm() / soln.l2_norm());
  control.soln_diff_linfty.push_back(soln_gmres.linfty_norm());
  felspa_log << "Inf-norm between soln: " << soln_gmres.linfty_norm()
             << " and l2-norm: " << soln_gmres.l2_norm() << std::endl;
}


template <int dim, typename NumberType>
void StokesLinearSystem<dim, NumberType>::solve_gmres(
  vector_type& soln, vector_type& rhs, dealii::SolverControl& solver_control,
  bool apply_scaling_to_matrix)
{
  LOG_PREFIX(FELSPA_DEMANGLE(*this));
  auto& control = dynamic_cast<StokesSolverControl&>(solver_control);

  dealii::Timer timer;
  timer.start();

  // scale the matrix
  if (control.apply_diagonal_scaling)
    this->pre_solve_scaling(soln, rhs, apply_scaling_to_matrix);

  for (decltype(soln.size()) i = 0; i < soln.size(); ++i)
    if (this->constraints.is_constrained(i)) soln[i] = 0;

  // allocate the block preconditioner
  BlockSchurPreconditioner<dim, NumberType> block_precond(
    this->get_matrix(), this->get_preconditioner_matrix(),
    *ptr_A_preconditioner, *ptr_S_preconditioner, ptr_A_preconditioner_control,
    ptr_S_preconditioner_control);
  block_precond.reinitialize();

  solver_control.set_tolerance(1.0e-8 * rhs.l2_norm());
  solver_control.set_max_steps(rhs.size());
  felspa_log << "Solver tolerance is set to " << solver_control.tolerance()
             << std::endl;
  typename dealii::SolverGMRES<vector_type>::AdditionalData gmres_control;
  gmres_control.max_n_tmp_vectors = 25;


  dealii::SolverGMRES<vector_type> solver(solver_control, gmres_control);
  solver.solve(this->matrix, soln, rhs, block_precond);

  if (control.apply_diagonal_scaling) this->post_solve_scaling(soln);
  this->constraints.distribute(soln);

  ptr_A_preconditioner_control->use_previous_sparsity = true;
  ptr_S_preconditioner_control->use_previous_sparsity = true;

  timer.stop();

  // record solver statistics
  control.n_gmres_iter.push_back(solver_control.last_step());
  control.gmres_error.push_back(solver_control.last_value());
  control.gmres_timer.push_back(timer.wall_time());
  control.log_gmres = true;

  felspa_log << FELSPA_DEMANGLE(solver) << " converged in "
             << solver_control.last_step() << " steps to value "
             << solver_control.last_value() << std::endl;
}


template <int dim, typename NumberType>
void StokesLinearSystem<dim, NumberType>::solve_cg(
  vector_type& soln, vector_type& rhs, dealii::SolverControl& schur_control,
  bool apply_scaling_to_matrix)
{
  ASSERT(ptr_A_preconditioner_control != nullptr &&
           ptr_S_preconditioner_control != nullptr,
         EXCEPT_MSG(
           "Control is not attached. Call attach_preconditioner_control()"));
  ASSERT_SAME_SIZE(soln.block(0), rhs.block(0));
  ASSERT_SAME_SIZE(soln.block(1), rhs.block(1));

  LOG_PREFIX("StokeLinearSystem");
  felspa_log << ">>>>> Solving Linear System <<<<<" << std::endl;

  auto& control = dynamic_cast<StokesSolverControl&>(schur_control);

  dealii::Timer timer;
  timer.start();

  if (control.apply_diagonal_scaling)
    pre_solve_scaling(soln, rhs, apply_scaling_to_matrix);

  // Preconditioner for A^{-1}
  auto init_A_precond = dealii::Threads::new_task([&]() {
    this->ptr_A_preconditioner->initialize(this->matrix.block(0, 0),
                                           *ptr_A_preconditioner_control);
  });

  auto init_S_precond = dealii::Threads::new_task([&]() {
    this->ptr_S_preconditioner->initialize(
      this->preconditioner_matrix.block(1, 1), *ptr_S_preconditioner_control);
  });


#ifdef EXPORT_MATRIX
  std::ofstream VMatrix("VelocityMatrix.svg");
  std::ofstream VPrecondMatrix("VelocityPreconditionMatrix.svg");
  this->matrix.block(0, 0).get_sparsity_pattern().print_svg(VMatrix);
  ptr_A_preconditioner->print_sparsity_svg(VPrecondMatrix);
  VMatrix.close();
  VPrecondMatrix.close();
#endif  // EXPORT MATRIX //

  // A^{-1}
  using A_inverse_type = InverseMatrixCG<matrix_block_type, APreconditioner>;
  A_inverse_type A_inverse(this->matrix.block(0, 0),
                           *this->ptr_A_preconditioner, 1.0e-15, 1.0e-15);

  vector_block_type tmp(rhs.block(0).size());

  {
    // --------------------------------- //
    // SETTING UP SCHUR-COMPLEMENTED RHS //
    // --------------------------------- //
    felspa_log << "Computing the RHS for Schur complement system..."
               << std::endl;

    SaddlePointSchurComplement<matrix_type, A_inverse_type> schur_complement(
      this->matrix, A_inverse);

    vector_block_type schur_rhs(rhs.block(1).size());
    init_A_precond.join();
    A_inverse.vmult(tmp, rhs.block(0));

    // simply use this as an iteration count
    control.n_cg_inner_iter.push_back(A_inverse.get_control().last_step());

    this->matrix.block(1, 0).vmult(schur_rhs, tmp);
    schur_rhs -= rhs.block(1);


    ASSERT_SOLVER_CONVERGED(A_inverse.get_control());
    felspa_log << " A^{-1} converged in " << A_inverse.get_control().last_step()
               << " steps to a value of "
               << A_inverse.get_control().last_value() << std::endl;

    // ------------------ //
    // SOLVE FOR PRESSURE //
    // ------------------ //
    felspa_log << "Computing Schur complement inverse to get pressure..."
               << std::endl;

    InverseMatrixCG<matrix_block_type, SPreconditioner> S_inverse(
      this->preconditioner_matrix.block(1, 1), *this->ptr_S_preconditioner);

#ifdef PROFILING
    schur_control.enable_history_data();
#endif

    schur_control.set_tolerance(1.0e-8 * schur_rhs.l2_norm());
    // schur_control.set_tolerance(1.0e-8);
    felspa_log << "Schur complement tolerance is set to "
               << schur_control.tolerance() << std::endl;
    schur_control.set_max_steps(schur_rhs.size());
    dealii::SolverCG<> schur_cg(schur_control);
    init_S_precond.join();
    schur_cg.solve(schur_complement, soln.block(1), schur_rhs, S_inverse);

#ifdef PROFILING
    for (auto i : schur_control.get_history_data()) std::cout << i << ", ";
    std::cout << std::endl;
#endif  // PROFILING

    ASSERT_SOLVER_CONVERGED(schur_control);
    felspa_log << "Schur complement pressure solve converged after "
               << schur_control.last_step() << " steps to a value of "
               << schur_control.last_value() << std::endl;


    if (control.apply_diagonal_scaling)
      ptr_scaling_operator->apply_to_vector(soln);

    this->constraints.distribute(soln);

    if (control.apply_diagonal_scaling)
      ptr_scaling_operator->apply_inverse_to_vector(soln);
  }

  {
    // ------------------ //
    // SOLVE_FOR_VELOCITY //
    // ------------------ //
    felspa_log << "Computing velocity from pressure solution..." << std::endl;
    this->matrix.block(0, 1).vmult(tmp, soln.block(1));
    tmp *= -1.0;
    tmp += rhs.block(0);
    A_inverse.vmult(soln.block(0), tmp);

    ASSERT_SOLVER_CONVERGED(A_inverse.get_control());
    felspa_log << "Velocity solve converged in "
               << A_inverse.get_control().last_step() << " steps" << std::endl;

    if (control.apply_diagonal_scaling)
      ptr_scaling_operator->apply_to_vector(soln);
    this->constraints.distribute(soln);
  }

  ptr_A_preconditioner_control->use_previous_sparsity = true;
  ptr_S_preconditioner_control->use_previous_sparsity = true;

  timer.stop();
  control.n_cg_outer_iter.push_back(schur_control.last_step());
  control.cg_error.push_back(schur_control.last_value());
  control.cg_timer.push_back(timer.wall_time());
  control.log_cg = true;
}


template <int dim, typename NumberType>
void StokesLinearSystem<dim, NumberType>::DiagonalScalingOp::allocate(
  const matrix_type& matrix)
{
  // allocate sizes
  size_type v_block_size = matrix.block(0, 0).m();
  size_type p_block_size = matrix.block(1, 1).m();

  this->scaling_coeffs.reinit(2);
  this->scaling_coeffs.block(0).reinit(v_block_size);
  this->scaling_coeffs.block(1).reinit(p_block_size);
  this->scaling_coeffs.collect_sizes();
}


/* ----------------------------------------------------- */
/* StokesAssembler <StokesLinearSystem<dim, NumberType>> */
/* ----------------------------------------------------- */
template <int dim, typename NumberType>
StokesAssembler<StokesLinearSystem<dim, NumberType>>::StokesAssembler(
  linsys_type& linsys, bool construct_mapping_adhoc)
  : StokesAssemblerBase<StokesLinearSystem<dim, NumberType>>(
      linsys, construct_mapping_adhoc)
{}


template <int dim, typename NumberType>
void StokesAssembler<StokesLinearSystem<dim, NumberType>>::local_assembly(
  const active_cell_iterator& cell, ScratchData& scratch, CopyData& copy)
{
  const auto& p_source = this->ptr_momentum_source;

  scratch.reinit(cell);

  cell->get_dof_indices(copy.local_dof_indices);

  const dealii::FEValuesExtractors::Vector velocities(0);
  const dealii::FEValuesExtractors::Scalar pressure(dim);

  const auto ndof = scratch.fe_values.dofs_per_cell;
  const auto nqpt = scratch.fe_values.n_quadrature_points;

  copy.local_matrix = 0.0;
  copy.local_preconditioner = 0.0;
  copy.local_rhs = 0.0;

  using ndof_type = typename std::remove_const<decltype(ndof)>::type;
  using nqpt_type = typename std::remove_const<decltype(nqpt)>::type;

  // This is the way to compute material parameters
  scratch.ptr_material_accessor->eval_scalars(
    MaterialParameter::viscosity, scratch.pts_field, scratch.viscosity);

  if (p_source) {
    scratch.ptr_material_accessor->eval_scalars(
      MaterialParameter::density,
      scratch.pts_field,
      scratch.density);  // density model
    for (unsigned int iq = 0; iq < nqpt; iq++)
      scratch.source[iq] =
        (*p_source)((*scratch.pts_field.ptr_pts)[iq]) * scratch.density[iq];
  }

  // assembly loop
  for (nqpt_type iq = 0; iq < nqpt; ++iq) {
    // cache shape fcn/grad values
    for (ndof_type i = 0; i < ndof; ++i) {
      scratch.sym_grad_v[i] =
        scratch.fe_values[velocities].symmetric_gradient(i, iq);
      scratch.div_v[i] = scratch.fe_values[velocities].divergence(i, iq);
      scratch.p[i] = scratch.fe_values[pressure].value(i, iq);
    }
    if (p_source)
      for (ndof_type i = 0; i < ndof; ++i)
        scratch.v[i] = scratch.fe_values[velocities].value(i, iq);

    for (ndof_type i = 0; i < ndof; ++i) {
      // LHS Stokes system //
      // assemble lower part of the local matrices, including diagonal
      for (ndof_type j = 0; j <= i; ++j) {
        copy.local_matrix(i, j) +=
          (2.0 * (scratch.sym_grad_v[i] * scratch.sym_grad_v[j]) *
             scratch.viscosity[iq] -
           scratch.div_v[i] * scratch.p[j] - scratch.p[i] * scratch.div_v[j]) *
          scratch.fe_values.JxW(iq);

        // preconditioner matrix, mass matrix in pressure
        copy.local_preconditioner(i, j) += 1.0 / scratch.viscosity[iq] *
                                           scratch.p[i] * scratch.p[j] *
                                           scratch.fe_values.JxW(iq);
      }  // j-loop

      // RHS: source and Neumann boundary //
      // TODO RHS: Neumann boundary conditions
      // source_term
      copy.local_rhs[i] += (p_source ? scratch.source[iq] * scratch.v[i] *
                                         scratch.fe_values.JxW(iq)
                                     : 0.0);
    }  // i-loop

    // upper part of local matrices: use symmetry
    for (ndof_type i = 0; i < ndof; ++i) {
      for (ndof_type j = i + 1; j < ndof; ++j) {
        copy.local_matrix(i, j) = copy.local_matrix(j, i);
        copy.local_preconditioner(i, j) = copy.local_preconditioner(j, i);
      }  // j-loop
    }    // i-loop
  }      // iq-loop
}


/* -------------------------------------------------- */
template class StokesLinearSystem<2, types::DoubleType>;
template class StokesLinearSystem<3, types::DoubleType>;

template class StokesAssembler<StokesLinearSystem<2, types::DoubleType>>;
template class StokesAssembler<StokesLinearSystem<3, types::DoubleType>>;

template class StokesSimulator<2, types::DoubleType,
                               StokesLinearSystem<2, types::DoubleType>>;
template class StokesSimulator<3, types::DoubleType,
                               StokesLinearSystem<3, types::DoubleType>>;
/* -------------------------------------------------- */

FELSPA_NAMESPACE_CLOSE
