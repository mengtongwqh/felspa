#include <felspa/pde/stokes_trilinos.h>

FELSPA_NAMESPACE_OPEN

#if defined(FELSPA_HAS_MPI) && defined(DEAL_II_WITH_TRILINOS)


/* ************************************************** *
 * StokesSimulator<dim, double,
 *              mpi::trilinos::StokesLinearSystem<dim>>
 * ************************************************** */
template <int dim>
StokesSimulator<dim, types::TrilinosScalar,
                mpi::trilinos::StokesLinearSystem<dim>>::
  StokesSimulator(Mesh<dim, value_type>& mesh, unsigned int degree_v,
                  unsigned int degree_p, const std::string& label,
                  const MPI_Comm& mpi_comm)
  : base_type(mesh, degree_v, degree_p, label), mpi_communicator(mpi_comm)
{
  do_construct();
}


template <int dim>
StokesSimulator<dim, types::TrilinosScalar,
                mpi::trilinos::StokesLinearSystem<dim>>::
  StokesSimulator(Mesh<dim, value_type>& mesh,
                  const dealii::FiniteElement<dim>& fe_v,
                  const dealii::FiniteElement<dim>& fe_p,
                  const std::string& label, const MPI_Comm& mpi_comm)
  : base_type(mesh, fe_v, fe_p, label), mpi_communicator(mpi_comm)
{
  do_construct();
}


template <int dim>
void StokesSimulator<dim, types::TrilinosScalar,
                     mpi::trilinos::StokesLinearSystem<dim>>::
  allocate_solution_vector(vector_type& vect)
{
  vect.reinit(this->linear_system().get_owned_dofs_per_block(),
              mpi_communicator);
}


template <int dim>
auto StokesSimulator<dim, types::TrilinosScalar,
                     mpi::trilinos::StokesLinearSystem<dim>>::control()
  -> Control&
{
  ASSERT(this->ptr_control != nullptr, ExcNullPointer());
  return static_cast<Control&>(*this->ptr_control);
}


template <int dim>
auto StokesSimulator<dim, types::TrilinosScalar,
                     mpi::trilinos::StokesLinearSystem<dim>>::get_control()
  const -> const Control&
{
  ASSERT(this->ptr_control != nullptr, ExcNullPointer());
  return static_cast<const Control&>(*this->ptr_control);
}


template <int dim>
void StokesSimulator<dim, types::TrilinosScalar,
                     mpi::trilinos::StokesLinearSystem<dim>>::
  attach_control(const std::shared_ptr<Control>& sp_control)
{
  ASSERT(sp_control != nullptr, ExcNullPointer());
  this->ptr_control = sp_control;
}


template <int dim>
void StokesSimulator<dim, types::TrilinosScalar,
                     mpi::trilinos::StokesLinearSystem<dim>>::
  solve_linear_system(vector_type& soln, vector_type& rhs)
{
  // allocate the block preconditioner if not allocated yet
  if (ptr_block_preconditioner == nullptr) {
    ptr_block_preconditioner = std::make_shared<
      mpi::BlockSchurPreconditioner<dim, APreconditioner, SPreconditioner>>(
      this->linear_system().get_matrix(),
      this->linear_system().get_preconditioner_matrix(),
      this->get_dof_handler(), control().ptr_precond_A, control().ptr_precond_S,
      mpi_communicator);

    this->linear_system().set_block_preconditioner(ptr_block_preconditioner);

    LOG_PREFIX("StokesSimulatorTrilinos");
    felspa_log << "Block preconditioner allocated." << std::endl;
  }

  base_type::solve_linear_system(soln, rhs);
  // this->linear_system().solve(soln, rhs, *this->ptr_control->ptr_solver);
  felspa_log << "Linear system solution completed." << std::endl;
}


template <int dim>
void StokesSimulator<dim, types::TrilinosScalar,
                     mpi::trilinos::StokesLinearSystem<dim>>::
  solve_linear_system(vector_type& soln)
{
  // allocate the block preconditioner if not allocated yet
  if (ptr_block_preconditioner == nullptr) {
    ptr_block_preconditioner = std::make_shared<
      mpi::BlockSchurPreconditioner<dim, APreconditioner, SPreconditioner>>(
      this->linear_system().get_matrix(),
      this->linear_system().get_preconditioner_matrix(),
      this->get_dof_handler(), control().ptr_precond_A, control().ptr_precond_S,
      mpi_communicator);

    this->linear_system().set_block_preconditioner(ptr_block_preconditioner);

    LOG_PREFIX("StokesSimulatorTrilinos");
    felspa_log << "Block preconditioner allocated." << std::endl;
  }

  base_type::solve_linear_system(soln);
  // this->linear_system().solve(soln, rhs, *this->ptr_control->ptr_solver);
  felspa_log << "Linear system solution completed." << std::endl;
}


template <int dim>
void StokesSimulator<dim, types::TrilinosScalar,
                     mpi::trilinos::StokesLinearSystem<dim>>::do_construct()
{
  this->ptr_control = std::make_shared<Control>();
  this->linear_system().set_mpi_communicator(mpi_communicator);
  this->linear_system().set_additional_control(
    control().ptr_solver_additional_control);
}


/* ************************************************** */
/*           StokesSimulator::Control                 */
/* ************************************************** */
template <int dim>
StokesSimulator<dim, types::TrilinosScalar,
                mpi::trilinos::StokesLinearSystem<dim>>::Control::Control()
  : StokesControlBase<value_type>(),
    ptr_precond_A(std::make_shared<APreconditionerControl>()),
    ptr_precond_S(std::make_shared<SPreconditionerControl>()),
    ptr_solver_additional_control(std::make_shared<SolverAdditionalControl>(30))
{
  ptr_precond_A->elliptic = true;
  ptr_precond_A->higher_order_elements = true;
  ptr_precond_A->smoother_sweeps = 2;
  ptr_precond_A->aggregation_threshold = 0.02;
}


/* ----------------------------------------------------- *
 * StokesAssembler<                                      *
 *   mpi::trilinos::StokesLinearSystem<dim, NumberType>> *
 * ----------------------------------------------------- */
template <int dim>
StokesAssembler<mpi::trilinos::StokesLinearSystem<dim>>::StokesAssembler(
  linsys_type& linsys, bool construct_mapping_adhoc)
  : base_type(linsys, construct_mapping_adhoc)
{}


template <int dim>
void StokesAssembler<mpi::trilinos::StokesLinearSystem<dim>>::local_assembly(
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
      scratch.grad_v[i] = scratch.fe_values[velocities].gradient(i, iq);
      scratch.p[i] = scratch.fe_values[pressure].value(i, iq);
    }

    if (p_source)
      for (ndof_type i = 0; i < ndof; ++i)
        scratch.v[i] = scratch.fe_values[velocities].value(i, iq);

    for (ndof_type i = 0; i < ndof; ++i) {
      // LHS Stokes system //
      // assemble lower part of the local matrices, including diagonal
      for (ndof_type j = 0; j < ndof; ++j) {
        copy.local_matrix(i, j) +=
          (2.0 * (scratch.sym_grad_v[i] * scratch.sym_grad_v[j]) *
             scratch.viscosity[iq] -
           scratch.div_v[i] * scratch.p[j] - scratch.p[i] * scratch.div_v[j]) *
          scratch.fe_values.JxW(iq);

        // preconditioner matrix
        copy.local_preconditioner(i, j) +=
          (scratch.viscosity[iq] *
             dealii::scalar_product(scratch.grad_v[i], scratch.grad_v[j]) +
           1.0 / scratch.viscosity[iq] * scratch.p[i] * scratch.p[j]) *
          scratch.fe_values.JxW(iq);
      }  // j-loop

      // RHS: source and Neumann boundary //
      // TODO RHS: Neumann boundary conditions
      // source_term
      copy.local_rhs[i] += (p_source ? scratch.source[iq] * scratch.v[i] *
                                         scratch.fe_values.JxW(iq)
                                     : 0.0);
    }  // i-loop
  }    // iq-loop

#ifdef DEBUG
  for (auto i : scratch.viscosity)
    ASSERT(!std::isnan(i) && std::isfinite(i),
           ExcUnexpectedValue<value_type>(i));
  for (auto i : copy.local_matrix)
    ASSERT(!std::isnan(i) && std::isfinite(i),
           ExcUnexpectedValue<value_type>(i));
  for (auto i : copy.local_preconditioner)
    ASSERT(!std::isnan(i) && std::isfinite(i),
           ExcUnexpectedValue<value_type>(i));
  for (auto i : copy.local_rhs)
    ASSERT(!std::isnan(i) && std::isfinite(i),
           ExcUnexpectedValue<value_type>(i));
#endif
}

#endif  // FELSPA_HAS_MPI && DEAL_II_WITH_TRILINOS //


#ifdef FELSPA_HAS_MPI
/* ---------- */
namespace mpi
/* ---------- */
{
#ifdef DEAL_II_WITH_TRILINOS
  /* --------------- */
  namespace trilinos
  /* --------------- */
  {
    /* ************************************************** */
    /*                 StokesLinearSystem                 */
    /* ************************************************** */
    template <int dim>
    FELSPA_FORCE_INLINE StokesLinearSystem<dim>::StokesLinearSystem(
      const ::dealii::DoFHandler<dim>& dofh, MPI_Comm mpi_comm)
      : base_type(dofh),
        ptr_scaling_operator(std::make_unique<DiagonalScalingOp>()),
        mpi_communicator(mpi_comm)
    {}


    template <int dim>
    FELSPA_FORCE_INLINE StokesLinearSystem<dim>::StokesLinearSystem(
      const ::dealii::DoFHandler<dim>& dofh,
      const ::dealii::Mapping<dim>& mapping, MPI_Comm mpi_comm)
      : base_type(dofh, mapping),
        ptr_scaling_operator(std::make_unique<DiagonalScalingOp>()),
        mpi_communicator(mpi_comm)
    {}


    template <int dim>
    FELSPA_FORCE_INLINE void StokesLinearSystem<dim>::set_mpi_communicator(
      const MPI_Comm& mpi_comm)
    {
      mpi_communicator = mpi_comm;
    }


    template <int dim>
    FELSPA_FORCE_INLINE auto
    StokesLinearSystem<dim>::get_preconditioner_matrix() const
      -> const matrix_type&
    {
      return preconditioner_matrix;
    }


    template <int dim>
    void StokesLinearSystem<dim>::apply_pressure_scaling(
      value_type ref_viscosity, value_type ref_length)
    {
      const value_type pressure_scaling = ref_viscosity / ref_length;
      this->matrix.block(1, 0) *= pressure_scaling;
      this->matrix.block(0, 1) *= pressure_scaling;
      preconditioner_matrix.block(1, 1) *=
        (1.0 / ref_viscosity * pressure_scaling * pressure_scaling);
    }


    template <int dim>
    void StokesLinearSystem<dim>::solve(vector_type& soln, vector_type& rhs,
                                        ::dealii::SolverControl& solver_control)
    {
      ASSERT(ptr_block_preconditioner != nullptr,
             EXCEPT_MSG("Block preconditioner has not been attached. Call "
                        "set_block_preconditioner()."));
      ASSERT(ptr_additional_control != nullptr,
             EXCEPT_MSG("Additional control parameters have not been attached. "
                        " Call set_additional_control()."));
      LOG_PREFIX(FELSPA_DEMANGLE(*this));
      felspa_log << "Entering solution sequence..." << std::endl;
      auto& control = static_cast<StokesSolverControl&>(solver_control);

      ::dealii::Timer timer;
      timer.start();

      // scale the matrix and rhs
      if (control.apply_diagonal_scaling) {
        felspa_log << "Scaling matrix and vectors..." << std::endl;
        ptr_scaling_operator->allocate(owned_dofs_per_block, mpi_communicator);
        ptr_scaling_operator->initialize(this->matrix);
        ptr_scaling_operator->apply_to_matrix(this->matrix);
        ptr_scaling_operator->apply_to_matrix(this->preconditioner_matrix);
        ptr_scaling_operator->apply_to_preconditioner(
          this->preconditioner_matrix.block(1, 1));
        ptr_scaling_operator->apply_inverse_to_vector(soln);
        ptr_scaling_operator->apply_to_vector(rhs);
      }

      felspa_log << "Reiniting AMG preconditioner..." << std::endl;
      // if (preconditioner_requires_reinit)
      ptr_block_preconditioner->reinitialize();

      for (size_type i = 0; i < soln.size(); ++i)
        if (this->constraints.is_constrained(i)) soln[i] = 0;

      const value_type tol = 1.0e-8;
      solver_control.set_max_steps(rhs.size());
      solver_control.set_tolerance(tol);
      ptr_additional_control->max_n_tmp_vectors = 10;
      felspa_log << "Solver tolerance is set at " << tol * rhs.l2_norm() << '.'
                 << std::endl;

      SolverType krylov_solver(solver_control, *ptr_additional_control);
      krylov_solver.solve(this->matrix, soln, rhs, *ptr_block_preconditioner);

      if (control.apply_diagonal_scaling)
        ptr_scaling_operator->apply_to_vector(soln);

      this->constraints.distribute(soln);
      timer.stop();

      control.n_gmres_iter.push_back(solver_control.last_step());
      control.gmres_error.push_back(solver_control.last_value());
      control.gmres_timer.push_back(timer.wall_time());
      control.log_gmres = true;

      felspa_log << "Solver " << FELSPA_DEMANGLE(krylov_solver)
                 << " converges in " << solver_control.last_step()
                 << " iterations." << std::endl;
    }


    template <int dim>
    void StokesLinearSystem<dim>::solve(vector_type& soln,
                                        ::dealii::SolverControl& solver_control)
    {
      solve(soln, this->rhs, solver_control);
    }


    template <int dim>
    void StokesLinearSystem<dim>::set_additional_control(
      const std::shared_ptr<SolverAdditionalControl>& ptr_additional_control_)
    {
      ASSERT(ptr_additional_control_ != nullptr, ExcNullPointer());
      ptr_additional_control = ptr_additional_control_;
    }


    template <int dim>
    void StokesLinearSystem<dim>::setup_constraints_and_system(
      const BCBookKeeper<dim, value_type>& bcs)
    {
      LOG_PREFIX(FELSPA_DEMANGLE(*this));
      using namespace ::dealii;

      // setup constraints //
      setup_constraints(bcs);
      felspa_log << this->constraints.n_constraints()
                 << " constraints has been setup in the system." << std::endl;

      // setup sparisty for matrix and preconditioner //
      make_sparsity_pattern();

      // allocate rhs vector //
      this->rhs.reinit(owned_dofs_per_block, mpi_communicator);

      ASSERT(this->rhs.size() == this->matrix.m(),
             ExcSizeMismatch(this->rhs.size(), this->matrix.m()));

      // set status //
      this->populated = true;
      felspa_log << "Constraints and linear system allocated." << std::endl;
    }


    template <int dim>
    void StokesLinearSystem<dim>::upon_mesh_update()
    {
      count_dofs();
      constraints_updated = false;
      preconditioner_requires_reinit = true;
    }


    template <int dim>
    void StokesLinearSystem<dim>::count_dofs()
    {
      ASSERT(this->ptr_dof_handler->has_active_dofs(), ExcDoFHandlerNotInit());
      using namespace ::dealii;

      std::vector<types::SizeType> block_component(dim + 1, 0);
      block_component[dim] = 1;

      ndofs_per_block = DoFTools::count_dofs_per_fe_block(
        this->get_dof_handler(), block_component);

      const size_type n_v = ndofs_per_block[0];
      const size_type n_p = ndofs_per_block[1];

      felspa_log << "DoFs count: " << n_v << " velocity dofs and " << n_p
                 << " pressure dofs." << std::endl;


      locally_owned_dofs = this->get_dof_handler().locally_owned_dofs();
      owned_dofs_per_block.resize(2);
      owned_dofs_per_block[0] = locally_owned_dofs.get_view(0, n_v);
      owned_dofs_per_block[1] = locally_owned_dofs.get_view(n_v, n_v + n_p);

      DoFTools::extract_locally_relevant_dofs(this->get_dof_handler(),
                                              locally_relevant_dofs);
      relevant_dofs_per_block.resize(2);
      relevant_dofs_per_block[0] = locally_relevant_dofs.get_view(0, n_v);
      relevant_dofs_per_block[1] =
        locally_relevant_dofs.get_view(n_v, n_v + n_p);
    }


    template <int dim>
    FELSPA_FORCE_INLINE const std::vector<::dealii::IndexSet>&
    StokesLinearSystem<dim>::get_owned_dofs_per_block() const
    {
      return owned_dofs_per_block;
    }


    template <int dim>
    void StokesLinearSystem<dim>::zero_out(bool lhs, bool rhs,
                                           bool precond_matrix)
    {
      base_type::zero_out(lhs, rhs);
      if (precond_matrix) preconditioner_matrix = 0.0;
    }


    template <int dim>
    void StokesLinearSystem<dim>::make_sparsity_pattern()
    {
      using namespace ::dealii;

      ASSERT(this->get_dof_handler().has_active_dofs(), ExcDoFHandlerNotInit());
      ASSERT(
        this->constraints_updated,
        EXCEPT_MSG(
          "Constraints must be updated before constructing matrix sparsity"));

      felspa_log << "Locally owned dofs: ";
      locally_owned_dofs.print(felspa_log);
      felspa_log << "Locally relevant dofs: ";
      locally_relevant_dofs.print(felspa_log);

      // system matrix //
      {
        this->matrix.clear();
        BlockDynamicSparsityPattern dsp(ndofs_per_block, ndofs_per_block);

        Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
        for (unsigned int i = 0; i < dim + 1; ++i)
          for (unsigned int j = 0; j < dim + 1; ++j)
            if (i == dim && j == dim)
              coupling[i][j] = DoFTools::none;
            else
              coupling[i][j] = DoFTools::always;

        DoFTools::make_sparsity_pattern(this->get_dof_handler(), coupling, dsp,
                                        this->get_constraints(), false);
        SparsityTools::distribute_sparsity_pattern(
          dsp, locally_owned_dofs, mpi_communicator, locally_relevant_dofs);
        this->matrix.reinit(owned_dofs_per_block, dsp, mpi_communicator);
      }

      // preconditioner matrix //
      {
        preconditioner_matrix.clear();
        BlockDynamicSparsityPattern dsp(ndofs_per_block, ndofs_per_block);
        Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
        for (unsigned int i = 0; i < dim + 1; ++i)
          for (unsigned int j = 0; j < dim + 1; ++j)
            if (i == j)
              coupling[i][j] = DoFTools::always;
            else
              coupling[i][j] = DoFTools::none;

        DoFTools::make_sparsity_pattern(this->get_dof_handler(), coupling, dsp,
                                        this->get_constraints(), false);
        SparsityTools::distribute_sparsity_pattern(
          dsp, locally_owned_dofs, mpi_communicator, locally_relevant_dofs);
        this->preconditioner_matrix.reinit(owned_dofs_per_block, dsp,
                                           mpi_communicator);
      }
    }


    template <int dim>
    void StokesLinearSystem<dim>::setup_constraints(
      const BCBookKeeper<dim, value_type>& bcs)
    {
      using namespace ::dealii;
      this->constraints.reinit(locally_owned_dofs);

      // ------------------------ //
      // Hanging node constraints //
      // ------------------------ //
      DoFTools::make_hanging_node_constraints(this->get_dof_handler(),
                                              this->constraints);
      // -------------------------- //
      // Impose boundary conditions //
      // -------------------------- //
      const FEValuesExtractors::Vector velocities(0);

      // Dirichlet BC
      if (bcs.has_category(BCCategory::dirichlet)) {
        const auto drchlt_bcs = bcs(BCCategory::dirichlet);
        for (const auto& pbc : drchlt_bcs) {
          bool mask_test = false;
          for (int idim = 0; idim < dim; ++idim)
            mask_test |= pbc->get_component_mask()[idim];
          if (!mask_test) continue;

          auto pbc_fcn = static_cast<const BCFunction<dim, value_type>*>(pbc);
          auto bdry_id = pbc_fcn->get_boundary_id();
          VectorTools::interpolate_boundary_values(
            this->get_dof_handler(),
            bdry_id,
            *pbc_fcn,
            this->constraints,
            pbc_fcn->get_component_mask());
        }

        // Periodic BC
        if (bcs.has_category(BCCategory::periodic)) {
          const auto periodic_bcs = bcs(BCCategory::periodic);
          for (const auto& pbc : periodic_bcs) {
            auto pbc_fcn =
              static_cast<const PeriodicBCFunction<dim, value_type>*>(pbc);
            auto periodicity_vector =
              pbc_fcn->collect_periodic_faces(this->get_dof_handler());
            DoFTools::make_periodicity_constraints<dim, dim, value_type>(
              periodicity_vector, this->constraints, pbc->get_component_mask(),
              {1});
          }
        }

        // No normal flux BC
        if (bcs.has_category(BCCategory::no_normal_flux)) {
          const auto no_normal_flux_bcs = bcs(BCCategory::no_normal_flux);
          std::set<::dealii::types::boundary_id> bdry_ids;

          for (const auto& pbc : no_normal_flux_bcs) {
            auto pbc_fcn = static_cast<const BCFunction<dim, value_type>*>(pbc);
            auto [it, status] = bdry_ids.insert(pbc_fcn->get_boundary_id());
            ASSERT(status,
                   EXCEPT_MSG("Boundary condition cannot be inserted twice."));
          }

          VectorTools::compute_no_normal_flux_constraints(
            this->get_dof_handler(), 0, bdry_ids, this->constraints,
            this->get_mapping());
        }
      }

      // constrain the first pressure dof to 0
      // const auto n_u = ndofs_per_block[0];
      // this->constraints.add_line(n_u);

      this->constraints.close();
      constraints_updated = true;
    }


    template <int dim>
    FELSPA_FORCE_INLINE bool StokesLinearSystem<dim>::constraints_are_updated()
      const
    {
      return constraints_updated;
    }


    template <int dim>
    FELSPA_FORCE_INLINE void
    StokesLinearSystem<dim>::flag_constraints_for_update()
    {
      constraints_updated = false;
    }


    template <int dim>
    void StokesLinearSystem<dim>::set_block_preconditioner(
      const std::shared_ptr<MatrixPreconditionerBase<matrix_type, vector_type>>&
        sp_block_preconditioner)
    {
      ASSERT(&sp_block_preconditioner->get_matrix() == &this->get_matrix(),
             EXCEPT_MSG("The block preconditioner and the linear system must "
                        "point to the same system matrix."));
      ptr_block_preconditioner = sp_block_preconditioner;
    }


    template <int dim>
    void StokesLinearSystem<dim>::DiagonalScalingOp::allocate(
      const std::vector<::dealii::IndexSet>& owned_dofs_per_block,
      MPI_Comm mpi_comm)
    {
      this->scaling_coeffs.reinit(owned_dofs_per_block, mpi_comm);
    }
  }     // namespace trilinos
#endif  // DEAL_II_WITH_TRILINOS //
}  // namespace mpi
#endif  // FELSPA_HAS_MPI //


#if defined(FELSPA_HAS_MPI) && defined(DEAL_II_WITH_TRILINOS)
template class mpi::trilinos::StokesLinearSystem<2>;
template class mpi::trilinos::StokesLinearSystem<3>;

template class StokesAssembler<mpi::trilinos::StokesLinearSystem<2>>;
template class StokesAssembler<mpi::trilinos::StokesLinearSystem<3>>;

template class StokesSimulator<2, types::TrilinosScalar,
                               mpi::trilinos::StokesLinearSystem<2>>;
template class StokesSimulator<3, types::TrilinosScalar,
                               mpi::trilinos::StokesLinearSystem<3>>;

#endif  // FELSPA_HAS_MPI && DEAL_II_WITH_TRILINOS //

FELSPA_NAMESPACE_CLOSE
