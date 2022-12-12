#ifndef _FELSPA_PDE_LINEAR_SYSTEMS_IMPLEMENT_H_
#define _FELSPA_PDE_LINEAR_SYSTEMS_IMPLEMENT_H_

#include <felspa/pde/linear_systems.h>

FELSPA_NAMESPACE_OPEN

namespace dg
{
  /* ************************************************** */
  /** \class DGLinearSystem */
  /* ************************************************** */

  template <int dim, typename NumberType>
  DGLinearSystem<dim, NumberType>::DGLinearSystem(
    const dealii::DoFHandler<dim>& dof_handler)
    : LinearSystem<dim, NumberType>(dof_handler)
  {}


  template <int dim, typename NumberType>
  DGLinearSystem<dim, NumberType>::DGLinearSystem(
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::Mapping<dim>& mapping)
    : LinearSystem<dim, NumberType>(dof_handler, mapping)
  {}


  template <int dim, typename NumberType>
  void DGLinearSystem<dim, NumberType>::solve(
    vector_type& solution,
    const vector_type& rhsvec,
    dealii::SolverControl& solver_control) const
  {
    dealii::SolverCG<vector_type> solver(solver_control);
    dealii::PreconditionJacobi<matrix_type> preconditioner;
    preconditioner.initialize(this->get_matrix());
    AssertIsFinite(rhsvec.l2_norm());
    AssertIsFinite(solution.l2_norm());

    try {
      solver.solve(this->matrix, solution, rhsvec, preconditioner);
    }
    catch (...) {
      THROW(EXCEPT_MSG("Solver Error"));
    }
  }


  template <int dim, typename NumberType>
  void DGLinearSystem<dim, NumberType>::populate_system_from_dofs()
  {
    // make sure that dof distribution is successful
    ASSERT(this->ptr_dof_handler->has_active_dofs(), ExcDoFHandlerNotInit());

    // construct sparsity pattern
    dealii::DynamicSparsityPattern dsp(this->get_dof_handler().n_dofs());
    dealii::DoFTools::make_flux_sparsity_pattern(this->get_dof_handler(), dsp);
    this->sparsity_pattern.copy_from(dsp);

    // allocate space for rhs vector, matrix
    this->matrix.reinit(this->sparsity_pattern);
    this->rhs.reinit(this->get_dof_handler().n_dofs());

    // allocation successful, flag as populated
    this->populated = true;
  }
}  // namespace dg


/* ************************************************** */
/*                MixedVPLinearSystem                 */
/* ************************************************** */
template <int dim, typename NumberType>
MixedVPLinearSystem<dim, NumberType>::MixedVPLinearSystem(
  const dealii::DoFHandler<dim>& dofh)
  : BlockLinearSystem<dim, NumberType>(dofh)
{}


template <int dim, typename NumberType>
MixedVPLinearSystem<dim, NumberType>::MixedVPLinearSystem(
  const dealii::DoFHandler<dim>& dofh, const dealii::Mapping<dim>& mapping)
  : BlockLinearSystem<dim, NumberType>(dofh, mapping)
{}


template <int dim, typename NumberType>
void MixedVPLinearSystem<dim, NumberType>::populate_system_from_dofs()
{
  using namespace dealii;
  LOG_PREFIX("MixedVPLinearSystem");

  ASSERT(this->ptr_dof_handler->has_active_dofs(), ExcDoFHandlerNotInit());
  ASSERT(this->constraints_updated,
         EXCEPT_MSG("Constraints need to be updated before populating system. "
                    "Call setup_constraints()"));

  const unsigned int ndof_u(this->ndofs_per_block[0]),
    ndof_p(this->ndofs_per_block[1]);

  felspa_log << "Populating the system with " << ndof_u << " velocity dofs and "
             << ndof_p << " pressure dofs" << std::endl;

  make_matrix_sparsity();  // allocate space for LHS matrix

#ifdef EXPORT_MATRIX
  felspa_log << "Exporting Stokes system sparsity pattern..." << std::endl;
  ExportFile sparsity_file("StokesSystemSparsity.gnuplot");
  this->print_sparsity(sparsity_file);
  felspa_log << "Exporting Stokes system preconiditioner sparsity..."
             << std::endl;
  ExportFile preconditioner_sparsity("StokesPreconditionerSparsity.gnuplot");
  this->preconditioner_sparsity.print_gnuplot(
    preconditioner_sparsity.access_stream());
#endif

  // allocate space for rhs vector //
  this->rhs.reinit(2);
  this->rhs.block(0).reinit(ndof_u);
  this->rhs.block(1).reinit(ndof_p);
  this->rhs.collect_sizes();

  // mark system as populated
  this->populated = true;
  // set constraints to be expired
  this->constraints_updated = false;
}


template <int dim, typename NumberType>
void MixedVPLinearSystem<dim, NumberType>::setup_constraints(
  const BCBookKeeper<dim, value_type>& bcs)
{
  ASSERT(this->ptr_dof_handler->has_active_dofs(), ExcDoFHandlerNotInit());

  // clear constraints from last step
  this->constraints.clear();

  // make handing node constraints
  dealii::DoFTools::make_hanging_node_constraints(this->get_dof_handler(),
                                                  this->constraints);

  setup_bc_constraints(bcs);

  // close and accept no more constraints
  this->constraints.close();
  this->constraints_updated = true;

  LOG_PREFIX("MixedVPLinearSystem")
  felspa_log << this->constraints.n_constraints()
             << " constraints has been set up." << std::endl;
}


template <int dim, typename NumberType>
void MixedVPLinearSystem<dim, NumberType>::setup_constraints_and_system(
  const BCBookKeeper<dim, value_type>& bcs)
{
  using namespace dealii;
  using this_type = MixedVPLinearSystem<dim, NumberType>;

  // setting up the constraints
  Threads::Task<void> task_setup_constraints =
    Threads::new_task(&this_type::setup_constraints, *this, std::cref(bcs));

  // allocate the rhs vector
  this->rhs.reinit(2);
  this->rhs.block(0).reinit(this->ndofs_per_block[0]);
  this->rhs.block(1).reinit(this->ndofs_per_block[1]);
  this->rhs.collect_sizes();

  // sync constraints
  task_setup_constraints.join();

  // setting up the linear system sparsity
  make_matrix_sparsity();

  // set status
  this->populated = true;
}


template <int dim, typename NumberType>
void MixedVPLinearSystem<dim, NumberType>::make_matrix_sparsity()
{
  using namespace dealii;

  LOG_PREFIX("MixedVPLinearSystem");
  ASSERT(this->ptr_dof_handler->has_active_dofs(), ExcDoFHandlerNotInit());
  ASSERT(this->constraints_updated,
         EXCEPT_MSG("Constraints need to be updated before populating system. "
                    "Call setup_constraints()"));

  const auto ndof_u(this->ndofs_per_block[0]), ndof_p(this->ndofs_per_block[1]);
  felspa_log << "Populating the system matrix with " << ndof_u
             << " velocity dofs and " << ndof_p << " pressure dofs"
             << std::endl;

  BlockDynamicSparsityPattern dsp(2, 2);
  dsp.block(0, 0).reinit(ndof_u, ndof_u);
  dsp.block(0, 1).reinit(ndof_u, ndof_p);
  dsp.block(1, 0).reinit(ndof_p, ndof_u);
  dsp.block(1, 1).reinit(ndof_p, ndof_p);
  dsp.collect_sizes();

  Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);

  for (unsigned int c = 0; c < dim + 1; ++c)
    for (unsigned int d = 0; d < dim + 1; ++d)
      if (!(c == dim && d == dim))
        coupling[c][d] = DoFTools::always;
      else
        coupling[c][d] = DoFTools::none;

  DoFTools::make_sparsity_pattern(this->get_dof_handler(), coupling, dsp,
                                  this->constraints, false);
  this->sparsity_pattern.copy_from(dsp);
  this->matrix.reinit(this->sparsity_pattern);
}


FELSPA_NAMESPACE_CLOSE
#endif  // _FELSPA_PDE_LINEAR_SYSTEMS_IMPLEMENT_H_ //