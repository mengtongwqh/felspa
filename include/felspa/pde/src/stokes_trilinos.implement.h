#ifndef _FELSPA_PDE_STOKES_TRILINOS_IMPLEMENT_H_
#define _FELSPA_PDE_STOKES_TRILINOS_IMPLEMENT_H_
#include <felspa/pde/stokes_trilinos.h>

FELSPA_NAMESPACE_OPEN

#ifdef FELSPA_HAS_MPI
/* ---------- */
namespace mpi
/* ---------- */
{
#ifdef DEAL_II_WITH_TRILINOS
  template <int dim, typename SPreconditionerType>
  BlockSchurPreconditioner<dim, trilinos::dealii::PreconditionAMG,
                           SPreconditionerType>::
    BlockSchurPreconditioner(
      const matrix_type& matrix,
      const matrix_type& stokes_precond_matrix,
      const ::dealii::DoFHandler<dim>& dofh,
      const std::shared_ptr<A_preconditioner_control>& sp_A_control,
      const std::shared_ptr<S_preconditioner_control>& sp_S_control,
      MPI_Comm mpi_communicator)
    : base_type(matrix, mpi_communicator),
      ptr_preconditioner_matrix(&stokes_precond_matrix),
      ptr_dof_handler(&dofh),
      ptr_A_preconditioner(std::make_unique<A_preconditioner_type>()),
      ptr_S_preconditioner(std::make_unique<S_preconditioner_type>()),
      ptr_control_A(sp_A_control),
      ptr_control_S(sp_S_control)
  {
    // allocate the preconditioners
    base_type::initialize(ptr_preconditioner_matrix->block(1, 1),
                          *ptr_A_preconditioner, *ptr_S_preconditioner);
  }


  template <int dim, typename SPreconditionerType>
  FELSPA_FORCE_INLINE auto
  BlockSchurPreconditioner<dim, trilinos::dealii::PreconditionAMG,
                           SPreconditionerType>::get_preconditioner_matrix()
    const -> const matrix_type&
  {
    ASSERT(ptr_preconditioner_matrix != nullptr, ExcNullPointer());
    return *ptr_preconditioner_matrix;
  }


  template <int dim, typename SPreconditionerType>
  void BlockSchurPreconditioner<dim, trilinos::dealii::PreconditionAMG,
                                SPreconditionerType>::reinitialize()
  {
    using namespace ::dealii;

    ASSERT(ptr_control_A != nullptr, ExcNullPointer());
    ASSERT(ptr_control_S != nullptr, ExcNullPointer());

    base_type::initialize(ptr_preconditioner_matrix->block(1, 1),
                          *ptr_A_preconditioner, *ptr_S_preconditioner);

    FEValuesExtractors::Vector v(0);
    std::vector<std::vector<bool>> constant_modes;
    DoFTools::extract_constant_modes(
      *this->ptr_dof_handler, this->ptr_dof_handler->get_fe().component_mask(v),
      constant_modes);
    ptr_control_A->constant_modes = constant_modes;
    ptr_control_A->higher_order_elements = true;
    // ptr_control_A->smoother_sweeps = 2;
    // ptr_control_A->smoother_type = "IC";
    // ptr_control_A->n_cycles = 5;
    // ptr_control_A->w_cycle = true;
    // ptr_control_A->aggregation_threshold = 0.02;
    // ptr_control_A->output_details = true;

    ptr_A_preconditioner->initialize(ptr_preconditioner_matrix->block(0, 0),
                                     *ptr_control_A);
    ptr_S_preconditioner->initialize(ptr_preconditioner_matrix->block(1, 1),
                                     *ptr_control_S);

    this->resize_tmp_vector();
    felspa_log << "A and S preconditioners have been reinitialized."
               << std::endl;
  }


  template <int dim, typename SPreconditionerType>
  void BlockSchurPreconditioner<dim, trilinos::dealii::PreconditionAMG,
                                SPreconditionerType>::resize_tmp_vector()
  {
    this->tmp.reinit(
      this->ptr_matrix->block(1, 1).locally_owned_domain_indices(),
      this->mpi_communicator);
  }

#endif  // DEAL_II_HAS_TRILINOS
}  // namespace mpi
#endif  // FELSPA_HAS_MPI


FELSPA_NAMESPACE_CLOSE
#endif  // _FELSPA_PDE_STOKES_TRILINOS_IMPLEMENT_H_ //
