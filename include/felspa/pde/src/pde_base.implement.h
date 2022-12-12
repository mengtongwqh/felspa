#ifndef _FELSPA_PDE_PDE_BASE_IMPLEMENT_H_
#define _FELSPA_PDE_PDE_BASE_IMPLEMENT_H_

#include <deal.II/numerics/data_out.h>
#include <felspa/pde/pde_base.h>

#include <fstream>

#define PRIMARY_SIMULATOR_PRIORITY 0

FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/*               SimulatorControlBase                 */
/* ************************************************** */

template <typename NumberType>
SimulatorControlBase<NumberType>::SimulatorControlBase(
  const SimulatorControlBase<NumberType>& that)
  : ptr_mesh(std::make_shared<MeshControl<NumberType>>(*that.ptr_mesh)),
    ptr_solver(std::make_shared<dealii::SolverControl>(*that.ptr_solver))
{}


template <typename NumberType>
SimulatorControlBase<NumberType>& SimulatorControlBase<NumberType>::operator=(
  const SimulatorControlBase<NumberType>& that)
{
  if (this == &that) return *this;
  ptr_mesh = std::make_shared<MeshControl<NumberType>>(*that.ptr_mesh);
  ptr_solver = std::make_shared<dealii::SolverControl>(*that.ptr_solver);
  return *this;
}


/* ************************************************** */
/*     FESimulator<dim, FEType, LinsysType, void>     */
/* ************************************************** */

template <int dim, typename FEType, typename LinsysType>
FESimulator<dim, FEType, LinsysType>::FESimulator(Mesh<dim, value_type>& mesh,
                                                  const unsigned int fe_degree_,
                                                  const std::string& label)
  : SimulatorBase<dim, value_type>(mesh, label, PRIMARY_SIMULATOR_PRIORITY),
    ptr_fe(std::make_shared<const FEType>(fe_degree_)),
    ptr_linear_system(
      std::make_shared<LinsysType>(this->dof_handler(), this->mapping())),
    solution(std::make_shared<vector_type>()),
    ptr_mesh_refiner(nullptr),
    primary_simulator(true)
{
  ASSERT(fe().degree == fe_degree_, ExcInternalErr());
}


template <int dim, typename FEType, typename LinsysType>
FESimulator<dim, FEType, LinsysType>::FESimulator(
  Mesh<dim, value_type>& mesh, const dealii::FiniteElement<dim>& fe1,
  unsigned int n1, const dealii::FiniteElement<dim>& fe2, unsigned int n2,
  const std::string& label)
  : SimulatorBase<dim, value_type>(mesh, label, PRIMARY_SIMULATOR_PRIORITY),
    ptr_fe(std::make_shared<const FEType>(fe1, n1, fe2, n2)),
    ptr_linear_system(
      std::make_shared<LinsysType>(this->dof_handler(), this->mapping())),
    solution(std::make_shared<vector_type>()),
    ptr_mesh_refiner(nullptr),
    primary_simulator(true)
{}


template <int dim, typename FEType, typename LinsysType>
FESimulator<dim, FEType, LinsysType>::FESimulator(
  const FESimulator<dim, FEType, LinsysType>& that)
  : SimulatorBase<dim, value_type>(that),
    bcs(),
    ptr_fe(that.ptr_fe),
    ptr_linear_system(that.ptr_linear_system),
    solution(that.solution),
    ptr_mesh_refiner(nullptr),
    ptr_solution_transfer(that.ptr_solution_transfer),
    primary_simulator(false)
{}


template <int dim, typename FEType, typename LinsysType>
FESimulator<dim, FEType, LinsysType>&
FESimulator<dim, FEType, LinsysType>::operator=(
  const FESimulator<dim, FEType, LinsysType, void>& that)
{
  if (this == &that) return *this;
  this->SimulatorBase<dim, value_type>::operator=(that);
  bcs.clear();
  ptr_fe = that.ptr_fe;
  ptr_linear_system = that.ptr_linear_system;
  solution = that.solution;
  ptr_mesh_refiner = nullptr;
  ptr_solution_transfer = that.ptr_solution_transfer;
  primary_simulator = false;
  return *this;
}


template <int dim, typename FEType, typename LinsysType>
void FESimulator<dim, FEType, LinsysType>::attach_mesh_refiner(
  const std::shared_ptr<MeshRefiner<dim, value_type>>& p_mesh_refiner)
{
  ASSERT(p_mesh_refiner, ExcNullPointer());
  // hook up mesh refiner
  ptr_mesh_refiner = p_mesh_refiner;
  // append SolutionTransfer to the mesh refiner
  ptr_mesh_refiner->append(*this);
}


template <int dim, typename FEType, typename LinsysType>
void FESimulator<dim, FEType, LinsysType>::flag_mesh_for_coarsen_and_refine(
  MeshControl<value_type>& mesh_control, bool force_refine)
{
  // only execute if refiner is attached
  if (this->ptr_mesh_refiner == nullptr) return;

  LOG_PREFIX("FESimulator");

  dealii::TimerOutput::Scope t(this->simulation_timer,
                               "flag_mesh_for_coarsen_and_refine");

  ++this->n_steps_without_refinement;

  if (this->n_steps_without_refinement ==
        mesh_control.refinement_interval ||
      force_refine) {
    this->do_flag_mesh_for_coarsen_and_refine(mesh_control);
    ptr_mesh_refiner->set_update_pending();

    this->n_steps_without_refinement = 0;
    felspa_log << "Completed labelling mesh for refinement" << std::endl;
  }

  else {
    felspa_log << "Grid refinement idling: "
               << this->n_steps_without_refinement << '/'
               << mesh_control.refinement_interval << std::endl;
  }
}


template <int dim, typename FEType, typename LinsysType>
void FESimulator<dim, FEType, LinsysType>::refine_mesh(
  MeshControl<value_type>& mesh)
{
  flag_mesh_for_coarsen_and_refine(mesh, true);
  ptr_mesh_refiner->run_coarsen_and_refine();
}


template <int dim, typename FEType, typename LinsysType>
void FESimulator<dim, FEType, LinsysType>::upon_mesh_update()
{
  SimulatorBase<dim, value_type>::upon_mesh_update();
  if (this->initialized && primary_simulator) {
    this->dof_handler().distribute_dofs(this->fe());
    this->linear_system().populated = false;
  }
}


template <int dim, typename FEType, typename LinsysType>
void FESimulator<dim, FEType, LinsysType>::add_temporal_passive_member(
  const std::shared_ptr<TimedPhysicsObject<time_step_type>>& p_member)
{
  ASSERT(p_member != nullptr, ExcNullPointer());

  // only append this member if it allows passive update
  if (p_member->passive_update_allowed()) {
    temporal_passive_members.push_back(p_member);
  }
}


template <int dim, typename FEType, typename LinsysType>
void FESimulator<dim, FEType, LinsysType>::set_time_temporal_passive_members(
  time_step_type current_time)
{
  this->bcs.set_time(current_time);
  for (const auto& wp : temporal_passive_members) {
    auto p = wp.lock();
    ASSERT(p != nullptr, ExcExpiredPointer());
    p->set_time(current_time);
  }
}


template <int dim, typename FEType, typename LinsysType>
template <typename QuadratureType>
void FESimulator<dim, FEType, LinsysType>::set_quadrature(
  const QuadratureType& quad)
{
  ptr_quadrature = std::make_shared<QuadratureType>(quad);
}


template <int dim, typename FEType, typename LinsysType>
void FESimulator<dim, FEType, LinsysType>::append_boundary_condition(
  const std::shared_ptr<BCFunctionBase<dim, value_type>>& ptr_bc)
{
  ASSERT(
    this->initialized == false,
    EXCEPT_MSG("Initialization must happen after adding boundary conditions"));

  if (!this->bcs.is_initialized()) this->bcs.initialize(*(this->ptr_mesh));

  if (ptr_bc->get_category() == BCCategory::periodic) {
    // downcast to periodic BC
    auto pbc =
      std::dynamic_pointer_cast<PeriodicBCFunction<dim, value_type>>(ptr_bc);
    ASSERT(pbc != nullptr, ExcNullPointer());
    this->bcs.append(pbc);
  } else {
    auto pbc = std::dynamic_pointer_cast<BCFunction<dim, value_type>>(ptr_bc);
    ASSERT(pbc != nullptr, ExcNullPointer());
    this->bcs.append(pbc);
  }
}


template <int dim, typename FEType, typename LinsysType>
bool FESimulator<dim, FEType, LinsysType>::is_synchronized() const
{
  if (this->is_synced_with(this->solution) && this->is_synced_with(bcs)) {
    for (const auto& wp : temporal_passive_members) {
      auto p = wp.lock();
      ASSERT(p != nullptr, ExcExpiredPointer());
      if (!this->is_synced_with(*p)) return false;
    }
  } else {
    return false;
  }
  return true;
}


template <int dim, typename FEType, typename LinsysType>
bool FESimulator<dim, FEType, LinsysType>::is_synchronized(
  time_step_type current_time) const
{
  return this->is_synced_with(current_time) && is_synchronized();
}


template <int dim, typename FEType, typename LinsysType>
void FESimulator<dim, FEType, LinsysType>::reset_time()
{
  this->phsx_time = 0.0;
  this->bcs.set_time(0.0);
  this->solution.set_time(0.0);
  this->set_time_temporal_passive_members(0.0);
  this->pvd_collector.clear();
}


template <int dim, typename FEType, typename LinsysType>
void FESimulator<dim, FEType, LinsysType>::export_solution(
  ExportFile& file) const
{
#ifdef DEBUG
  LOG_PREFIX("FESimulator");
  felspa_log << "Writing solution to file " << file.get_file_name()
             << std::endl;
#endif

  // output the solution to vtk
  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(this->get_dof_handler());
  data_out.add_data_vector(*(this->solution), this->get_label_string());
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


/* ********************************************************** */
/*  FESimulator<dim, FEType, LinsysType, TempoIntegratorType> */
/* ********************************************************** */

template <int dim, typename FEType, typename LinsysType,
          typename TempoIntegratorType>
FESimulator<dim, FEType, LinsysType, TempoIntegratorType>::FESimulator(
  const FESimulator<dim, FEType, LinsysType, TempoIntegratorType>& that)
  : FESimulator<dim, FEType, LinsysType, void>(that),
    tempo_integrator(that.tempo_integrator)
{}


template <int dim, typename FEType, typename LinsysType,
          typename TempoIntegratorType>
FESimulator<dim, FEType, LinsysType, TempoIntegratorType>&
FESimulator<dim, FEType, LinsysType, TempoIntegratorType>::operator=(
  const FESimulator<dim, FEType, LinsysType, TempoIntegratorType>& that)
{
  if (this == &that) return *this;
  this->FESimulator<dim, FEType, LinsysType, TempoIntegratorType>::operator=(
    that);
  tempo_integrator = that.tempo_integrator;
  return *this;
}


template <int dim, typename FEType, typename LinsysType,
          typename TempoIntegrator>
void FESimulator<dim, FEType, LinsysType, TempoIntegrator>::reset_time()
{
  base_type::reset_time();
  this->tempo_integrator.initialize();
}


template <int dim, typename FEType, typename LinsysType,
          typename TempoIntegrator>
void FESimulator<dim, FEType, LinsysType, TempoIntegrator>::
  update_solution_and_sync(const vector_type& curr_soln,
                           time_step_type curr_time)
{
#ifdef DEBUG
  LOG_PREFIX("FESimulator");
  felspa_log << "Updated solution to simulation time " << curr_time
             << std::endl;
#endif  // DEBUG //
  this->solution.update(curr_soln, curr_time);
  this->set_time_temporal_passive_members(curr_time);
  this->phsx_time = curr_time;
}


template <int dim, typename FEType, typename LinsysType,
          typename TempoIntegrator>
void FESimulator<dim, FEType, LinsysType, TempoIntegrator>::
  update_solution_and_sync(vector_type&& curr_soln, time_step_type curr_time)
{
#ifdef DEBUG
  LOG_PREFIX("FESimulator");
  felspa_log << "Updated solution to simulation time " << curr_time
             << std::endl;
#endif  // DEBUG //
  this->solution.update(std::move(curr_soln), curr_time);
  this->set_time_temporal_passive_members(curr_time);
  this->phsx_time = curr_time;
}


FELSPA_NAMESPACE_CLOSE
#endif /* _FELSPA_PDE_PDE_BASE_IMPLEMENT_H_ */
