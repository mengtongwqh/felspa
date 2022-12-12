#include <deal.II/grid/grid_out.h>
#include <felspa/pde/pde_base.h>

FELSPA_NAMESPACE_OPEN


/* ************************************************** */
/*              \class SimulatorBase                  */
/* ************************************************** */

template <int dim, typename NumberType>
SimulatorBase<dim, NumberType>::SimulatorBase(Mesh<dim, value_type>& mesh,
                                              const std::string& label,
                                              int priority)
  : ptr_mesh(&mesh),
    mesh_signal_priority(priority),
    ptr_dof_handler(std::make_shared<dealii::DoFHandler<dim>>(mesh)),
    ptr_mapping(std::make_shared<dealii::MappingQGeneric<dim, dim>>(1)),
    label_string(label),
    parent_simulator(nullptr),
    simulation_timer(std::cout, dealii::TimerOutput::summary,
                     dealii::TimerOutput::wall_times),
    pvd_collector(label),
    initialized(false)
{
  // must be nonempty or export file will have no name
  ASSERT_NON_EMPTY(label);
  connect_mesh_signals(mesh, priority);
}


template <int dim, typename NumberType>
SimulatorBase<dim, NumberType>::SimulatorBase(
  const SimulatorBase<dim, NumberType>& that)
  : TimedPhysicsObject<NumberType>(that),
    boost::signals2::trackable(that),
    ptr_mesh(that.ptr_mesh),
    mesh_signal_priority(that.mesh_signal_priority + 1),
    ptr_dof_handler(that.ptr_dof_handler),
    ptr_mapping(that.ptr_mapping),
    label_string(that.label_string + "_copy"),
    parent_simulator(&that),
    simulation_timer(std::cout, dealii::TimerOutput::summary,
                     dealii::TimerOutput::wall_times),
    pvd_collector(that.label_string),
    initialized(false)
{
  connect_mesh_signals(*ptr_mesh, mesh_signal_priority);
}

template <int dim, typename NumberType>
FELSPA_FORCE_INLINE dealii::FEValues<dim>*
SimulatorBase<dim, NumberType>::fe_values(
  const dealii::Quadrature<dim>& quad, dealii::UpdateFlags update_flags) const
{
  // because the dealii::FEValues does not provide copy constructor
  return new dealii::FEValues<dim>(this->get_mapping(), this->get_fe(), quad,
                                   update_flags);
}


template <int dim, typename NumberType>
void SimulatorBase<dim, NumberType>::upon_mesh_update()
{
  mesh_update_detected = true;
}


template <int dim, typename NumberType>
SimulatorBase<dim, NumberType>& SimulatorBase<dim, NumberType>::operator=(
  const SimulatorBase<dim, NumberType>& that)
{
  if (this == &that) return *this;
  std::cout << "called simulatorbase assignment , priority = "
            << mesh_signal_priority << std::endl;
  disconnect_mesh_signals();
  TimedPhysicsObject<NumberType>::operator=(that);
  ptr_mesh = that.ptr_mesh;
  mesh_signal_priority = that.mesh_signal_priority + 1;
  ptr_dof_handler = that.ptr_dof_handler;
  ptr_mapping = that.ptr_mapping;
  label_string = that.label_string + "_copy";
  parent_simulator = &that;
  this->phsx_time = that.phsx_time;
  pvd_collector.clear();
  pvd_collector.set_file_name(this->get_label_string());
  initialized = false;
  connect_mesh_signals(*ptr_mesh, mesh_signal_priority);
  return *this;
}


template <int dim, typename NumberType>
void SimulatorBase<dim, NumberType>::export_solutions(
  const std::string& path) const
{
  using dealii::Utilities::int_to_string;

  pvd_collector.set_file_path(path);

  types::SizeType counter = pvd_collector.get_file_count() + 1;

  // construct file name
  std::string master_file_name =
    this->get_label_string() + '_' +
    int_to_string(counter, constants::max_export_numeric_digits);
  std::string vtu_file_name = path + master_file_name + ".vtu";

  // export this time step
  ExportFile vtu_file(vtu_file_name);
  export_solution(vtu_file);

  // append the record to the collector
  pvd_collector.append_record(this->get_time(), master_file_name + ".vtu");
}


template <int dim, typename NumberType>
void SimulatorBase<dim, NumberType>::connect_mesh_signals(
  Mesh<dim, value_type>& mesh, int priority)
{
  ASSERT(mesh_signal_connections.empty(),
         EXCEPT_MSG("The mesh_signal_connections is expected to be empty "
                    "before establishing mesh connection."));

#ifdef FELSPA_HAS_MPI
  mesh_signal_connections.push_back(
    mesh.signals.post_distributed_refinement.connect(
      priority,
      boost::bind(&SimulatorBase<dim, NumberType>::upon_mesh_update, this)));
#else
  mesh_signal_connections.push_back(mesh.signals.post_refinement.connect(
    priority,
    boost::bind(&SimulatorBase<dim, NumberType>::upon_mesh_update, this)));
#endif

  mesh_signal_connections.push_back(mesh.signals.create.connect(
    priority,
    boost::bind(&SimulatorBase<dim, NumberType>::upon_mesh_update, this)));
  mesh_signal_connections.push_back(mesh.signals.clear.connect(
    priority,
    boost::bind(&SimulatorBase<dim, NumberType>::upon_mesh_clear, this)));
}


template <int dim, typename NumberType>
void SimulatorBase<dim, NumberType>::disconnect_mesh_signals()
{
  while (!mesh_signal_connections.empty()) {
    mesh_signal_connections.back().disconnect();
    mesh_signal_connections.pop_back();
  }
}


template <int dim, typename NumberType>
FELSPA_FORCE_INLINE void SimulatorBase<dim, NumberType>::advance_time(
  time_step_type time_step)
{
  this->phsx_time += time_step;
}


template <int dim, typename NumberType>
FELSPA_FORCE_INLINE void SimulatorBase<dim, NumberType>::set_time(
  time_step_type current_time)
{
  ASSERT(current_time >= 0.0, ExcArgumentCheckFail());
  auto time_step = current_time - this->get_time();
  advance_time(time_step);
}

/* -------- Explicit Instantiations ----------*/
template class SimulatorBase<1, types::DoubleType>;
template class SimulatorBase<2, types::DoubleType>;
template class SimulatorBase<3, types::DoubleType>;

template struct SimulatorControlBase<types::DoubleType>;
/* -------------------------------------------*/
FELSPA_NAMESPACE_CLOSE
