#ifndef _FELSPA_PDE_PDE_BASE_H_
#define _FELSPA_PDE_PDE_BASE_H_

#include <deal.II/base/quadrature.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <felspa/base/base_classes.h>
#include <felspa/base/exceptions.h>
#include <felspa/base/felspa_config.h>
#include <felspa/base/io.h>
#include <felspa/base/utilities.h>
#include <felspa/linear_algebra/linear_system.h>
#include <felspa/linear_algebra/solution_vector.h>
#include <felspa/mesh/mesh.h>
#include <felspa/mesh/mesh_refine.h>
#include <felspa/pde/boundary_conditions.h>
#include <felspa/pde/time_integration.h>

#include <boost/bind.hpp>
#include <boost/signals2.hpp>
#include <fstream>
#include <string>

FELSPA_NAMESPACE_OPEN

// forward declaration
template <int dim, typename NumberType>
class SimulatorBase;


/* ************************************************** */
/**
 * @brief Type of DG polynomial space
 */
/* ************************************************** */
namespace dg
{
  template <int dim>
  using FEDGType = dealii::FE_DGP<dim>;

  // template <int dim>
  // using FEDGType = dealii::FE_DGQLegendre<dim>;

  // template <int dim>
  //  using FEDGType = dealii::FE_DGQ<dim>;
}  // namespace dg


/* ************************************************** */
/**
 * A type trait for testing if a type is dervied from
 * \c SimulatorBase class.
 */
/* ************************************************** */
template <typename T>
struct is_simulator
  : public std::is_base_of<SimulatorBase<T::dimension, typename T::value_type>,
                           T>
{};


/* ************************************************** */
/**
 * Base class for simulator contol parameters
 */
/* ************************************************** */
template <typename NumberType>
struct SimulatorControlBase
{
  using value_type = NumberType;

  /**
   * Constructor.
   */
  SimulatorControlBase()
    : ptr_mesh(std::make_shared<MeshControl<value_type>>()),
      ptr_solver(std::make_shared<dealii::SolverControl>())
  {}


  /**
   * Copy constructor
   */
  SimulatorControlBase(const SimulatorControlBase<NumberType>& that);


  /**
   * Copy assignment.
   */
  SimulatorControlBase<NumberType>& operator=(
    const SimulatorControlBase<NumberType>& that);


  /**
   * Destructor.
   */
  virtual ~SimulatorControlBase() = default;


  /**
   * Control parameters for mesh related parameters.
   */
  std::shared_ptr<MeshControl<value_type>> ptr_mesh;


  /**
   * Pointer to a linear solver control struct.
   */
  std::shared_ptr<dealii::SolverControl> ptr_solver;
};


/* ************************************************** */
/**
 * Abstract base class for all PDE simulators.
 * Contains only info about the geometry
 */
/* ************************************************** */
template <int dim, typename NumberType>
class SimulatorBase : public TimedPhysicsObject<NumberType>,
                      public boost::signals2::trackable
{
 public:
  using value_type = NumberType;
  using time_step_type = value_type;


  /** Spatial dimension */
  constexpr const static int spacedim = dim;
  constexpr const static int dimension = dim;


  /** \name Constructors and Destructors */
  //@{
  /** Constructor */
  SimulatorBase(Mesh<dim, value_type>& mesh, const std::string& label,
                int mesh_signal_priority = 0);

  /** Copy constructor */
  SimulatorBase(const SimulatorBase<dim, NumberType>&);

  /** Copy assignment */
  SimulatorBase<dim, NumberType>& operator=(
    const SimulatorBase<dim, NumberType>&);

  /** Destructor */
  virtual ~SimulatorBase() { disconnect_mesh_signals(); }
  //@}


  /** \name Getting Info */
  //@{
  /** Access the triangulation */
  Mesh<dim, value_type>& mesh() const
  {
    ASSERT(ptr_mesh, ExcNullPointer());
    return *ptr_mesh;
  }

  const Mesh<dim, value_type>& get_mesh() const
  {
    ASSERT(ptr_mesh, ExcNullPointer());
    return *ptr_mesh;
  }

  /** return dof_handler */
  const dealii::DoFHandler<dim>& get_dof_handler() const
  {
    ASSERT(ptr_dof_handler.get(), ExcNullPointer());
    return *ptr_dof_handler;
  }

  /** return Mapping object */
  const dealii::Mapping<dim>& get_mapping() const
  {
    ASSERT(ptr_mapping.get(), ExcNullPointer());
    return *ptr_mapping;
  }

  /** Return the reference to \c FiniteElement */
  const dealii::FiniteElement<dim, dim>& get_fe(unsigned int index = 0) const
  {
    ASSERT(this->get_dof_handler().has_active_dofs(), ExcDoFHandlerNotInit());
    return this->get_dof_handler().get_fe(index);
  }

  /** Get the identifier string */
  const std::string& get_label_string() const { return label_string; }

  /** obtain an FEValues based on the FEValues*/
  dealii::FEValues<dim>* fe_values(const dealii::Quadrature<dim>& quad,
                                   dealii::UpdateFlags update_flags) const;
  //@}


  /** \name Functions to deal with mesh-related signals */
  //@{
  /** Execute when change in mesh config detected */
  virtual void upon_mesh_update();

  /** Execute when the mesh is cleared */
  virtual void upon_mesh_clear() { THROW(ExcUnimplementedVirtualFcn()); }
  //@}


  /** \name Temporal-Related functions */
  //@{
  /**
   * Set the \c time member in the simulator to current_time.
   */
  void set_time(time_step_type current_time) override;

  /**
   * To increment simulation time by time_step.
   * This is the workhorse to advance the temporal discretization
   * To achieve a more predictable behaviour,
   * we should probably also update the solution vector,
   * meaning that the whole solver will be run if we evolve the time forward
   */
  void advance_time(time_step_type time_step) override;

  /**
   * Make an estimation of the next time step
   * without actually advancing the time step
   * @return time_step_type
   */
  virtual time_step_type estimate_max_time_step() const
  {
    THROW(ExcUnimplementedVirtualFcn());
  }
  //@}


  /**
   * Pure virtual function to write solution to file.
   */
  virtual void export_solution(ExportFile&) const = 0;

  /**
   *  Export the solution and append the record for a pvd file
   */
  void export_solutions(
    const std::string& export_path = constants::current_dir) const;


 protected:
  /**
   * Access \c DoFHandler object
   */
  dealii::DoFHandler<dim>& dof_handler()
  {
    ASSERT(ptr_dof_handler.get(), ExcNullPointer());
    return *ptr_dof_handler;
  }


  /**
   * Const overload of the previous function
   */
  const dealii::DoFHandler<dim>& dof_handler() const
  {
    ASSERT(ptr_dof_handler.get(), ExcNullPointer());
    return *ptr_dof_handler;
  }


  /**
   * Mapping object
   */
  dealii::Mapping<dim, dim>& mapping()
  {
    ASSERT(ptr_mapping, ExcNullPointer());
    return *ptr_mapping;
  }


  /**
   * Return if the simulator is initialized
   */
  bool is_initialized() const { return initialized; }


  /**
   * Pointer to a triangulation object
   */
  dealii::SmartPointer<Mesh<dim, value_type>, SimulatorBase<dim, value_type>>
    ptr_mesh;


  /**
   * List of connections to the mesh object
   */
  std::vector<boost::signals2::connection> mesh_signal_connections;


  /**
   * Priority for mesh signal
   */
  int mesh_signal_priority;


  /**
   * DoF handler.
   * This will be generated in the constructor by passing in the
   * \c Mesh<dim, value_type> object and later be initialized with
   * a reference to \c dealii::FiniteElement<dim>&
   */
  std::shared_ptr<dealii::DoFHandler<dim>> ptr_dof_handler;


  /**
   * Pointer to a mapping object
   */
  std::shared_ptr<dealii::Mapping<dim, dim>> ptr_mapping;


  /**
   * An \c std::string to identify the simulator
   */
  std::string label_string;


  /**
   * Pointer to the parent simulator.
   * If the simulator is a primary simulator, this pointer will be \c nullptr.
   * If the simulator is not a primary simulator, this pointer will point to
   * the parent simulator one level higher.
   */
  const SimulatorBase<dim, NumberType>* parent_simulator;


  /**
   * Simulation computation timer.
   */
  mutable dealii::TimerOutput simulation_timer;


  /**
   * This will allocate an object of \c ExportManager.
   * This object will be responsible for exporting multiple vtk files.
   */
  mutable PVDCollector<time_step_type> pvd_collector;


  /**
   * When mesh is updated, this flag is set to \c true.
   * Also when the simulator is first created, this is initialized
   * to \c true by default.
   */
  bool mesh_update_detected = true;


  /**
   * After initialization this will be set to \c true
   */
  bool initialized;


  /**
   * A counter that records how many time step cycles have passed
   * without mesh refinement.
   */
  unsigned int n_steps_without_refinement = 0;


 private:
  /**
   * Establish connection to receive signal from mesh
   */
  void connect_mesh_signals(Mesh<dim, value_type>& mesh, int priority);


  /**
   * Disconnect all mesh signal connections
   */
  void disconnect_mesh_signals();
};


/* ************************************************** */
/**
 * Abstract base class for PDE solvers.
 * Utilizing Finite Element discretization in deal.II
 */
/* ************************************************** */
template <int dim, typename FEType, typename LinsysType,
          typename TempoIntegratorType = void>
class FESimulator : public FESimulator<dim, FEType, LinsysType, void>
{
  /** All relevant time integration method */
  template <int order>
  friend struct RungeKuttaTVD;

 public:
  using base_type = FESimulator<dim, FEType, LinsysType, void>;
  using fe_type = FEType;
  using value_type =
    typename FESimulator<dim, FEType, LinsysType, void>::value_type;
  using time_step_type = typename TempoIntegratorType::value_type;
  using vector_type = typename LinsysType::vector_type;
  using typename base_type::bcs_type;


  const TempoIntegratorType& get_tempo_integrator() const
  {
    return tempo_integrator;
  }


  /**
   * Test if all temporally passive members of the simulator
   * is synchronized with the simulator.
   */
  bool is_synchronized() const
  {
    return base_type::is_synchronized() &&
           this->is_synced_with(tempo_integrator);
  }


  /**
   * Test if all temporally passive members of the simulator
   * is syncrhonized to current_time.
   */
  bool is_synchonized(time_step_type current_time) const
  {
    return base_type::is_synchronized(current_time) &&
           this->is_synced_with(tempo_integrator);
  }


 protected:
  /** \name Basic Object Behaviors */
  //@{
  /** Constructor */
  FESimulator(Mesh<dim, value_type>& triag, const unsigned int fe_degree,
              const std::string& label)
    : FESimulator<dim, FEType, LinsysType, void>(triag, fe_degree, label)
  {}

  /** Constructor taking two components */
  FESimulator(Mesh<dim, value_type>& triag,
              const dealii::FiniteElement<dim>& fe1,
              const unsigned int n1,
              const dealii::FiniteElement<dim>& fe2,
              const unsigned int n2,
              const std::string& label)
    : FESimulator<dim, FEType, LinsysType, void>(triag, fe1, n1, fe2, n2, label)
  {}


  /**
   * Copy constructor.
   */
  FESimulator(const FESimulator<dim, FEType, LinsysType, TempoIntegratorType>&);

  /**
   * Copy assignment.
   */
  FESimulator<dim, FEType, LinsysType, TempoIntegratorType>& operator=(
    const FESimulator<dim, FEType, LinsysType, TempoIntegratorType>&);
  //@}


  /** \name Temporal Simulator Functions */
  //@{
  virtual time_step_type advance_time(time_step_type time_step,
                                      bool compute_single_cycle) = 0;

  virtual void advance_time(time_step_type time_step) override
  {
    advance_time(time_step, false);
  }

  virtual vector_type explicit_time_derivative(
    time_step_type current_time, const vector_type& soln_prev_step)
  {
    UNUSED_VARIABLE(current_time);
    UNUSED_VARIABLE(soln_prev_step);
    THROW(ExcUnimplementedVirtualFcn());
  }

  virtual vector_type implicit_time_derivative(
    time_step_type current_time,
    time_step_type time_step,
    const vector_type& soln_prev_step)
  {
    UNUSED_VARIABLE(current_time);
    UNUSED_VARIABLE(time_step);
    UNUSED_VARIABLE(soln_prev_step);
    THROW(ExcUnimplementedVirtualFcn());
  }

  virtual vector_type mixed_time_derivative(time_step_type current_time,
                                            time_step_type time_step,
                                            const vector_type& soln_prev_step)
  {
    UNUSED_VARIABLE(current_time);
    UNUSED_VARIABLE(time_step);
    UNUSED_VARIABLE(soln_prev_step);
    THROW(ExcUnimplementedVirtualFcn());
  }

  /** Reset time to zero. */
  void reset_time();
  //@}


  /** \name Update solution as well as solution_time */
  //@{
  /**
   * Overriding the update_solution in \c SimulatorBase. Will update temporal
   * passive members after solution and time is updated.
   */
  void update_solution_and_sync(const vector_type&,
                                time_step_type current_time);

  /**
   * Same as above, but overloaded with
   * r-value reference for input solution vector.
   */
  void update_solution_and_sync(vector_type&&, time_step_type current_time);
  //@}


  /**
   * The temporal integrator.
   * Responsible for computing the temporal envolution of the solution.
   */
  TempoIntegratorType tempo_integrator;
};


/* ----------------- */
/**
 * \class FESimulator
 * \tparam dim, FEType, LinsysType
 * Specialization for \c FESimulator that is used for steady-state
 * problems
 */
/* ----------------- */
template <int dim, typename FEType, typename LinsysType>
class FESimulator<dim, FEType, LinsysType, void>
  : public SimulatorBase<dim, typename LinsysType::value_type>
{
 public:
  using vector_type = typename LinsysType::vector_type;
  using value_type = typename vector_type::value_type;
  using base_type = SimulatorBase<dim, value_type>;
  using typename base_type::time_step_type;
  using fe_type = FEType;
  using bcs_type = BCBookKeeper<dim, value_type>;

  friend struct SolutionTransferBase;
  friend class MeshRefiner<dim, value_type>;


  /**
   * \name Accessing Object Members/Info
   */
  //@{
  /** return finite element object */
  const FEType& get_fe() const { return fe(); }

  const bcs_type& get_bcs() const { return bcs; }

  /** Get quadrature object */
  const dealii::Quadrature<dim>& get_quadrature() const
  {
    ASSERT(ptr_quadrature != nullptr, ExcNullPointer());
    return *ptr_quadrature;
  }

  /** return linear system */
  const LinsysType& get_linear_system() const { return linear_system(); }

  /** Polynomial degree of the finite element approximation */
  virtual unsigned int fe_degree(unsigned int component = 0) const
  {
    UNUSED_VARIABLE(component);
    return this->dof_handler().get_fe().degree;
  }

  /** Get solution vector */
  const vector_type& get_solution_vector() const { return *solution; }

  /** Get the complete TimedSolutionVector object */
  const TimedSolutionVector<vector_type, time_step_type>& get_solution() const
  {
    return solution;
  }
  //@}


  /**
   * Setting the type of quadrature used in matrix assembly.
   */
  template <typename QuadratureType>
  void set_quadrature(const QuadratureType& quad);


  /**
   * Add an entry of a boundary condition function.
   */
  void append_boundary_condition(
    const std::shared_ptr<BCFunctionBase<dim, value_type>>& ptr_bc);


  /**
   * Test if all members are synchronized
   */
  bool is_synchronized() const;


  /**
   * Test if all members are synchronized to a certain time
   */
  bool is_synchronized(time_step_type current_time) const;


  /**
   * Setting the time of the temporal related members to 0
   */
  void reset_time();


  /**
   * Return \c true of \c MeshRefiner is attached
   */
  bool has_mesh_refiner() const { return ptr_mesh_refiner.get(); }


  /**
   * Attach shared pointer of a \c MeshRefiner object.
   * This will trigger mesh refinement routine in the simulator,
   * if defined.
   */
  void attach_mesh_refiner(
    const std::shared_ptr<MeshRefiner<dim, value_type>>& p_mesh_refiner);


  /**
   * Carry out the mesh refinement immediately.
   * All simulators that are appended into mesh_refiner will
   * have their solutions interpolated to new grid.
   */
  void refine_mesh(MeshControl<value_type>& mesh_control);


  /**
   * Only label the triangulation for coarsening/refinement.
   */
  void flag_mesh_for_coarsen_and_refine(MeshControl<value_type>& mesh_control,
                                        bool force = false);


  /**
   * Write the solution vector to .vtk file.
   */
  void export_solution(ExportFile& file) const override;


 protected:
  /**
   * \name Basic Object Behaviors
   */
  //@{
  /** Constructor */
  FESimulator(Mesh<dim, value_type>& triag, const unsigned int degree,
              const std::string& label);

  /** Constructor for two components */
  FESimulator(Mesh<dim, value_type>& triag,
              const dealii::FiniteElement<dim>& fe1, const unsigned int n1,
              const dealii::FiniteElement<dim>& fe2, const unsigned int n2,
              const std::string& label);

  /**
   * Copy constructor.
   * Note that \c DoFHandler has deleted copy constructor
   */
  FESimulator(const FESimulator<dim, FEType, LinsysType, void>&);

  /**
   * Copy assignment.
   * Note that \c DoFHandler has deleted copy assignment
   */
  FESimulator<dim, FEType, LinsysType, void>& operator=(
    const FESimulator<dim, FEType, LinsysType, void>&);
  //@}


  /**
   * \name Accessors to Members
   */
  //@{
  /** FE object */
  const FEType& fe() const
  {
    ASSERT(ptr_fe.get(), ExcNullPointer());
    return *ptr_fe;
  }

  /** Accessing affine constraints in the linear system */
  typename LinsysType::constraints_type& constraints() const
  {
    ASSERT(ptr_linear_system, ExcNullPointer());
    return ptr_linear_system->constraints;
  }

  /** Getting linear system object */
  LinsysType& linear_system() { return *ptr_linear_system; }

  /** const overload of the previous function */
  const LinsysType& linear_system() const { return *ptr_linear_system; }
  //@}



  /**
   * \name Mesh Operations
   */
  //@{
  /**
   * Actions to be done whenever mesh update is detected.
   * Set mesh_update_detected to \c true,
   * redistribute dofs and set flags in linear system for repopulation.
   */
  void upon_mesh_update() override;

  /**
   * Access mesh refiner.
   */
  MeshRefiner<dim, value_type>& mesh_refiner()
  {
    ASSERT(ptr_mesh_refiner, ExcNullPointer());
    return *ptr_mesh_refiner;
  }


  /**
   * The actual worker to carry out
   * labelling cell for coarsening/refinement.
   */
  virtual void do_flag_mesh_for_coarsen_and_refine(
    const MeshControl<value_type>&) const
  {
    THROW(ExcUnimplementedVirtualFcn());
  }
  //@}


  /**
   * Synchronize all time-dependent members
   * in the simulator class.
   * These members may include boundary conditions
   * and source terms. However, this will not affect
   * the time registered in \c tempo_integrator and \c solution_time.
   */
  void set_time_temporal_passive_members(time_step_type current_time);


  /**
   * Append a temporal passive member to
   * the \c temporal_passive_members vector.
   */
  void add_temporal_passive_member(
    const std::shared_ptr<TimedPhysicsObject<time_step_type>>& p_member);


  /**
   * BC Collection object
   */
  bcs_type bcs;


  /**
   * Finite element object
   */
  std::shared_ptr<const FEType> ptr_fe;


  /**
   * Quadrature Type
   */
  std::shared_ptr<const dealii::Quadrature<dim>> ptr_quadrature;


  /**
   * Linear system
   */
  std::shared_ptr<LinsysType> ptr_linear_system;


  /**
   * Pointer to the solution vector
   */
  TimedSolutionVector<vector_type, time_step_type> solution;


  /**
   * Pointer to \c MeshRefiner.
   * This object will be responsible for carry out mesh refinement
   * and inform other simulators connected to same mesh to transfer
   * (or, interpolate solution) to the new mesh.
   * If the pointer os \c NULL, then methods dealing with
   * mesh refinement will not execute.
   */
  std::shared_ptr<MeshRefiner<dim, value_type>> ptr_mesh_refiner = nullptr;


  /**
   * Polymorphic pointer to \c BatchSolutionTransfer.
   * Each simulator is free to define its own its implementation as long as
   * the implementation has the the following interfaces:
   * \c prepare_for_coarsening_and_refinement()
   * \c interpolate()
   */
  std::shared_ptr<SolutionTransferBase> ptr_solution_transfer = nullptr;


  /**
   * These objects will be updated when \c set_time_temporal_passive_members()
   * is called and these will also be tested for synchronization.
   */
  std::vector<std::weak_ptr<TimedPhysicsObject<time_step_type>>>
    temporal_passive_members;


  /**
   * Whether \c DoFHandler, \c LinearSystem and \c Solution
   * is controlled by this object
   */
  bool primary_simulator;


  /**
   * Mesh refinement pointer empty.
   */
  DECL_EXCEPT_0(
    ExcMeshRefinerEmpty,
    "Trying to execute mesh refinement while mesh refiner pointer is empty");
};

FELSPA_NAMESPACE_CLOSE

/* -------- Template Implementations ----------*/
#include "src/pde_base.implement.h"
/* -------------------------------------------*/

#endif  // _FELSPA_PDE_PDE_BASE_H_
