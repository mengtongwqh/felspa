/* ************************************************** */
/**
 * A simple test case to couple the level set simulator and
 * Stokes flow solver.
 */
/* ************************************************** */

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <felspa/coupled/level_set_stokes.h>
#include <felspa/pde/boundary_conditions.h>
#include <felspa/physics/functions.h>
#include <felspa/physics/viscous_flow.h>

#include <cstddef>
#include <filesystem>
using namespace felspa;


template <int dim, typename NumberType = double>
class Porphyroclast
{
 public:
  using value_type = NumberType;
  static constexpr int dimension = dim;
  struct Parameters;

  /**
   * Constructor
   */
  Porphyroclast(unsigned int advect_degree,
                unsigned int stokes_degree_velocity,
                unsigned int stokes_degree_pressure,
                Parameters& param);


  void initialize();


  /**
   * Run the simulator till the end_time.
   */
  void run(const value_type total_time);

  /**
   * Construct the simulation mesh
   */
  void create_mesh();


  /**
   * Exporting mesh
   */
  void export_mesh(const std::string& filename);


  /**
   * Model parameters
   */
  Parameters& prm;


  /**
   * Populate the mesh with the desired geometry.
   */
  Mesh<dim, value_type> mesh;


  /**
   * Coupled simulator object.
   */
  dg::LevelSetStokesSimulator<dim, value_type> simulator;

  /**
   * @brief Initial condition
   */
  std::shared_ptr<ls::ICBase<dim, value_type>> ptr_ic;

  /**
   * Pointer to mesh refiner
   */
  std::shared_ptr<MeshRefiner<dim, value_type>> ptr_mesh_refiner = nullptr;


  /**
   * @brief Export the mass error and time step
   */
  ExportFile time_step_data;
};


/* ************************************************** */
/**
 * @brief List of parameters used for the model
 */
/* ************************************************** */
template <int dim, typename NumberType>
struct Porphyroclast<dim, NumberType>::Parameters
{
  using value_type = NumberType;
  using point_type = dealii::Point<dim, value_type>;

  /** Constructor */
  Parameters();

  unsigned int level_of_fill = 0;
  StokesSolutionMethod solution_method;

  /** Kinematics */
  //@{
  value_type velocity_magnitude = 1.0;
  point_type box_lower;
  point_type box_upper;
  value_type radius = 0.25;
  value_type gravity_constant = 0.0;
  value_type coarsest_mesh_interval = 0.5;
  //@}


  /** Mesh configurations */
  //@{
  std::vector<unsigned int> subdivisions;
  unsigned int max_refine_level = 3;
  unsigned int min_refine_level = 1;
  bool use_adaptive_refinement = true;
  //@}

  /** Material Parameters */
  //@{
  value_type matrix_density = 1000.0;
  value_type matrix_viscosity = 1.0;
  value_type porphyroclast_density = 1000.0;
  value_type porphyroclast_viscosity = 10.0;
  //@}

  value_type artificial_viscosity = 0.000;

  std::string export_path;
};


/* ************************************************** */
/**                IMPLEMENTATIONS                    */
/* ************************************************** */
template <int dim, typename NumberType>
Porphyroclast<dim, NumberType>::Parameters::Parameters()
{
  if constexpr (dim == 2) {
    box_lower = dealii::Point<dim, value_type>(-2.0, -0.5);
    box_upper = dealii::Point<dim, value_type>(2.0, 0.5);
  }

  else if (dim == 3) {
    box_lower = dealii::Point<dim, value_type>(-2.0, -0.5, -0.5);
    box_upper = dealii::Point<dim, value_type>(2.0, 0.5, 0.5);
  }

  for (int idim = 0; idim < dim; ++idim) {
    unsigned int n = static_cast<unsigned int>(
      (box_upper(idim) - box_lower(idim)) / coarsest_mesh_interval);
    subdivisions.push_back(n);
  }
}


template <int dim, typename NumberType>
Porphyroclast<dim, NumberType>::Porphyroclast(
  unsigned int advect_degree,
  unsigned int stokes_degree_pressure,
  unsigned int stokes_degree_velocity,
  Parameters& param)
  : mesh(),
    prm(param),
    simulator(mesh,
              advect_degree,
              stokes_degree_pressure,
              stokes_degree_velocity,
              "Porphyroclast"),
    time_step_data(prm.export_path + "PorphyroclastTimeStepData.csv")
{
#if !defined(FELSPA_HAS_MPI) && defined(FELSPA_STOKES_USE_CUSTOM_ILU)
  simulator.control().ptr_stokes->solution_method = prm.solution_method;
  auto& solver_control = static_cast<StokesSolverControl&>(
    *simulator.control().ptr_stokes->ptr_solver);
  simulator.control().ptr_stokes->set_level_of_fill(prm.level_of_fill, 0);
#endif

  simulator.control().ptr_level_set->execute_reinit = true;
  simulator.control().ptr_level_set->ptr_reinit->ptr_tempo->set_cfl(0.1, 0.3);
  simulator.control().ptr_level_set->ptr_tempo->set_cfl(0.3, 0.5);

  // porphyroclast_test.simulator.control()
  //   .ptr_level_set->ptr_reinit->additional_control.global_solve = true;

  simulator.control().set_refine_reinit_interval(40);
  simulator.control().ptr_stokes->reference_viscosity = prm.matrix_viscosity;
  std::ostream& csv = time_step_data.access_stream();
  csv << "time,mass_error\n";

  // construct the mesh
  create_mesh();

  // add boundary condition
  simulator.set_bcs_stokes(
    {std::make_shared<bc::LinearSimpleShear<dim, value_type>>(
      prm.box_lower, prm.box_upper, prm.velocity_magnitude)});

  // now construct the porphyroclast material and level set
  ptr_ic = std::make_shared<ls::HyperSphere<dim, value_type>>(
    dealii::Point<dim, value_type>(), prm.radius);

  if (prm.use_adaptive_refinement) {
    ptr_mesh_refiner = std::make_shared<MeshRefiner<dim, value_type>>(mesh);
    simulator.attach_mesh_refiner(ptr_mesh_refiner, false);
    simulator.control().set_coarsen_refine_limit(prm.min_refine_level,
                                                 prm.max_refine_level);
  }

  simulator.control()
    .ptr_level_set->ptr_reinit->ptr_ldg->ptr_assembler->viscosity =
    prm.artificial_viscosity;

  auto p_porphyroclast_material =
    std::make_shared<NewtonianFlow<dim, value_type>>(
      prm.porphyroclast_density, prm.porphyroclast_viscosity, "Porphyroclast");

  simulator.add_material_domain(p_porphyroclast_material, ptr_ic);
  simulator.finalize_material_domains();
}


template <int dim, typename NumberType>
void Porphyroclast<dim, NumberType>::initialize()
{
  auto p_matrix_material = std::make_shared<NewtonianFlow<dim, value_type>>(
    prm.matrix_density, prm.matrix_viscosity);
  auto p_gravity_model =
    std::make_shared<GravityFunction<dim, value_type>>(prm.gravity_constant);
  simulator.initialize(p_matrix_material, nullptr);
  // simulator.setup_particles(1, false, 0, true);

#ifdef DEBUG
  simulator.get_materials().print(std::cout);
  if constexpr (dim == 2) export_mesh("Porphyroclast_after_initialization");
#endif  // DEBUG
}


template <int dim, typename NumberType>
void Porphyroclast<dim, NumberType>::create_mesh()
{
  dealii::GridGenerator::subdivided_hyper_rectangle(
    mesh, prm.subdivisions, prm.box_lower, prm.box_upper);
#ifdef DEBUG
  if constexpr (dim == 2) export_mesh("Porphyroclast_after_creation");
#endif  // DEBUG
}


template <int dim, typename NumberType>
void Porphyroclast<dim, NumberType>::run(const value_type total_time)
{
  double remaining_time = total_time;
  double cumulative_time = 0.0;

  while (!numerics::is_zero(remaining_time > 0.0)) {
    double dt = simulator.advance_time(remaining_time, true);
    ptr_mesh_refiner->run_coarsen_and_refine();
    remaining_time -= dt;
    cumulative_time += dt;

    simulator.export_solutions(prm.export_path);
    simulator.control().ptr_stokes->write_solver_statistics(
      prm.export_path + '/' + "SolverStats");

    std::ostream& csv = time_step_data.access_stream();
    csv << std::scientific << cumulative_time << ','
        << this->simulator.get_level_set(0).compute_mass_error(*ptr_ic)
        << std::endl;

    felspa_log << " >>>>> TIME STEP: " << dt
               << ", TOTAL SIMLATION TIME = " << cumulative_time << " <<<<<\n"
               << std::endl;
  }
}


template <int dim, typename NumberType>
void Porphyroclast<dim, NumberType>::export_mesh(const std::string& filename)
{
  std::ofstream out(filename + ".svg");
  dealii::GridOut grid_out;
  grid_out.write_svg(mesh, out);
}


/* ************************************************** */
/*                    Main Program                    */
/* ************************************************** */
int main(int argc, char* argv[])
{
  constexpr int dim = 3;

#ifdef FELSPA_HAS_MPI
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 4);
#endif

  felspa_log.depth_console(6);
  // dealii::deallog.depth_console(6);

  std::vector<StokesSolutionMethod> solution_methods{FC};
  std::vector<unsigned int> max_refine_levels{4};
  std::vector<unsigned int> lofs{1,0};
  std::vector<double> viscos{10.0};
  Porphyroclast<dim>::Parameters param;


  try {
    for (auto visco : viscos)
      for (auto lof : lofs)
        for (auto max_refine_level : max_refine_levels)
          for (auto solution_method : solution_methods) {
            param.max_refine_level = max_refine_level;
            param.solution_method = solution_method;
            param.level_of_fill = lof;
            param.porphyroclast_viscosity = visco;
            param.export_path =
              "Porphyroclast_" + 
              util::int_to_string(static_cast<int>(visco)) + '_' +
              util::int_to_string(max_refine_level) + '_' +
              util::int_to_string(lof) + '_' + to_string(solution_method) + '/';

            std::filesystem::create_directory(param.export_path);

            Porphyroclast<dim> porphyroclast_test(1, 2, 1, param);
            porphyroclast_test.initialize();
            porphyroclast_test.run(0.5);
          }
  }

  catch (const ExceptionBase& e) {
    error_info(e);
  }
  catch (const dealii::ExceptionBase& e) {
    std::cerr << e.what() << std::endl;
    return (EXIT_FAILURE);
  }
  catch (...) {
    std::cerr << "Unknown exception" << std::endl;
    return (EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
