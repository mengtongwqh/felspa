#include "test_rayleigh_taylor.h"

#include <filesystem>
#include <memory>

/**
 * @brief Rayleigh Taylor test class
 * @tparam dim  spatial dimension
 * @tparam NumberType  time pf
 */
template <int dim, typename NumberType = double>
class RayleighTaylor
{
 public:
  using value_type = NumberType;
  using point_type = dealii::Point<dim, value_type>;

  RayleighTaylor(unsigned int advect_degree, unsigned int stokes_degree_v,
                 unsigned int stokes_degree_p,
                 RayleighTaylorParameters<dim, NumberType>& prm,
                 const std::string& export_path,
                 const std::string& simulator_label);

  void initialize();

  void run(const value_type total_time);

  dg::LevelSetStokesSimulator<dim, value_type>& access_simulator()
  {
    return simulator;
  }


 private:
  void create_mesh(const point_type& lower, const point_type& upper,
                   const std::vector<unsigned int>& subdivisions);

  Mesh<dim, value_type> mesh;

  std::shared_ptr<MeshRefiner<dim, value_type>> ptr_mesh_refiner;

  dg::LevelSetStokesSimulator<dim, value_type> simulator;

  std::shared_ptr<ls::ICBase<dim, value_type>> ptr_ic;

  std::vector<std::shared_ptr<BCFunctionBase<dim, value_type>>> ptrs_bc;

  RayleighTaylorParameters<dim, value_type>& prm;

  std::string export_path;

  ExportFile csv_file;
};

template <int dim, typename NumberType>
RayleighTaylor<dim, NumberType>::RayleighTaylor(
  unsigned int advect_degree, unsigned int stokes_degree_v,
  unsigned int stokes_degree_p, RayleighTaylorParameters<dim, NumberType>& prm_,
  const std::string& path, const std::string& simulator_label)
  : simulator(mesh, advect_degree, stokes_degree_v, stokes_degree_p,
              simulator_label),
    export_path(path),
    prm(prm_),
    csv_file(path + "RayleighTaylorTimeStepData.csv")

{
  simulator.control().ptr_stokes->reference_length = 1.0;
  simulator.control().ptr_stokes->reference_viscosity = 100.0;
  simulator.control().ptr_stokes->solution_method = prm_.solution_method;
  auto& solver_control = static_cast<StokesSolverControl&>(
    *simulator.control().ptr_stokes->ptr_solver);
  solver_control.apply_diagonal_scaling = false;

#if !defined(FELSPA_HAS_MPI) && defined(FELSPA_STOKES_USE_CUSTOM_ILU)
  simulator.control().ptr_stokes->set_level_of_fill(prm.level_of_fill, 0);
#endif

  std::ostream& csv = csv_file.access_stream();
  csv << "time,velocity_norm, mass_error\n";

  create_mesh(prm.box_lower, prm.box_upper, prm.subdivisions);

  // setup bc
  ptrs_bc.push_back(
    std::make_shared<NoSlipBC<dim, value_type>>(prm.box_lower, prm.box_upper));
  ptrs_bc.push_back(std::make_shared<FreeSlipBC<dim, value_type>>(
    prm.box_lower, prm.box_upper));
  simulator.set_bcs_stokes({ptrs_bc[0], ptrs_bc[1]});
  simulator.get_stokes_simulator().get_bcs().print(std::cout);
  mesh.refine_global(prm.min_refine_level);


  // setup material and level set
  point_type center_pt = (prm.box_lower + prm.box_upper) * 0.5;
  center_pt[dim - 1] = 0.2;
  if constexpr (dim == 2) {
    center_pt[0] = 0.0;
    ptr_ic = std::make_shared<ls::RayleighTaylorLower<dim, value_type>>(
      center_pt, 0.9142, 0.02);
  } else if (dim == 3) {
    ptr_ic = std::make_shared<ls::RayleighTaylorLower<dim, value_type>>(
      center_pt, 1.0, 0.02);
  }

  // if (prm.use_adaptive_refinement) {
    ptr_mesh_refiner = std::make_shared<MeshRefiner<dim, value_type>>(mesh);
    simulator.attach_mesh_refiner(ptr_mesh_refiner, false);
    simulator.control().set_coarsen_refine_limit(prm.min_refine_level,
                                                 prm.max_refine_level);
    if (!prm.use_adaptive_refinement)
    mesh.refine_global(prm.max_refine_level);
  // }
  simulator.control().ptr_stokes->set_material_parameters_to_export(
    std::set<MaterialParameter>{MaterialParameter::density,
                                MaterialParameter::viscosity});
  auto p_material_lower = std::make_shared<NewtonianFlow<dim, value_type>>(
    prm.lower_density, prm.lower_viscosity, "RTLower");
  simulator.add_material_domain(p_material_lower, ptr_ic,
                                prm.use_adaptive_refinement);
  simulator.finalize_material_domains();
}


template <int dim, typename NumberType>
void RayleighTaylor<dim, NumberType>::initialize()
{
  auto p_matrix_material = std::make_shared<NewtonianFlow<dim, value_type>>(
    prm.upper_density, prm.upper_viscosity);
  auto p_gravity_model = std::make_shared<GravityFunction<dim, value_type>>(
    constants::earth_gravity);
  simulator.initialize(p_matrix_material, p_gravity_model);
  simulator.setup_particles(1, true, 0);
  simulator.get_materials().print(std::cout);
}


template <int dim, typename NumberType>
void RayleighTaylor<dim, NumberType>::run(const value_type total_time)
{
  double remaining_time = total_time;
  double cumulative_time = 0.0;

  while (!numerics::is_zero(remaining_time)) {
#if 0
  for (auto& cell : mesh.active_cell_iterators())
    for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary())
        std::cout << cell->face(f)->center() << " --> "
                  << cell->face(f)->boundary_id() << std::endl;
#endif

    double dt = simulator.advance_time(remaining_time, true);

    ptr_mesh_refiner->run_coarsen_and_refine();


    remaining_time -= dt;
    cumulative_time += dt;
    simulator.export_solutions(export_path);
    value_type velocity_norm = compute_velocity_norm(
      simulator.get_stokes_simulator(), dealii::QGauss<dim>(3));


    value_type volume = 1.0;
    for (int idim = 0; idim < dim; ++idim)
      volume *= (prm.box_upper[idim] - prm.box_lower[idim]);

    std::ostream& csv = csv_file.access_stream();
    csv << std::scientific << cumulative_time << ',' << velocity_norm / volume
        << ',' << this->simulator.get_level_set(0).compute_mass_error(*ptr_ic)
        << std::endl;

    if constexpr (dim == 2)
      felspa_log << " >>>>> TIME STEP: " << dt
                 << ", TOTAL SIMLATION TIME = " << cumulative_time
                 << " seconds, VELOCITY NORM(m/s) = " << std::scientific
                 << velocity_norm / volume << std::defaultfloat << " <<<<<\n"
                 << std::endl;

    else if (dim == 3)
      felspa_log << " >>>>> TIME STEP: " << dt
                 << ", TOTAL SIMLATION TIME = " << cumulative_time
                 << " seconds, VELOCITY NORM(m/s) = " << std::scientific
                 << velocity_norm / volume << std::defaultfloat << " <<<<<\n"
                 << std::endl;
  }
}


template <int dim, typename NumberType>
void RayleighTaylor<dim, NumberType>::create_mesh(
  const point_type& lower, const point_type& upper,
  const std::vector<unsigned int>& subdivisions)
{
  dealii::GridGenerator::subdivided_hyper_rectangle(mesh, subdivisions, lower,
                                                    upper);
}


int main(int argc, char** argv)
{
#ifdef FELSPA_HAS_MPI
#ifdef DEBUG
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
#else
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 8);
#endif
#endif

  // create a folder with the current time and use that as the path
  std::string export_path;


  constexpr int dim = 3;
  felspa_log.depth_console(5);
  // dealii::deallog.depth_console(5);

  std::vector<int> LoFs;
  std::vector<double> lower_visco = {10.0};
  std::vector<int> max_refine_levels;
  if constexpr (dim == 2) {
    max_refine_levels = {5};
    LoFs = {0};
  } else if (dim == 3) {
    max_refine_levels = {4};
#ifdef FELSPA_HAS_MPI
    LoFs = {0};
#else
    LoFs = {0};
#endif
  }

  try {
    for (auto max_refine_level : max_refine_levels) {
      for (auto visco : lower_visco) {
        for (auto lof : LoFs) {
          if (argc == 1)
            export_path = "./";
          else if (argc == 2)
            export_path = argv[1];
          else
            std::cerr << "Unexpected input arguments. Abort.\n"
                      << "Usage: example_rayleigh_taylor [export_path]"
                      << std::endl;

          RayleighTaylorParameters<dim, double> prm;
          prm.level_of_fill = lof;
          prm.lower_viscosity = visco;
          prm.max_refine_level = max_refine_level;
          prm.solution_method = FC;
          std::string simulator_label = "RT" + to_string(prm.solution_method);

          std::string secondary_export_path =
            "RayleighTaylor_" + util::int_to_string(prm.upper_viscosity) + '-' +
            util::int_to_string(static_cast<int>(visco)) + '_' +
            util::int_to_string(max_refine_level) + '_' +
            util::int_to_string(lof) + '_' + to_string(prm.solution_method) +
            '/';
          export_path += secondary_export_path;
          // now create this directory
          std::filesystem::create_directory(export_path);


          RayleighTaylor<dim> rayleigh_taylor(1, 2, 1, prm, export_path,
                                              simulator_label);
          rayleigh_taylor.initialize();
          rayleigh_taylor.run(500);
          rayleigh_taylor.access_simulator()
            .control()
            .ptr_stokes->write_solver_statistics(
              export_path + '/' + "SolverStats_" +
              util::int_to_string(prm.upper_viscosity) + '_' +
              util::int_to_string(prm.lower_viscosity) + '_' +
              util::int_to_string(lof) + '_' +
              util::int_to_string(max_refine_level));
        }
      }
    }
  }
  catch (const ExceptionBase& e) {
    error_info(e);
  }
  catch (const dealii::ExceptionBase& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  catch (...) {
    std::cerr << "Unknown exception" << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
