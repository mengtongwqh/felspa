#include <deal.II/base/convergence_table.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <felspa/base/exceptions.h>
#include <felspa/base/felspa_config.h>
#include <felspa/base/function.h>
#include <felspa/base/io.h>
#include <felspa/base/log.h>
#include <felspa/level_set/geometry.h>
#include <felspa/level_set/level_set.h>
#include <felspa/level_set/reinit.h>
#include <felspa/level_set/velocity_field.h>
#include <felspa/mesh/mesh.h>
#include <felspa/pde/advection.h>
#include <felspa/pde/boundary_conditions.h>
#include <felspa/pde/time_integration.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <utility>

using namespace felspa;
using constants::PI;
using dealii::Point;

#define VERBOSE

/* ************************************************** */
/**
 * Test class for Zalesak Disk
 */
/* ************************************************** */
template <int dim, typename NumberType>
class ZalesakDiskTest
{
 public:
  using value_type = NumberType;
  using Advect = dg::AdvectSimulator<dim, NumberType>;
  using Reinit = dg::ReinitSimulator<dim, NumberType>;
  using vector_type  = typename Advect::vector_type;

  /**
   * Constructor
   */
  ZalesakDiskTest(unsigned int fe_degree, bool run_mesh_refine);


  /**
   * Run test for one cycle
   */
  void run(unsigned int cycle);

  /**
   * Export table
   */
  void output_table(std::ostream& os);

  /**
   * Control parameters
   */
  std::shared_ptr<ls::LevelSetControl<Advect, Reinit>> ptr_control;

 private:
  void process_error(const ScalarFunction<dim, value_type>& ic);
  void process_error(const vector_type& ic);

  Mesh<dim, value_type> mesh;

  ls::LevelSetSurface<Advect, Reinit> level_set_simulator;

  std::shared_ptr<MeshRefiner<dim, value_type>> ptr_mesh_refiner = nullptr;

  /** Velocity field */
  std::shared_ptr<ls::RigidBodyRotation<dim, value_type>> ptr_velocity_field;


  /** \name Initial conditions */
  //@{
  ls::HyperRectangle<dim, value_type> ic_rectangle;

  ls::HyperSphere<dim, value_type> ic_sphere;
  //@}

  dealii::ConvergenceTable convergence_table;

  /** Control flags and parameters */
  //@{
  const unsigned int n_revolution = 1;

  /** By default do not export vtk */
  bool export_vtk = true;

  unsigned int cycle;
  //@}
};


template <int dim, typename NumberType>
ZalesakDiskTest<dim, NumberType>::ZalesakDiskTest(unsigned int fe_degree,
                                                  bool adaptive_refine)
  : ptr_control(std::make_shared<ls::LevelSetControl<Advect, Reinit>>()),
    level_set_simulator(mesh, fe_degree, "ZalesakDisk"),
    ptr_velocity_field(std::make_shared<ls::RigidBodyRotation<dim, value_type>>(
      dealii::Point<dim>{0.0, 0.0}, PI / 314.0)),
    ic_rectangle(Point<dim, NumberType>(-2.499999, 15.0),
                 Point<dim, NumberType>(2.499999, 40.0)),
    ic_sphere(Point<dim, NumberType>(0.0, 25), 15.0)
{
  // set up the mesh
  dealii::GridGenerator::subdivided_hyper_cube(mesh, 2, -50.0, 50.0);
  level_set_simulator.attach_control(ptr_control);

  // mesh refiner
  if (adaptive_refine) {
    ptr_mesh_refiner = std::make_shared<MeshRefiner<dim, value_type>>(mesh);
    level_set_simulator.attach_mesh_refiner(ptr_mesh_refiner);
    ptr_control->ptr_mesh->min_level = 2;
    ptr_control->ptr_mesh->max_level = ptr_control->ptr_mesh->min_level + 2;
    ptr_control->ptr_mesh->refinement_interval = 10;
    ptr_control->reinit_frequency = 40;
    mesh.refine_global(ptr_control->ptr_mesh->min_level);
  } else {
    ptr_control->ptr_mesh->min_level = 4;
    ptr_control->reinit_frequency = 40;
    mesh.refine_global(ptr_control->ptr_mesh->min_level);
  }

  ptr_control->execute_reinit = true;
}


template <int dim, typename NumberType>
void ZalesakDiskTest<dim, NumberType>::run(unsigned int icycle)
{
  cycle = icycle;

  if (level_set_simulator.has_mesh_refiner()) {
    ++ptr_control->ptr_mesh->max_level;
  } else
    mesh.refine_global(1);

  auto ic = ic_sphere - ic_rectangle;

  level_set_simulator.initialize(ic, ptr_velocity_field);
  auto initial_solution = level_set_simulator.get_solution_vector();
  
  std::string dir_name{"./ZalesaksDisk_Cycle" + std::to_string(cycle)};
  
  level_set_simulator.export_solutions(dir_name);

  for (unsigned int i = 0; i < 628 * n_revolution; ++i) {
    level_set_simulator.advance_time(1.0, false);
    if (ptr_mesh_refiner) ptr_mesh_refiner->run_coarsen_and_refine();
    if (export_vtk) {
      if (i == 0 || i == 628 * n_revolution - 1)
        {
          level_set_simulator.export_solutions(dir_name);
        }
    }
  }  // i-loop

  // output final grid
  dealii::GridOut grid_out;
  const std::string grid_filename{"ZalesaksDiskGrid_cycle" +
                                  std::to_string(cycle) + ".svg"};
  std::ofstream svg_file(grid_filename);
  grid_out.write_svg(mesh, svg_file);

  process_error(initial_solution);
}


template <int dim, typename NumberType>
void ZalesakDiskTest<dim, NumberType>::process_error(
  const ScalarFunction<dim, value_type>& ic)
{
  auto norm_type = dealii::VectorTools::L2_norm;

  const double L2_error = level_set_simulator.compute_error(ic, norm_type);
  const unsigned int n_dofs = level_set_simulator.get_dof_handler().n_dofs();

  convergence_table.add_value("cycle", cycle);
  convergence_table.add_value("dofs", n_dofs);
  convergence_table.add_value("h", mesh.get_info().min_diameter);
  // convergence_table.add_value("cells", mesh.n_active_cells());
  convergence_table.add_value("L2", L2_error);
  convergence_table.add_value("mass_error",
                              level_set_simulator.compute_mass_error(ic));
}



template <int dim, typename NumberType>
void ZalesakDiskTest<dim, NumberType>::process_error(
  const vector_type& ic)
{
  // auto norm_type = dealii::VectorTools::L2_norm;

  const double L2_error = level_set_simulator.compute_error(ic);
  const unsigned int n_dofs = level_set_simulator.get_dof_handler().n_dofs();

  convergence_table.add_value("cycle", cycle);
  convergence_table.add_value("dofs", n_dofs);
  convergence_table.add_value("h", mesh.get_info().min_diameter);
  // convergence_table.add_value("cells", mesh.n_active_cells());
  convergence_table.add_value("L2", L2_error);
  // convergence_table.add_value("mass_error",
  //                             level_set_simulator.compute_mass_error(ic));
}


template <int dim, typename NumberType>
void ZalesakDiskTest<dim, NumberType>::output_table(std::ostream& stream)
{
  convergence_table.set_precision("L2", 5);
  convergence_table.set_scientific("L2", true);

  stream << std::endl;
  convergence_table.write_text(stream);

  // convergence_table.set_tex_caption("cells", "\\# cells");
  convergence_table.set_tex_caption("dofs", "\\# dofs");
  convergence_table.set_tex_caption("L2", "$L^2$-error");
  // convergence_table.set_tex_caption("mass_error", "Mass Error");

  std::string error_filename = "zalesak_disk_error.tex";
  std::ofstream error_table_file(error_filename);
  convergence_table.write_tex(error_table_file);
}

int main(int argc, char** argv)
{
#ifdef FELSPA_HAS_MPI
#ifdef DEBUG
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
#else
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 8);
#endif  // DEBUG //
#endif

  using value_type = double;
  felspa_log.depth_console(6);

  // Constants
  constexpr const int dim = 2;
  const unsigned int fe_degree = 1;
  const unsigned int n_cycles = 1;
  const value_type viscosity = 0.001;
  const bool adaptive_refine = false;

  ZalesakDiskTest<dim, value_type> test(fe_degree, adaptive_refine);
  test.ptr_control->set_artificial_viscosity(viscosity);
  test.ptr_control->ptr_reinit->ptr_tempo->set_cfl(0.1, 0.3);

  for (unsigned int icycle = 0; icycle < n_cycles; ++icycle) test.run(icycle);
  test.output_table(std::cout);

  return EXIT_SUCCESS;
}
