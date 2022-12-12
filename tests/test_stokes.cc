#include <deal.II/base/function.h>
#include <deal.II/grid/grid_generator.h>
#include <felspa/base/exceptions.h>
#include <felspa/base/io.h>
#include <felspa/pde/boundary_conditions.h>
#include <felspa/pde/stokes.h>
#include <felspa/physics/viscous_flow.h>

#define VERBOSE

using namespace felspa;
using namespace dealii;


template <int dim, typename NumberType = felspa::types::DoubleType>
class GravityFunction : public felspa::TensorFunction<1, dim, NumberType>
{
 public:
  using value_type = NumberType;
  using base_type = felspa::TensorFunction<1, dim, NumberType>;
  using typename base_type::tensor_type;

  GravityFunction(value_type gravity = 0.0) : gravity_const(gravity) {}

  virtual tensor_type evaluate(const dealii::Point<dim>& pt) const override
  {
    UNUSED_VARIABLE(pt);
    tensor_type output;
    for (int idim = 0; idim < dim; ++idim)
      output[idim] = (idim == dim - 1) ? gravity_const : 0.0;
    return output;
  }

 private:
  value_type gravity_const;
};


int main(int argc, char* argv[])
{
  // deallog.depth_console(5);
  felspa_log.depth_console(5);

#ifdef FELSPA_HAS_MPI
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
#endif

  const int dim = 3;
  const unsigned int p_degree = 1;

  // Mesh
  Mesh<dim> mesh;
  // dealii::GridGenerator::hyper_cube(mesh, 0.0, 1.0);
  auto p_mesh_refiner = std::make_shared<MeshRefiner<dim, double>>(mesh);


  std::vector<unsigned int> subdivisions(dim, 1);
  subdivisions[0] = 4;
  const Point<dim> bottom_left = (dim == 2 ?  //
                                    Point<dim>(-2, -1)
                                           :                 // 2d case
                                    Point<dim>(-2, 0, -1));  // 3d case
  const Point<dim> top_right = (dim == 2 ?                   //
                                  Point<dim>(2, 0)
                                         :               // 2d case
                                  Point<dim>(2, 1, 0));  // 3d case
  dealii::GridGenerator::subdivided_hyper_rectangle(mesh, subdivisions,
                                                    bottom_left, top_right);

  // boundary conditions
  auto ptr_bc = std::make_shared<bc::MidOceanRift<dim, double>>(1.0, 0.0);
  // std::make_shared<bc::LidDrivenCavity<dim, double>>(1.0, 0.0, 1.0);
  auto ptr_material = std::make_shared<NewtonianFlow<dim, double>>(1.0e3, 1.0);
  auto v_source_rhs = std::make_shared<GravityFunction<dim>>(0.0);


  try {
    // impose the boundary conditions
    StokesSimulator<dim> stokes_solver(mesh, FE_Q<dim>(p_degree + 1),
                                       FE_Q<dim>(p_degree), "StokesCavity");
    stokes_solver.control().ptr_mesh->refinement_interval = 1;
    stokes_solver.control().ptr_mesh->refine_top_fraction = 0.3;
    stokes_solver.control().ptr_mesh->coarsen_bottom_fraction = 0.0;
    stokes_solver.control().ptr_mesh->max_level = 10;
    stokes_solver.attach_mesh_refiner(p_mesh_refiner);
    stokes_solver.append_boundary_condition(ptr_bc);
    mesh.refine_global(4 - dim);
    stokes_solver.initialize(ptr_material, v_source_rhs);
    for (unsigned int cycle = 0; cycle < 6; ++cycle) {
      if (cycle != 0) p_mesh_refiner->run_coarsen_and_refine();
      stokes_solver.advance_time(0.1);
      stokes_solver.export_solutions();
    }
  }
  catch (const ::felspa::ExceptionBase& exc) {
    std::cerr << exc << std::endl;
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
  catch (...) {
    std::cerr << "<unknown error>" << std::endl;
  }

  return EXIT_SUCCESS;
}
