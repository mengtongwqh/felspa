#ifndef _FELSPA_MESH_MESH_REFINE_H_
#define _FELSPA_MESH_MESH_REFINE_H_

#include <deal.II/base/smartpointer.h>
#include <deal.II/dofs/dof_handler.h>
#include <felspa/base/felspa_config.h>
#include <felspa/mesh/mesh.h>

#include <map>
#include <vector>

FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/**
 * Centralizes vector interpolation through one interface.
 */
/* ************************************************** */
struct SolutionTransferBase : public dealii::Subscriptor
{
  /**
   * Prepare the solution for coarsening and refinement.
   */
  virtual void prepare_for_coarsening_and_refinement() = 0;

  /**
   * Interpolate the solution to the new grid.
   */
  virtual void interpolate() = 0;
};


/* ************************************************** */
/**
 * Operate on the cell refinement/coarsening flags.
 */
/* ************************************************** */
template <int dim, typename NumberType>
class MeshFlagOperator
{
 public:
  using value_type = NumberType;

  /**
   * Constructor
   */
  MeshFlagOperator(Mesh<dim, value_type>& mesh);


  /**
   * Copy constructor - deleted.
   */
  MeshFlagOperator(const MeshFlagOperator<dim, value_type>&) = delete;


  /**
   * Copy assignment - deleted.
   */
  MeshFlagOperator<dim, value_type>& operator=(
    const MeshFlagOperator<dim, value_type>&) = delete;


  /**
   * Limit the coarsening and refinement level of the mesh.
   * If any cell is above/below the upper/lower level limit,
   * cancel the refinement/coarsening flag.
   */
  void limit_level(int min_level, int max_level) const;


  /**
   * When this is called, the most updated flags of the mesh will be
   * compared with cached flags. If there is a clash between refinement
   * and coarsening flag, refinement flag will be prioritized.
   */
  void prioritize_refinement() const;


  /**
   * Print the initial and current count of coarsening/refinement flags
   */
  template <typename OstreamType>
  void print_info(OstreamType& os);


 private:
  /**
   * Pointer to the mesh object.
   */
  Mesh<dim, NumberType>* const ptr_mesh;


  /**
   * Refinement flags that were cached when the object is constructed.
   */
  std::vector<bool> cached_refine_flags;


  /**
   * Coarsening flags that were cached when the object is constructed.
   */
  std::vector<bool> cached_coarsen_flags;
};


/* ************************************************** */
/**
 * A class that actually carries out mesh refinement.
 * When the coarsen_and_refine is called, mesh refinement
 * will be carried out immediately, and the solution vectors that belong to
 * those simulators which have their solution_transfer object registered
 * in this object will be interpolated to the new grid automatically.
 */
/* ************************************************** */
template <int dim, typename NumberType>
class MeshRefiner final : public dealii::Subscriptor
{
 public:
  constexpr static int dimension = dim;
  using value_type = NumberType;


  /**
   * \name Basic Object Behavior
   */
  //@{
  /**
   * Constructor
   */
  MeshRefiner(Mesh<dim, value_type>& mesh) : ptr_mesh(&mesh) {}

  /**
   * Destructor
   */
  ~MeshRefiner() = default;

  /**
   * Do not allow copy construction.
   */
  MeshRefiner(const MeshRefiner<dim, NumberType>&) = delete;

  /**
   * Do not allow fopy assignment
   */
  MeshRefiner<dim, NumberType>& operator=(const MeshRefiner<dim, NumberType>&) =
    delete;
  //@}


  /**
   * Access the mesh
   */
  Mesh<dim, value_type>& mesh() { return *ptr_mesh; }


  /**
   * Prepare for grid refiement, do refinement and interpolate solutions
   */
  void run_coarsen_and_refine(bool run_solution_transfer = true);


  /** \name Adding/Clearing Solution Vectors */
  //@{
  /**
   * Adding solution vector to the object along with the
   * DoFHandler controlling it.
   */
  void append(const std::shared_ptr<SolutionTransferBase>& soln_trans);

  /**
   * Add a simulator with a solution transfer object.
   */
  template <typename SimulatorType>
  void append(const SimulatorType& sim);

  /**
   * Clear all entries in the soln_transfers
   */
  void clear() { solution_transfers.clear(); }
  //@}


  /**
   * When cells that need to be refined / coarsened is labelled,
   * the \c update_pending will be set to \c true.
   */
  void set_update_pending() { update_pending = true; }


  /**
   * return \c update_pending flag.
   */
  bool has_update_pending() const { return update_pending; }


 private:
  /**
   * Pointer to the mesh object.
   */
  const dealii::SmartPointer<Mesh<dim, NumberType>,
                             MeshRefiner<dim, NumberType>>
    ptr_mesh;


  /**
   * Will be set to \c true when mesh is labelled for coarsening/refinement.
   */
  bool update_pending = false;


  /**
   * These solutions will be transferred to new grid whenever
   * grid refinement/coarsening is done.
   * Each simulator class will define its derived \c SolutionTransfer
   * object to implement interpolation.
   */
  std::vector<std::shared_ptr<SolutionTransferBase>> solution_transfers;
};

FELSPA_NAMESPACE_CLOSE

/* ----- Template Implementations ----- */
#include "src/mesh_refine.implement.h"
/* ------------------------------------ */

#endif  // _FELSPA_MESH_MESH_REFINE_H_
