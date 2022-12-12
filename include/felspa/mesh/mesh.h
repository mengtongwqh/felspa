#ifndef _FELSPA_MESH_MESH_H_
#define _FELSPA_MESH_MESH_H_

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria.h>
#include <felspa/base/felspa_config.h>
#include <felspa/base/log.h>
#include <felspa/base/types.h>

#include <algorithm>

FELSPA_NAMESPACE_OPEN

// forward declaration
template <int dim, typename NumberType>
class MeshWrapper;

/** Alias the Mesh object */
template <int dim, typename NumberType = types::DoubleType>
using Mesh = MeshWrapper<dim, NumberType>;

namespace internal
{
#ifdef FELSPA_HAS_MPI
  template <int dim>
  using MeshImpl = ::dealii::parallel::distributed::Triangulation<dim>;
#else
  template <int dim>
  using MeshImpl = ::dealii::Triangulation<dim>;
#endif

}  // namespace internal

/* ************************************************** */
/**
 * Mesh Related control Parameters
 */
/* ************************************************** */
template <typename NumberType>
struct MeshControl
{
  using value_type = NumberType;

  /**
   * Set coarsen and refinement level limit
   */
  void set_coarsen_refine_limit(int coarsen_limit, int refine_limit);


  /**
   *  Set refine and coarsen fraction
   */
  void set_coarsen_refine_fraction(value_type coarsen_fraction,
                                   value_type refine_fraction);


  /**
   * Coarsening of the mesh will not exceed this level.
   */
  int min_level = 2;

  /**
   * Refinement of the mesh will not exceed this level.
   */
  int max_level = 5;

  /**
   * Top fraction of the cells to be refined.
   */
  value_type refine_top_fraction = 0.05;

  /**
   * Bottom fraction of the cells to be coarsened.
   */
  value_type coarsen_bottom_fraction = 0.5;

  /**
   * Refinement will be carried out after this many time step cycles.
   */
  unsigned int refinement_interval = 0;
};


/* ************************************************** */
/**
 * Wrapper over the dealii::Triangulation to cache some
 * mesh parameters.
 */
/* ************************************************** */
template <int dim, typename NumberType>
class MeshWrapper : public internal::MeshImpl<dim>
{
 public:
  struct MeshInfo;
  using value_type = NumberType;
  using boundary_id_type = dealii::types::boundary_id;

  /**
   * \name Basic Object Behavior
   */
  //@{
  /**
   * @brief  Constructor
   */
  MeshWrapper(const typename dealii::Triangulation<dim>::MeshSmoothing
                smooth_grid = dealii::Triangulation<dim>::none);

  /**
   * @brief Destructor 
   */
  ~MeshWrapper() = default;
  //@}


  /**
   * \name Questing Info About Triangulation
   */
  //@{
  /**
   * Getter for MeshInfo struct
   */
  const MeshInfo& get_info() const { return info; }

  /**
   * Compute the max boundary id on the triangulation.
   * Useful for computing boundary condition id.
   */
  boundary_id_type max_boundary_id() const;
  //@}


 private:
  /**
   * Called when mesh update is detected.
   */
  void update_info();


  /**
   * Mesh information to be recorded.
   */
  MeshInfo info;
};


template <int dim, typename NumberType>
struct MeshWrapper<dim, NumberType>::MeshInfo
{
  /** Maximum cell diameter */
  value_type max_diameter = 0.0;

  /** Minimum cell diameter */
  value_type min_diameter = 0.0;

  /** Maximum refinement level */
  int max_level = 0;

  /** Minimum refinement level */
  int min_level = 0;

  /** Boundary ID */
  dealii::types::boundary_id max_boundary_id = 0;
};


FELSPA_NAMESPACE_CLOSE

/* ------ IMPLEMENTATIONS ------- */
#include "src/mesh.implement.h"
/* ------------------------------ */

#endif  // _FELSPA_MESH_MESH_H_ //
