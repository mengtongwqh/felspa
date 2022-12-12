#ifndef _FELSPA_MESH_MESH_IMPLEMENT_H_
#define _FELSPA_MESH_MESH_IMPLEMENT_H_

#include <felspa/base/exceptions.h>
#include <felspa/mesh/mesh.h>

FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/*          \class MeshControl                        */
/* ************************************************** */
template <typename NumberType>
void MeshControl<NumberType>::set_coarsen_refine_limit(int coarsen_limit,
                                                       int refine_limit)
{
  ASSERT(coarsen_limit > 0, ExcArgumentCheckFail());
  ASSERT(refine_limit > 0, ExcArgumentCheckFail());
  ASSERT(coarsen_limit <= refine_limit, ExcArgumentCheckFail());

  min_level = coarsen_limit;
  max_level = refine_limit;
}


template <typename NumberType>
void MeshControl<NumberType>::set_coarsen_refine_fraction(
  value_type coarsen_fraction, value_type refine_fraction)
{
  ASSERT(coarsen_fraction >= 0.0 && coarsen_fraction <= 1.0,
         ExcArgumentCheckFail());
  ASSERT(refine_fraction >= 0.0 && refine_fraction <= 1.0,
         ExcArgumentCheckFail());

  refine_top_fraction = refine_fraction;
  coarsen_bottom_fraction = coarsen_fraction;
}


/* ************************************************** */
/*          \class MeshWrapper                        */
/* ************************************************** */
template <int dim, typename NumberType>
MeshWrapper<dim, NumberType>::MeshWrapper(
  const typename dealii::Triangulation<dim>::MeshSmoothing smooth_grid)
  :
#ifdef FELSPA_HAS_MPI
    internal::MeshImpl<dim>(MPI_COMM_WORLD, smooth_grid)
#else
    internal::MeshImpl<dim>(smooth_grid, /*check_for_distorted_cells*/ false)
#endif
{
  this->signals.post_refinement.connect(
    0, boost::bind(&MeshWrapper<dim, NumberType>::update_info, this));
  this->signals.create.connect(
    0, boost::bind(&MeshWrapper<dim, NumberType>::update_info, this));
}


template <int dim, typename NumberType>
void MeshWrapper<dim, NumberType>::update_info()
{
  std::vector<value_type> diam;
  std::vector<int> levels;
  for (const auto& cell : this->active_cell_iterators()) {
    diam.push_back(cell->diameter());
    levels.push_back(cell->level());
  }

  info.max_diameter = *(std::max_element(diam.begin(), diam.end()));
  info.min_diameter = *(std::min_element(diam.begin(), diam.end()));
  info.max_level = *(std::max_element(levels.begin(), levels.end()));
  info.min_level = *(std::min_element(levels.begin(), levels.end()));

#ifdef VERBOSE
  LOG_PREFIX("MeshWrapper");
  felspa_log << "Mesh updated with " << this->n_active_cells()
             << " active cells, cell diameter range [" << info.min_diameter
             << ", " << info.max_diameter << ']' << std::endl;
#endif  // VERBOSE //
}


template <int dim, typename NumberType>
auto MeshWrapper<dim, NumberType>::max_boundary_id() const -> boundary_id_type
{
  const auto ids = this->get_boundary_ids();
  return *std::max_element(ids.begin(), ids.end());
}

FELSPA_NAMESPACE_CLOSE
#endif  // _FELSPA_MESH_MESH_IMPLEMENT_H_ //
