#ifndef _FELSPA_PDE_BOUNDARY_CONDITIONS_IMPLEMENT_H_
#define _FELSPA_PDE_BOUNDARY_CONDITIONS_IMPLEMENT_H_

#include <felspa/pde/boundary_conditions.h>

FELSPA_NAMESPACE_OPEN

template <int dim, typename NumberType>
template <typename MeshType>
void PeriodicBCFunction<dim, NumberType>::collect_face_pairs(
  const MeshType& mesh,
  std::vector<
    dealii::GridTools::PeriodicFacePair<typename MeshType::cell_iterator>>&
    face_pairs) const
{
  ASSERT(ptr_geometry_pair.first, ExcNullPointer());
  ASSERT(ptr_geometry_pair.second, ExcNullPointer());

  dealii::GridTools::collect_periodic_faces(
    mesh,
    ptr_geometry_pair.first->get_boundary_id(),
    ptr_geometry_pair.second->get_boundary_id(),
    direction,
    face_pairs,
    offset_vector,
    rotation_matrix);
}


template <int dim, typename NumberType>
template <typename MeshType>
auto PeriodicBCFunction<dim, NumberType>::collect_periodic_faces(
  const MeshType& mesh) const
{
  ASSERT(ptr_geometry_pair.first, ExcNullPointer());
  ASSERT(ptr_geometry_pair.second, ExcNullPointer());

  std::vector<
    dealii::GridTools::PeriodicFacePair<typename MeshType::cell_iterator>>
    periodicity_vector;
  dealii::GridTools::collect_periodic_faces(
    mesh,
    ptr_geometry_pair.first->get_boundary_id(),
    ptr_geometry_pair.second->get_boundary_id(),
    direction,
    periodicity_vector,
    offset_vector,
    rotation_matrix);
  return periodicity_vector;
}


FELSPA_NAMESPACE_CLOSE

#endif  // _FELSPA_PDE_BOUNDARY_CONDITIONS_IMPLEMENT_H_ //
