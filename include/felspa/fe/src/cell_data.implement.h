#ifndef _FELSPA_FE_CELL_DATA_IMPLEMENT_H_
#define _FELSPA_FE_CELL_DATA_IMPLEMENT_H_

#include <felspa/base/constants.h>
#include <felspa/fe/cell_data.h>

#include <utility>

#include "felspa/base/felspa_config.h"
#include "felspa/base/numerics.h"

FELSPA_NAMESPACE_OPEN

/* ----------------------------------------- *
 *           CellScratchDataBox              *
 *------------------------------------------ */
template <typename CellScratchDataType>
template <typename... Args>
CellScratchDataBox<CellScratchDataType>::CellScratchDataBox(Args&&... args)
  : cell(FEValuesEnum::cell, std::forward<Args>(args)...),
    face_in(FEValuesEnum::face, std::forward<Args>(args)...),
    face_ex(FEValuesEnum::face, std::forward<Args>(args)...),
    subface(FEValuesEnum::subface, std::forward<Args>(args)...)
{}

template <typename CellScratchDataType>
CellScratchDataBox<CellScratchDataType>::CellScratchDataBox()
  : cell(FEValuesEnum::cell),
    face_in(FEValuesEnum::face),
    face_ex(FEValuesEnum::face),
    subface(FEValuesEnum::subface)
{}


template <typename CellScratchDataType>
template <template <int> class QuadratureType>
void CellScratchDataBox<CellScratchDataType>::add(
  const dealii::Mapping<dim>& mapping,
  const dealii::FiniteElement<dim>& fe,
  const QuadratureType<dim>& quad,
  const dealii::UpdateFlags update_flags)
{
  using namespace dealii;
  const unsigned int quad_1d_size = numerics::dimth_root<dim>(quad.size());

  cell.add(std::make_shared<FEValues<dim>>(mapping, fe, quad, update_flags));

  face_in.add(std::make_shared<FEFaceValues<dim>>(
    mapping, fe, QuadratureType<dim - 1>(quad_1d_size),
    update_flags | update_normal_vectors));

  face_ex.add(std::make_shared<FEFaceValues<dim>>(
    mapping, fe, QuadratureType<dim - 1>(quad_1d_size),
    update_flags | update_normal_vectors));

  subface.add(std::make_shared<FESubfaceValues<dim>>(
    mapping, fe, QuadratureType<dim - 1>(quad_1d_size),
    update_flags | update_normal_vectors));
}


template <typename CellScratchDataType>
void CellScratchDataBox<CellScratchDataType>::clear()
{
  cell.clear();
  face_in.clear();
  face_ex.clear();
  subface.clear();
}


template <typename CellScratchDataType>
template <typename CellItrType>
CellScratchDataType& CellScratchDataBox<CellScratchDataType>::reinit_cell(
  const CellItrType& cell_it)
{
  cell.reinit(cell_it);
  return cell;
}


template <typename CellScratchDataType>
template <typename CellItrType>
auto CellScratchDataBox<CellScratchDataType>::reinit_faces(
  const CellItrType& cell_it, const unsigned int face_no,
  const CellItrType& neighbor_it, const unsigned int neighbor_face_no)
  -> scratch_pair_type
{
  ASSERT(!cell_it->has_children(), ExcInternalErr());
  ASSERT(!neighbor_it->has_children(), ExcInternalErr());
  ASSERT(cell_it->level() == neighbor_it->level(), ExcInternalErr());

  if constexpr (std::is_same<CellItrType, synced_iterators_type>::value) {
    ASSERT(cell_it.is_synchronized(), ExcIteratorsNotSynced());
    ASSERT(neighbor_it.is_synchronized(), ExcIteratorsNotSynced());
  }

  // First reinit the internal face
  face_in.reinit(cell_it, face_no);
  // Initialize the exterior face
  face_ex.reinit(neighbor_it, neighbor_face_no);

  return scratch_pair_type(face_in, face_ex);
}


template <typename CellScratchDataType>
template <typename CellItrType>
auto CellScratchDataBox<CellScratchDataType>::reinit_faces(
  const CellItrType& cell_it, const unsigned int face_no,
  const CellItrType& neighbor_it, const unsigned int neighbor_face_no,
  const unsigned int neighbor_subface_no) -> scratch_pair_type
{
  ASSERT(!cell_it->has_children(), ExcInternalErr());
  ASSERT(!neighbor_it->has_children(), ExcInternalErr());
  ASSERT(cell_it->level() > neighbor_it->level(), ExcInternalErr());

  if constexpr (std::is_same<CellItrType, synced_iterators_type>::value) {
    ASSERT(cell_it.is_synchronized(), ExcIteratorsNotSynced());
    ASSERT(neighbor_it.is_synchronized(), ExcIteratorsNotSynced());
  }

  // First reinit the internal face
  face_in.reinit(cell_it, face_no);
  // Initialize the subface
  subface.reinit(neighbor_it, neighbor_face_no, neighbor_subface_no);

  return scratch_pair_type(face_in, subface);
}


template <typename CellScratchDataType>
template <typename CellItrType>
CellScratchDataType& CellScratchDataBox<CellScratchDataType>::reinit_face(
  const CellItrType& cell_it, const unsigned int face_no)
{
  face_in.reinit(cell_it, face_no);
  return face_in;
}


/* ----------------------------------------- *
 *             CellScratchData               *
 *------------------------------------------ */
template <int dim>
CellScratchData<dim>::CellScratchData(FEValuesEnum fevalenum)
  : feval_enum(fevalenum),
    face_no(constants::invalid_unsigned_int),
    subface_no(constants::invalid_unsigned_int)
{}


template <int dim>
CellScratchData<dim>::CellScratchData(
  FEValuesEnum fevalenum,
  std::initializer_list<std::shared_ptr<dealii::FEValuesBase<dim>>>
    ptrs_fe)
  : CellScratchData(fevalenum)
{
  for (auto i = ptrs_fe.begin(); i != ptrs_fe.end(); ++i) add(*i);
}


template <int dim>
CellScratchData<dim>::CellScratchData(const CellScratchData<dim>& that)
  : feval_enum(that.feval_enum), ptrs_feval(that.ptrs_feval.size())
{
  // copy the fevalues entries one by one
  auto it_src = that.ptrs_feval.cbegin();
  for (auto it_dst = ptrs_feval.begin(); it_dst != ptrs_feval.end();
       ++it_dst, ++it_src)
    *it_dst = duplicate_fe_values(*it_src);
}


template <int dim>
CellScratchData<dim>& CellScratchData<dim>::operator=(
  const CellScratchData<dim>& that)
{
  if (this != &that) {
    CellScratchData<dim> cell_local_data(that);
    this->move(cell_local_data);
  }
  return *this;
}


template <int dim>
auto CellScratchData<dim>::add(
  const std::shared_ptr<dealii::FEValuesBase<dim>>& pfeval) -> size_type
{
  ASSERT(pfeval.get(), ExcNullPointer());
  ptrs_feval.push_back(pfeval);
  return ptrs_feval.size();
}


template <int dim>
void CellScratchData<dim>::move(CellScratchData<dim>& that)
{
  ASSERT(feval_enum == that.feval_enum, ExcIncompatibleData());
  ptrs_feval = std::move(that.ptrs_feval);
}


template <int dim>
std::shared_ptr<dealii::FEValuesBase<dim>>
CellScratchData<dim>::duplicate_fe_values(
  const std::shared_ptr<dealii::FEValuesBase<dim>>& pfeval)
{
  ASSERT(pfeval.get(), ExcNullPointer());

  switch (feval_enum) {
    case FEValuesEnum::cell: {
      using FEVal = dealii::FEValues<dim>;
      const FEVal& fe = dynamic_cast<const FEVal&>(*pfeval);
      return std::make_shared<FEVal>(fe.get_mapping(), fe.get_fe(),
                                     fe.get_quadrature(),
                                     fe.get_update_flags());
    }
    case FEValuesEnum::face: {
      using FEFaceVal = dealii::FEFaceValues<dim>;
      const FEFaceVal& fe = dynamic_cast<const FEFaceVal&>(*pfeval);
      return std::make_shared<FEFaceVal>(fe.get_mapping(), fe.get_fe(),
                                         fe.get_quadrature(),
                                         fe.get_update_flags());
    }
    case FEValuesEnum::subface: {
      using FESubfaceVal = dealii::FESubfaceValues<dim>;
      const FESubfaceVal& fe = dynamic_cast<const FESubfaceVal&>(*pfeval);
      return std::make_shared<FESubfaceVal>(fe.get_mapping(), fe.get_fe(),
                                            fe.get_quadrature(),
                                            fe.get_update_flags());
    }
    default:
      THROW(ExcInternalErr());
  }
}


template <int dim>
void CellScratchData<dim>::reinit(const synced_iterators_type& cell)
{
  const auto& cell_iters = cell.get_iterators();
  ASSERT_SAME_SIZE(cell_iters, ptrs_feval);
  ASSERT(feval_enum == FEValuesEnum::cell,
         ExcFEValEnumMismatch(feval_enum, FEValuesEnum::cell));

  auto iter = cell_iters.cbegin();
  for (auto pfeval : ptrs_feval)
    static_cast<dealii::FEValues<dim>&>(*pfeval).reinit(*iter++);

  face_no = constants::invalid_unsigned_int;
  subface_no = constants::invalid_unsigned_int;
  cell_iter = cell.get(0);
}


template <int dim>
void CellScratchData<dim>::reinit(const synced_iterators_type& cell,
                                  const unsigned int face_no_)
{
  const auto& cell_iters = cell.get_iterators();
  ASSERT_SAME_SIZE(cell_iters, ptrs_feval);
  ASSERT(feval_enum == FEValuesEnum::face,
         ExcFEValEnumMismatch(feval_enum, FEValuesEnum::face));

  auto iter = cell_iters.cbegin();
  for (auto pfeval : ptrs_feval)
    static_cast<dealii::FEFaceValues<dim>&>(*pfeval).reinit(*iter++, face_no_);

  face_no = face_no_;
  subface_no = constants::invalid_unsigned_int;
  cell_iter = cell.get(0);
  face_iter = cell.get(0)->face(face_no_);
}


template <int dim>
void CellScratchData<dim>::reinit(const synced_iterators_type& cell,
                                  const unsigned int face_no_,
                                  const unsigned int subface_no_)
{
  const auto& cell_iters = cell.get_iterators();
  ASSERT_SAME_SIZE(cell_iters, ptrs_feval);
  ASSERT(feval_enum == FEValuesEnum::subface,
         ExcFEValEnumMismatch(feval_enum, FEValuesEnum::subface));

  auto iter = cell_iters.cbegin();
  for (auto pfeval : ptrs_feval)
    static_cast<dealii::FESubfaceValues<dim>&>(*pfeval).reinit(
      *iter++, face_no_, subface_no_);

  face_no = face_no_;
  subface_no = subface_no_;
  cell_iter = cell.get(0);
  face_iter = cell.get(0)->face(face_no_);
}


template <int dim>
void CellScratchData<dim>::reinit(const CellIterator& cell)
{
  ASSERT(feval_enum == FEValuesEnum::cell,
         ExcFEValEnumMismatch(feval_enum, FEValuesEnum::cell));

  for (auto& pfeval : ptrs_feval)
    static_cast<dealii::FEValues<dim>&>(*pfeval).reinit(cell);

  face_no = constants::invalid_unsigned_int;
  subface_no = constants::invalid_unsigned_int;
  cell_iter = cell;
}


template <int dim>
void CellScratchData<dim>::reinit(const CellIterator& cell,
                                  const unsigned int face_no_)
{
  ASSERT(feval_enum == FEValuesEnum::face,
         ExcFEValEnumMismatch(feval_enum, FEValuesEnum::face));

  for (auto& pfeval : ptrs_feval)
    static_cast<dealii::FEFaceValues<dim>&>(*pfeval).reinit(cell, face_no_);

  face_no = face_no_;
  subface_no = constants::invalid_unsigned_int;
  cell_iter = cell;
  face_iter = cell->face(face_no_);
}


template <int dim>
void CellScratchData<dim>::reinit(const CellIterator& cell,
                                  const unsigned int face_no_,
                                  const unsigned int subface_no_)
{
  ASSERT(feval_enum == FEValuesEnum::subface,
         ExcFEValEnumMismatch(feval_enum, FEValuesEnum::subface));

  for (auto& pfeval : ptrs_feval)
    static_cast<dealii::FESubfaceValues<dim>&>(*pfeval).reinit(cell, face_no_,
                                                               subface_no_);
  face_no = face_no_;
  subface_no = subface_no_;
  cell_iter = cell;
  face_iter = cell->face(face_no_);
}


/* ************************************************** */
/*              class CellCopyData                    */
/* ************************************************** */
template <int dim, typename NumberType>
FELSPA_FORCE_INLINE void CellCopyData<dim, NumberType>::allocate(size_type n)
{
  dof_indices.resize(n);
  local_vector.reinit(n);
  reset(true);
}


template <int dim, typename NumberType>
FELSPA_FORCE_INLINE bool CellCopyData<dim, NumberType>::is_active() const
{
  return active;
}


template <int dim, typename NumberType>
FELSPA_FORCE_INLINE auto CellCopyData<dim, NumberType>::vector()
  -> dealii::Vector<value_type>&
{
  return local_vector;
}

template <int dim, typename NumberType>
FELSPA_FORCE_INLINE void CellCopyData<dim, NumberType>::set_active()
{
  ASSERT_SAME_SIZE(dof_indices, local_vector);
  active = true;
}


template <int dim, typename NumberType>
FELSPA_FORCE_INLINE void CellCopyData<dim, NumberType>::reset(bool force)
{
  if (active || force) {
    local_vector = 0.0;
    std::fill(dof_indices.begin(), dof_indices.end(),
              constants::invalid_unsigned_int);
    active = false;
  }
}


template <int dim, typename NumberType>
void CellCopyData<dim, NumberType>::reinit(const CellIterator& cell_iter)
{
  ASSERT(
    dof_indices.size() == cell_iter->get_fe().dofs_per_cell,
    ExcSizeMismatch(dof_indices.size(), cell_iter->get_fe().dofs_per_cell));
  ASSERT(
    dof_indices.size() == cell_iter->get_fe().dofs_per_cell,
    ExcSizeMismatch(dof_indices.size(), cell_iter->get_fe().dofs_per_cell));
  cell_iter->get_dof_indices(dof_indices);
  local_vector = 0.0;
  active = true;
}


template <int dim, typename NumberType>
template <typename DstVectorType>
void CellCopyData<dim, NumberType>::assemble(
  const dealii::AffineConstraints<NumberType>& constraints,
  DstVectorType& dst_vector) const
{
  if (active) {
#ifdef DEBUG
    // make sure that the data is suitable for copying to global
    bool no_invalid_dof_idx =
      std::accumulate(dof_indices.cbegin(), dof_indices.cend(), true,
                      [](bool init, const size_t dof) {
                        return init && dof != constants::invalid_unsigned_int;
                      });
    ASSERT(no_invalid_dof_idx,
           EXCEPT_MSG("Some dof indices are not initialized."));
    ASSERT(!dof_indices.empty(), EXCEPT_MSG("DoF Indices vector is empty."));
    ASSERT_SAME_SIZE(dof_indices, local_vector);
#endif  // DEBUG

    constraints.distribute_local_to_global(local_vector, dof_indices,
                                           dst_vector);
  }
}
/* ************************************************** */
/*              class CellCopyDataBox                 */
/* ************************************************** */
template <typename CopyDataType>
CellCopyDataBox<CopyDataType>::CellCopyDataBox(size_type n)
{
  cell.allocate(n);
  for (auto& f : interior_faces) { f.allocate(n); }
  for (auto& f : exterior_faces) { f.allocate(n); }
}


template <typename CopyDataType>
void CellCopyDataBox<CopyDataType>::reset()
{
  cell.reset();
  for (auto& f : interior_faces) { f.reset(); }
  for (auto& f : exterior_faces) { f.reset(); }
}


template <typename CopyDataType>
template <typename DstVectorType>
void CellCopyDataBox<CopyDataType>::assemble(
  const dealii::AffineConstraints<value_type>& constraints,
  DstVectorType& rhs_vector) const
{
  cell.assemble(constraints, rhs_vector);
  for (auto& face : interior_faces) face.assemble(constraints, rhs_vector);
  for (auto& face : exterior_faces) face.assemble(constraints, rhs_vector);
}

FELSPA_NAMESPACE_CLOSE

#endif  // _FELSPA_FE_CELL_DATA_IMPLEMENT_H_ //
