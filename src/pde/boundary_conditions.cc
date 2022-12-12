#include <deal.II/base/types.h>
#include <deal.II/grid/tria_iterator_base.h>
#include <felspa/base/utilities.h>
#include <felspa/pde/boundary_conditions.h>

#include <algorithm>

FELSPA_NAMESPACE_OPEN

/* ---------------------------- *
 * enum class BCCategory
 * ---------------------------- */
std::ostream& operator<<(std::ostream& os, BCCategory bc_category)
{
  switch (bc_category) {
    case BCCategory::dirichlet:
      os << "Dirichlet";
      break;
    case BCCategory::neumann:
      os << "Neumann";
      break;
    case BCCategory::robin:
      os << "Robin";
      break;
    case BCCategory::periodic:
      os << "Periodic";
      break;
    case BCCategory::no_normal_flux:
      os << "NoNormalFlux";
      break;
    default:
      THROW(ExcInternalErr());
  }
  os << " b.c.";
  return os;
}


/* ---------------------------- *
 * BCGeometry
 * ---------------------------- */
template <int dim, typename NumberType>
auto BCGeometry<dim, NumberType>::get_boundary_id() const -> size_type
{
  ASSERT_WARN(boundary_id != dealii::numbers::invalid_boundary_id,
              EXCEPT_MSG("The boundary id has not been properly set"));
  return boundary_id;
}


/* ---------------------------- *
 * \class BCFunctionBase
 * ---------------------------- */
template <int dim, typename NumberType>
BCFunctionBase<dim, NumberType>::BCFunctionBase(
  BCCategory bc_category,
  unsigned int n_component,
  const dealii::ComponentMask& component_mask_,
  time_type initial_time)
  : dealii::Function<dim, NumberType>(n_component, initial_time),
    category(bc_category),
    component_mask(component_mask_)
{}


template <int dim, typename NumberType>
void BCFunctionBase<dim, NumberType>::set_time(time_type curr_time)
{
  dealii::Function<dim, NumberType>::set_time(curr_time);
}


template <int dim, typename NumberType>
bool BCFunctionBase<dim, NumberType>::is_synchronized(
  time_type current_time) const
{
  return numerics::is_zero(current_time - this->get_time());
}


template <int dim, typename NumberType>
const dealii::ComponentMask&
BCFunctionBase<dim, NumberType>::get_component_mask() const
{
  return component_mask;
}


/* ---------------------------- *
 * \class BCFunction
 * ---------------------------- */
template <int dim, typename NumberType>
BCFunction<dim, NumberType>::BCFunction(
  BCCategory bc_category, unsigned int n_components,
  const dealii::ComponentMask& component_mask, time_type initial_time)
  : BCFunctionBase<dim, NumberType>(bc_category, n_components, component_mask,
                                    initial_time)
{}


template <int dim, typename NumberType>
void BCFunction<dim, NumberType>::set_geometry(
  const std::shared_ptr<BCGeometry<dim, NumberType>>& ptr_geom)
{
  ASSERT(ptr_geom != nullptr, ExcNullPointer());
  ptr_geometry = ptr_geom;
}


template <int dim, typename NumberType>
auto BCFunction<dim, NumberType>::get_boundary_id() const -> size_type
{
  ASSERT(ptr_geometry != nullptr, EXCEPT_MSG("BCGeometry is not set"));
  return ptr_geometry->get_boundary_id();
}


/* ------------------------------- *
 * \class PeriodicBCFunction
 * ------------------------------- */
template <int dim, typename NumberType>
PeriodicBCFunction<dim, NumberType>::PeriodicBCFunction(
  unsigned int n_component, const dealii::ComponentMask& component_mask,
  time_type initial_time)
  : BCFunctionBase<dim, NumberType>(BCCategory::periodic, n_component,
                                    component_mask, initial_time)
{
  ptr_geometry_pair.first = nullptr;
  ptr_geometry_pair.second = nullptr;
}


template <int dim, typename NumberType>
void PeriodicBCFunction<dim, NumberType>::set_geometry(
  const std::shared_ptr<BCGeometry<dim, value_type>>& ptr_geom1,
  const std::shared_ptr<BCGeometry<dim, value_type>>& ptr_geom2,
  const dealii::Tensor<1, dim, value_type>& offset,
  const dealii::FullMatrix<value_type>& rotation, int direction_)
{
  ASSERT(ptr_geom1 != nullptr, ExcNullPointer());
  ASSERT(ptr_geom2 != nullptr, ExcNullPointer());

  ptr_geometry_pair.first = ptr_geom1;
  ptr_geometry_pair.second = ptr_geom2;

  offset_vector = offset;
  rotation_matrix = rotation;
  direction = direction_;
}


template <int dim, typename NumberType>
void PeriodicBCFunction<dim, NumberType>::set_geometry(
  const std::shared_ptr<BCGeometry<dim, value_type>>& ptr_geom1,
  const std::shared_ptr<BCGeometry<dim, value_type>>& ptr_geom2,
  int direction_)
{
  ASSERT(0 <= direction_ && direction_ < dim, ExcArgumentCheckFail());
  ASSERT(ptr_geom1 != nullptr, ExcNullPointer());
  ASSERT(ptr_geom2 != nullptr, ExcNullPointer());

  ptr_geometry_pair.first = ptr_geom1;
  ptr_geometry_pair.second = ptr_geom2;

  direction = direction_;
  offset_vector = dealii::Tensor<1, dim, value_type>();
  rotation_matrix = dealii::FullMatrix<value_type>();
}


/* ------------------------------- *
 * \class BCBookKeeper
 * ------------------------------- */
template <int dim, typename NumberType>
BCBookKeeper<dim, NumberType>::BCBookKeeper(Mesh<dim, value_type>& triag)
  : TimedPhysicsObject<NumberType>(0.0), ptr_mesh(&triag)
{
  cache_boundary_faces();
}


template <int dim, typename NumberType>
void BCBookKeeper<dim, NumberType>::initialize(Mesh<dim, value_type>& triag)
{
  if (!empty()) clear();
  ptr_mesh = &triag;
  cache_boundary_faces();
}


template <int dim, typename NumberType>
void BCBookKeeper<dim, NumberType>::cache_boundary_faces()
{
  ASSERT(ptr_mesh->n_levels() > 0, ExcEmptyTriangulation());
  ASSERT(
    ptr_mesh->n_levels() == 1,
    EXCEPT_MSG("The boundary face must be cached at the coarsest mesh level."));
  boundary_iterators.clear();

  // cache all active cells that contain the boundary
  for (auto& cell : ptr_mesh->active_cell_iterators())
    for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary())
        boundary_iterators.push_back(cell->face(f));
}


template <int dim, typename NumberType>
bool BCBookKeeper<dim, NumberType>::empty() const
{
  ASSERT(ptrs_geometry_set.empty() == category_to_pbcs.empty(),
         ExcInternalErr());
  ASSERT(ptrs_geometry_set.empty() == category_to_pbcs.empty(),
         ExcInternalErr());
  return ptrs_geometry_set.empty();
}


template <int dim, typename NumberType>
bool BCBookKeeper<dim, NumberType>::is_initialized() const
{
  return ptr_mesh && (!boundary_iterators.empty());
}


template <int dim, typename NumberType>
void BCBookKeeper<dim, NumberType>::append(
  const std::shared_ptr<BCFunction<dim, NumberType>>& ptr_bc_fcn)
{
  ASSERT(ptr_bc_fcn->get_category() != BCCategory::periodic,
         EXCEPT_MSG("Periodic BC must be added with PeriodicBCPair."));
  ASSERT(!boundary_iterators.empty(), ExcEmptyBoundaryCellIterators());
  ASSERT(ptr_bc_fcn->ptr_geometry, ExcNullPointer());
  using dealii::numbers::invalid_boundary_id;

  const auto ptr_geom = ptr_bc_fcn->ptr_geometry;

  // try to add the geometry to the set
  if (auto [bdry_id, status] = try_insert_geometry(*ptr_geom); status) {
    ASSERT(bdry_id_to_pbcs.find(bdry_id) == bdry_id_to_pbcs.end(),
           ExcInternalErr());
    // set the updated boundary id
    ptr_geom->boundary_id = bdry_id;
  } else {
#ifdef DEBUG
    // double check it does also exist in the map
    ASSERT(bdry_id_to_pbcs.find(ptr_bc_fcn->ptr_geometry->get_boundary_id()) !=
             bdry_id_to_pbcs.end(),
           ExcInternalErr());
#endif  // DEBUG //
  }

  // add this entry to bdry_id -- bc_fcn map
  bdry_id_to_pbcs[ptr_geom->get_boundary_id()].push_back(ptr_bc_fcn.get());
  // register this boundary entry in category_to_bcs map
  category_to_pbcs[ptr_bc_fcn->get_category()].push_back(ptr_bc_fcn);
}


template <int dim, typename NumberType>
void BCBookKeeper<dim, NumberType>::append(
  const std::shared_ptr<PeriodicBCFunction<dim, value_type>>& pbc)
{
  ASSERT(!boundary_iterators.empty(), ExcEmptyBoundaryCellIterators());
  ASSERT(pbc->ptr_geometry_pair.first, ExcNullPointer());
  ASSERT(pbc->ptr_geometry_pair.second, ExcNullPointer());
  ASSERT(
    pbc->ptr_geometry_pair.first != pbc->ptr_geometry_pair.second,
    EXCEPT_MSG(
      "Pair of the Boundary Conditions should describe different boundaries."));

  auto [bdry_id1, status1] = try_insert_geometry(*pbc->ptr_geometry_pair.first);
  auto [bdry_id2, status2] =
    try_insert_geometry(*pbc->ptr_geometry_pair.second);

  // enforce periodic constraints on the mesh
  std::vector<dealii::GridTools::PeriodicFacePair<
    typename Mesh<dim, value_type>::cell_iterator>>
    matched_face_pairs;
  dealii::GridTools::collect_periodic_faces(
    static_cast<dealii::Triangulation<dim>&>(*ptr_mesh),
    bdry_id1,
    bdry_id2,
    pbc->direction,
    matched_face_pairs,
    pbc->offset_vector,
    pbc->rotation_matrix);
  ptr_mesh->add_periodicity(matched_face_pairs);

  // register boundary id in BCGeometry
  if (status1) pbc->ptr_geometry_pair.first->boundary_id = bdry_id1;
  if (status2) pbc->ptr_geometry_pair.second->boundary_id = bdry_id2;

  bdry_id_to_pbcs[bdry_id1].push_back(pbc.get());
  bdry_id_to_pbcs[bdry_id2].push_back(pbc.get());
  category_to_pbcs[BCCategory::periodic].push_back(pbc);
}


template <int dim, typename NumberType>
void BCBookKeeper<dim, NumberType>::clear()
{
  for (auto& bface : boundary_iterators) bface->set_all_boundary_ids(0);
  boundary_iterators.clear();
  ptrs_geometry_set.clear();
  category_to_pbcs.clear();
  bdry_id_to_pbcs.clear();
}


template <int dim, typename NumberType>
FELSPA_FORCE_INLINE bool BCBookKeeper<dim, NumberType>::has_category(
  BCCategory category) const
{
  return category_to_pbcs.find(category) != category_to_pbcs.end();
}


template <int dim, typename NumberType>
FELSPA_FORCE_INLINE bool BCBookKeeper<dim, NumberType>::has_boundary_id(
  boundary_id_type bid) const
{
  return bdry_id_to_pbcs.find(bid) != bdry_id_to_pbcs.end();
}


template <int dim, typename NumberType>
auto BCBookKeeper<dim, NumberType>::operator()(BCCategory category) const
  -> std::vector<const BCFunctionBase<dim, value_type>*>
{
  ASSERT(has_category(category), ExcCategoryNotInBCBookKeeper(category));
  std::vector<const BCFunctionBase<dim, value_type>*> bcs;
  for (const auto& pbc : category_to_pbcs.at(category))
    bcs.push_back(pbc.get());
  return bcs;
}


template <int dim, typename NumberType>
auto BCBookKeeper<dim, NumberType>::operator()(boundary_id_type bdry_id) const
  -> std::vector<const BCFunctionBase<dim, value_type>*>
{
  ASSERT(has_boundary_id(bdry_id), ExcBdryIdNotInBCBookKeeper(bdry_id));
  return bdry_id_to_pbcs.at(bdry_id);
}


template <int dim, typename NumberType>
auto BCBookKeeper<dim, NumberType>::get_time() const -> time_type
{
  return TimedPhysicsObject<value_type>::get_time();
}


template <int dim, typename NumberType>
void BCBookKeeper<dim, NumberType>::set_time(time_type current_time)
{
  for (auto& [category, pbcs] : category_to_pbcs)
    for (auto& pbc : pbcs) pbc->set_time(current_time);
  this->phsx_time = current_time;
  ASSERT(is_synchronized(), ExcNotSynchronized());
}


template <int dim, typename NumberType>
void BCBookKeeper<dim, NumberType>::advance_time(time_type time_step)
{
  for (auto& [category, pbcs] : category_to_pbcs)
    for (auto& pbc : pbcs) pbc->advance_time(time_step);
  this->phsx_time += time_step;
  ASSERT(is_synchronized(), ExcNotSynchronized());
}


template <int dim, typename NumberType>
FELSPA_FORCE_INLINE bool BCBookKeeper<dim, NumberType>::is_synchronized(
  time_type current_time) const
{
  return numerics::is_zero(current_time - get_time()) && is_synchronized();
}


template <int dim, typename NumberType>
bool BCBookKeeper<dim, NumberType>::is_synchronized() const
{
  for (const auto& [bdry_id, pbcs] : bdry_id_to_pbcs)
    for (const auto& pbc : pbcs)
      if (!this->is_synced_with(pbc->get_time())) return false;
  return true;
}


template <int dim, typename NumberType>
auto BCBookKeeper<dim, NumberType>::boundary_id_at_point(
  const dealii::Point<dim>& pt) const -> boundary_id_type
{
  ASSERT(point_has_unique_bdry_id(pt), ExcDuplicateGeometry(pt));
  for (const auto& ptr_geom : ptrs_geometry_set)
    if (ptr_geom->at_boundary(pt)) return ptr_geom->get_boundary_id();
  return 0;
}


template <int dim, typename NumberType>
void BCBookKeeper<dim, NumberType>::print(std::ostream& os) const
{
  os << "*** List of Existing Boundary Conditions: ***\n";

  for (const auto& [bdry_id, pbcs] : bdry_id_to_pbcs) {
    os << " Boundary ID = " << bdry_id << " | BC Category: ";
    for (const auto& pbc : pbcs) os << pbc->get_category() << ' ';
    os << '\n';
  }

  os << "******\n";
}


template <int dim, typename NumberType>
auto BCBookKeeper<dim, NumberType>::next_available_boundary_id() const
  -> size_type
{
  return ptr_mesh->max_boundary_id() + 1;
}


template <int dim, typename NumberType>
auto BCBookKeeper<dim, NumberType>::try_insert_geometry(
  const BCGeometry<dim, value_type>& bc_geometry) -> std::pair<size_type, bool>
{
  ASSERT(is_initialized(), ExcNotInitialized());
  using dealii::numbers::invalid_boundary_id;

  if (auto [it, status] = ptrs_geometry_set.insert(&bc_geometry); status) {
    // this is inserted, resolve the boundary id
    size_type existing_bdry_id = invalid_boundary_id;

    // grab the first bdry id we can get
    for (const auto& bface : boundary_iterators)
      if (bc_geometry.at_boundary(bface->center())) {
        existing_bdry_id = bface->boundary_id();
        break;
      }
    ASSERT(existing_bdry_id != invalid_boundary_id,
           EXCEPT_MSG(
             "The BCGeometry being added does not correspond to any boundary"));

    if (existing_bdry_id == 0) {
      // there is no bdry id associated with this boundary
      size_type bdry_id = next_available_boundary_id();
      for (auto& bface : boundary_iterators)
        if (bc_geometry.at_boundary(bface->center()))
          bface->set_all_boundary_ids(bdry_id);

      return {bdry_id, true};
    } else {
#ifdef DEBUG
      // make sure the bdry_id is consistent through the geometry definition
      // i.e. BCGeometry has the same boundary id
      ASSERT(check_bdry_id_geometry_consistency(existing_bdry_id, bc_geometry),
             ExcInternalErr());
#endif  // DEBUG //
      return {existing_bdry_id, true};
    }

  } else {
    // the entry is already there.
    size_type bdry_id = bc_geometry.get_boundary_id();
    ASSERT(bdry_id > 0 && bdry_id != dealii::numbers::invalid_boundary_id,
           ExcInternalErr());
    return {bdry_id, false};
  }
}

template <int dim, typename NumberType>
bool BCBookKeeper<dim, NumberType>::point_has_unique_bdry_id(
  const dealii::Point<dim, value_type>& pt) const
{
  unsigned int counter = 0;
  for (const auto& pbc : ptrs_geometry_set)
    if (pbc->at_boundary(pt)) ++counter;
  return counter == 1;
}


template <int dim, typename NumberType>
bool BCBookKeeper<dim, NumberType>::check_bdry_id_geometry_consistency(
  size_type bdry_id, const BCGeometry<dim, value_type>& bc_geom) const
{
  for (const auto& face : boundary_iterators) {
    // bdry id => BCGeometry
    if (face->boundary_id() == bdry_id)
      if (!bc_geom.at_boundary(face->center())) return false;
    // BCGeometry => bdry_id
    if (bc_geom.at_boundary(face->center()))
      if (face->boundary_id() != bdry_id) return false;
  }
  return true;
}


/* -------------------------------------------*/
namespace bc
/* -------------------------------------------*/
{
  /* ************************************************** */
  /* \class LidDrivenCavity */
  /* ************************************************** */
  template <int dim, typename NumberType>
  class LidDrivenCavity<dim, NumberType>::Geometry
    : public BCGeometry<dim, value_type>
  {
   public:
    Geometry(value_type lower_bound_, value_type upper_bound_)
      : lower_bound(lower_bound_), upper_bound(upper_bound_)
    {
      ASSERT(lower_bound < upper_bound, ExcArgumentCheckFail());
    }


    bool at_boundary(const dealii::Point<dim, value_type>& pt) const
    {
      using numerics::is_zero;
      for (int idim = 0; idim < dim; ++idim)
        if (is_zero(pt(idim) - lower_bound) || is_zero(pt(idim) - upper_bound))
          return true;
      return false;
    }


   protected:
    value_type lower_bound;
    value_type upper_bound;
  };


  template <int dim, typename NumberType>
  LidDrivenCavity<dim, NumberType>::LidDrivenCavity(value_type velo_magnitude_,
                                                    value_type lower_bound_,
                                                    value_type upper_bound_)
    : BCFunction<dim, NumberType>(BCCategory::dirichlet, dim + 1),
      velo_magnitude(velo_magnitude_),
      lower_bound(lower_bound_),
      upper_bound(upper_bound_)
  {
    ASSERT(lower_bound < upper_bound, ExcArgumentCheckFail());
    if constexpr (dim == 1) {
      std::vector<bool> mask = {true, false};
      this->component_mask = mask;
    } else if (dim == 2) {
      std::vector<bool> mask = {true, true, false};
      this->component_mask = mask;
    } else if (dim == 3) {
      std::vector<bool> mask = {true, true, true, false};
      this->component_mask = mask;
    } else {
      THROW(ExcNotImplemented());
    }
    this->ptr_geometry = std::make_shared<Geometry>(lower_bound, upper_bound);
  }


  template <int dim, typename NumberType>
  auto LidDrivenCavity<dim, NumberType>::value(const dealii::Point<dim>& pt,
                                               unsigned int component) const
    -> value_type
  {
    ASSERT(this->ptr_geometry->at_boundary(pt), ExcPointNotOnBoundary<dim>(pt));
    ASSERT(component < this->n_components, ExcArgumentCheckFail());

    using numerics::is_zero;
    switch (dim) {
      case 1:
        THROW(ExcNotImplemented());
        break;
      case 2:  // fall through //
      case 3:
        return (this->at_top_boundary(pt))
                 ? ((component == 0) ? velo_magnitude : 0.0)
                 : 0.0;
        break;
      default:
        THROW(EXCEPT_MSG("Not implemented for dim < 1 or dim > 3"));
    };
  }


  template <int dim, typename NumberType>
  bool LidDrivenCavity<dim, NumberType>::at_top_boundary(
    const dealii::Point<dim, value_type>& pt) const
  {
    ASSERT(this->ptr_geometry->at_boundary(pt),
           EXCEPT_MSG("Requesting value from point not on boundary"));
    return (numerics::is_zero(pt(dim - 1) - upper_bound));
  }


  /* ************************************************** */
  /* \class LinearSimpleShear */
  /* ************************************************** */
  template <int dim, typename NumberType>
  class LinearSimpleShear<dim, NumberType>::Geometry
    : public BCGeometry<dim, NumberType>
  {
   public:
    Geometry(const point_type& lower_, const point_type& upper_)
      : lower(lower_), upper(upper_)
    {}

    bool at_boundary(const point_type& pt) const override
    {
      const static unsigned remaining_dims[3][2] = {{1, 2}, {0, 2}, {0, 1}};
      for (int idim = 0; idim < dim; ++idim) {
        if (numerics::is_zero(pt(idim) - lower(idim)) ||
            numerics::is_zero(pt(idim) - upper(idim))) {
#ifdef DEBUG
          // double check that the point is in the bounding box
          for (int jdim = 0; jdim < dim - 1; ++jdim) {
            unsigned int idx = remaining_dims[idim][jdim];
            ASSERT(pt(idx) >= lower(idx) && pt(idx) <= upper(idx),
                   ExcInternalErr());
          }
#endif  // DEBUG //
          return true;
        }
      }  // idim-loop
      return false;
    }


   protected:
    const point_type lower;
    const point_type upper;
  };


  template <int dim, typename NumberType>
  LinearSimpleShear<dim, NumberType>::LinearSimpleShear(
    const point_type& lower_, const point_type& upper_,
    value_type velocity_magnitude_)
    : BCFunction<dim, NumberType>(BCCategory::dirichlet, dim + 1),
      lower(lower_),
      upper(upper_),
      velocity_magnitude(velocity_magnitude_),
      center_level(0.5 * (lower(dim - 1) + upper(dim - 1))),
      half_thickness(0.5 * (upper(dim - 1) - lower(dim - 1)))
  {
#ifdef DEBUG
    for (int idim = 0; idim < dim; ++idim)
      ASSERT(lower_(idim) < upper_(idim), ExcArgumentCheckFail());
#endif  // DEBUG

    std::vector<bool> mask(dim + 1, true);
    mask[dim] = false;
    this->component_mask = mask;

    ASSERT(half_thickness > 0, ExcInternalErr());
    if constexpr (!(dim == 2 || dim == 3)) THROW(ExcWorksOnlyInSpaceDim(23));
    this->ptr_geometry = std::make_shared<Geometry>(lower_, upper_);
  }


  template <int dim, typename NumberType>
  auto LinearSimpleShear<dim, NumberType>::value(const point_type& pt,
                                                 unsigned int component) const
    -> value_type
  {
    ASSERT(component < this->n_components, ExcArgumentCheckFail());
    ASSERT(this->ptr_geometry->at_boundary(pt),
           EXCEPT_MSG("Requesting value from point not on boundary"));

    if (component == 0)
      return (pt(dim - 1) - center_level) / half_thickness * velocity_magnitude;
    else
      return 0.0;
  }


  /* ************************************************** */
  /* \class MidOceanRift */
  /* ************************************************** */
  template <int dim, typename NumberType>
  class MidOceanRift<dim, NumberType>::Geometry
    : public BCGeometry<dim, NumberType>
  {
   public:
    Geometry(const value_type& upper) : upper_bound(upper) {}

    bool at_boundary(const point_type& pt) const override
    {
      return numerics::is_zero(pt[dim - 1] - upper_bound);
    }

   protected:
    value_type upper_bound;
  };


  template <int dim, typename NumberType>
  MidOceanRift<dim, NumberType>::MidOceanRift(value_type velo_magnitude_,
                                              value_type upper_bound_)
    : BCFunction<dim, value_type>(BCCategory::dirichlet, dim + 1,
                                  std::vector<bool>(dim + 1, true)),
      velo_magnitude(velo_magnitude_),
      upper_bound(upper_bound_)
  {
    ASSERT(velo_magnitude >= 0.0, ExcArgumentCheckFail());
    this->component_mask.set(dim, false);
    this->ptr_geometry = std::make_shared<Geometry>(upper_bound);
  }


  template <int dim, typename NumberType>
  auto MidOceanRift<dim, NumberType>::value(const dealii::Point<dim>& pt,
                                            unsigned int component) const
    -> value_type
  {
    ASSERT(component < this->n_components, ExcArgumentCheckFail());
    ASSERT(this->ptr_geometry->at_boundary(pt), ExcPointNotOnBoundary<dim>(pt));
    if (component == 0)
      return (pt[0] > 0.0) ? velo_magnitude
                           : (pt[0] < 0.0 ? -velo_magnitude : 0.0);
    return 0.0;
  }

}  // namespace bc

/* -------- Explicit Instantiations ----------*/
#include "boundary_conditions.inst"
/* -------------------------------------------*/

FELSPA_NAMESPACE_CLOSE
