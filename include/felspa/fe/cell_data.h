#ifndef _FELSPA_FE_CELL_DATA_H_
#define _FELSPA_FE_CELL_DATA_H_

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <felspa/base/constants.h>
#include <felspa/base/exceptions.h>
#include <felspa/base/felspa_config.h>
#include <felspa/base/types.h>
#include <felspa/fe/sync_iterators.h>

#include <memory>

FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/**
 * Each cell has the following four entities.
 * - (\c dim)-dimension cell
 * - (\c dim-1)-dimensional face
 * - (\c dim-1)-dimensional subface
 */
/* ************************************************** */
enum class FEValuesEnum
{
  cell,
  face,
  subface
};

std::ostream& operator<<(std::ostream& os, FEValuesEnum assembly_entity);


/* ************************************************** */
/**
 * Collection of cell local data for
 * each cell assembly entity.
 */
/* ************************************************** */
template <typename CellScratchDataType>
class CellScratchDataBox
{
 public:
  constexpr static const int dim = CellScratchDataType::dimension;
  using scratch_data_type = CellScratchDataType;
  using CellIteratorType =
    typename dealii::DoFHandler<dim>::active_cell_iterator;
  using synced_iterators_type = SyncedActiveIterators<dim>;
  using scratch_pair_type =
    std::pair<CellScratchDataType&, CellScratchDataType&>;


  /**
   * Constructor
   */
  CellScratchDataBox();


  /**
   * Constructor with a parameter pack passed to \c CellScratchDataType.
   */
  template <typename... Args>
  CellScratchDataBox(Args&&... args);


  /**
   * Adding entry into cell local data.
   */
  template <template <int> class QuadratureType>
  void add(const dealii::Mapping<dim>& mapping,
           const dealii::FiniteElement<dim>& fe,
           const QuadratureType<dim>& quad,
           const dealii::UpdateFlags update_flags);


  /**
   * Run cleanup for \c cell, \c face_in, \c face_ex, \c subface.
   */
  void clear();


  /**
   * Copy Constructor, use default.
   */
  CellScratchDataBox(const CellScratchDataBox<CellScratchDataType>&) = default;


  /**
   * Copy assignment, use default.
   */
  CellScratchDataBox<CellScratchDataType>& operator=(
    const CellScratchDataBox<CellScratchDataType>& that) = default;


  /**
   * Virtual destructor
   */
  virtual ~CellScratchDataBox() = default;


  /** \name Reinit FEValues for Each Entity */
  //@{

  /**
   * Reinit local data on the given cell.
   */
  // CellScratchDataType& reinit_cell(const synced_iterators_type& cell);
  template <typename CellItrType>
  CellScratchDataType& reinit_cell(const CellItrType& cell_it);


  /**
   * Reinit local data on cell face and neighbor face.
   * When the neighboring face has the same level,
   * return \c face_in and \c face_ex.
   */
  template <typename CellItrType>
  scratch_pair_type reinit_faces(const CellItrType& cell_it,
                                 const unsigned int face_no,
                                 const CellItrType& neighbor_it,
                                 const unsigned int neighbor_face_no);


  /**
   * Reinit local data on cell face and neighbor face.
   * When the neighboring face is coarser,
   * return \c face_in and \c subface.
   */
  template <typename CellItrType>
  scratch_pair_type reinit_faces(const CellItrType& cell_it,
                                 const unsigned int face_no,
                                 const CellItrType& neighbor_it,
                                 const unsigned int neighbor_face_no,
                                 const unsigned int neighbor_subface_no);


  /**
   * Reinit only the internal face.
   * Return a reference to \c face_in.
   */
  // CellScratchDataType& reinit_face(const synced_iterators_type& cell,
  //                                  const unsigned int face_no);
  template <typename CellItrType>
  CellScratchDataType& reinit_face(const CellItrType& cell_it,
                                   const unsigned int face_no);
  //@}


 protected:
  /**
   * Data related to the current cell.
   */
  CellScratchDataType cell;

  /**
   * Data related to the current face.
   */
  CellScratchDataType face_in;

  /**
   * Data of the current face seen from the neighboring cell.
   */
  CellScratchDataType face_ex;

  /**
   * Data related to the a subface. This subface is usually an external one
   */
  CellScratchDataType subface;
};


/* ************************************************** */
/**
 * Local data for each cell assembly entity
 * (cell, face, neighbor, subface).
 * Include an \c FEValues<dim> object pertains to this entity.
 * The first entry of the sync_iteraor and ptrs_feval is the
 * dominant one.
 */
/* ************************************************** */
template <int dim>
class CellScratchData
{
  friend class CellScratchDataBox<CellScratchData<dim>>;

 public:
  using CellIterator = typename dealii::DoFHandler<dim>::active_cell_iterator;
  using FaceIterator = typename dealii::DoFHandler<dim>::face_iterator;
  using size_type = typename std::vector<CellIterator>::size_type;
  using synced_iterators_type = SyncedActiveIterators<dim>;

  constexpr static const int dimension = dim;


  /**
   * Constructor
   */
  CellScratchData(FEValuesEnum feval_enum);

  /**
   * Constructor that takes a list of pointers to \c FEValuesBase
   */
  CellScratchData(
    FEValuesEnum,
    std::initializer_list<std::shared_ptr<dealii::FEValuesBase<dim>>>);


  /**
   * Copy Constructor.
   * Pointer to FEValues type object will be deep copied.
   */
  CellScratchData(const CellScratchData<dim>&);


  /**
   * Copy Assignment operator
   */
  CellScratchData<dim>& operator=(const CellScratchData<dim>&);


  /**
   * Destructor. Memory will be automatically released by shared_ptr
   */
  virtual ~CellScratchData() = default;


  /**
   * Add one entry to the vector of pointers to \c FEValuesBase.
   */
  size_type add(const std::shared_ptr<dealii::FEValuesBase<dim>>&);


  /**
   * Clearing all entries in the vector of pointers.
   */
  void clear() { ptrs_feval.clear(); }


  /**
   * Category of the assembly entity.
   */
  const FEValuesEnum feval_enum;


  /**
   * Obtain a reference to the primary \c FEValues
   */
  dealii::FEValuesBase<dim>& fe_values()
  {
    ASSERT(!ptrs_feval.empty(), ExcInternalErr());
    return *ptrs_feval[0];
  }


  /**
   * Obtain a reference to the primary \c FEValues.
   * Const overload of the above function.
   */
  const dealii::FEValuesBase<dim>& fe_values() const
  {
    ASSERT(!ptrs_feval.empty(), ExcInternalErr());
    return *ptrs_feval[0];
  }


  /** \name Reinit the each \c FEValues entry */
  //@{
  /**
   * Reinit the cell data with synced iterators
   * Each iterator in synced_iterators_type
   * corresponds to an FEValues entry.
   */
  void reinit(const synced_iterators_type& cell);

  /**
   * Reinit the face data with synced iterators
   * Each iterator in synced_iterators_type
   * corresponds to an FEValues entry.
   */
  void reinit(const synced_iterators_type& cell, const unsigned int face_no);

  /**
   * Reinit the subface data with synced iterators
   * Each iterator in synced_iterators_type
   * corresponds to an FEValues entry.
   */
  void reinit(const synced_iterators_type& cell, const unsigned int face_no,
              const unsigned int subface_no);

  /**
   * Use one CellIterator to reinit all FEValues.
   */
  void reinit(const CellIterator& cell);

  /**
   * Use one CellIterator to reinit all FEFaceValues.
   */
  void reinit(const CellIterator& cell, const unsigned int face);

  /**
   * Use one CellIterator to reinit all FESubFaceValues.
   */
  void reinit(const CellIterator& cell, const unsigned int face_no,
              const unsigned int subface_no);
  //@}


  /**
   * Obtain current face no
   */
  unsigned int get_face_no() const { return face_no; }

  /**
   * Obtain current subface no
   */
  unsigned int get_subface_no() const { return subface_no; }

  /**
   * Obtain current cell iterator
   */
  const CellIterator& get_cell_iter() const { return cell_iter; }

  /**
   * Obtain current face iterator
   */
  const FaceIterator& get_face_iter() const { return face_iter; }

  /**
   * Obtain current cell accessor
   */
  const typename CellIterator::AccessorType& cell() const { return *cell_iter; }

  /**
   * Obtain current face accessor
   */
  const typename FaceIterator::AccessorType& face() const { return *face_iter; }


  DECL_EXCEPT_0(ExcIncompatibleData,
                "Different assembly entity cannot be swapped or assigned");

  DECL_EXCEPT_2(ExcFEValEnumMismatch,
                "Expected FEValuesBase type of " << arg1 << " but we got "
                                                 << arg2,
                FEValuesEnum, FEValuesEnum);


 protected:
  /**
   * Pointer to \c FEValuesBase.
   * The type of FEValues used will be specified by the constructor.
   */
  std::vector<std::shared_ptr<dealii::FEValuesBase<dim>>> ptrs_feval;

  CellIterator cell_iter;

  FaceIterator face_iter;

  unsigned int face_no;

  unsigned int subface_no;


 private:
  /**
   * Move ownership of the object resources into this object.
   */
  void move(CellScratchData<dim>&);


  /**
   * Duplicate each \c FEValuesBase entry.
   */
  std::shared_ptr<dealii::FEValuesBase<dim>> duplicate_fe_values(
    const std::shared_ptr<dealii::FEValuesBase<dim>>& pfeval);
};


/* ************************************************** */
/**
 *  Data that will be copied into the global vector
 */
/* ************************************************** */
template <int dim, typename NumberType>
struct CellCopyData
{
 public:
  using value_type = NumberType;
  using size_type = types::DoFIndex;
  using CellIterator = typename dealii::DoFHandler<dim>::active_cell_iterator;
  constexpr static int dimension = dim;


  /**
   * Default constructor. Construct with empty vectors.
   */
  CellCopyData() = default;

  /**
   * Virtual destructor
   */
  virtual ~CellCopyData() = default;


  /**
   * Allocate memory for all data members and then
   * \c reset the state of the object.
   */
  void allocate(size_type n);


  /**
   * Resetting local vector to zero, local dof indices to invalid
   * and active flag to \c false.
   */
  void reset(bool force = false);


  /**
   * Reinit with cell iterator.
   * Fill the dof_indices with that of the current cell,
   * Setting local vector to zero and set active flag to \c true.
   */
  void reinit(const CellIterator& cell_iter);


  /**
   * Provide interface for accessing and modifying the vector.
   */
  dealii::Vector<value_type>& vector();


  /**
   * Return \c active flag of the current object.
   */
  bool is_active() const;


  /**
   * When set to be active,
   * this object will participate in assembly if called.
   */
  void set_active();


  /**
   * Assemble the local vector into the global vector.
   */
  template <typename DstVectorType>
  void assemble(const dealii::AffineConstraints<NumberType>& constraints,
                DstVectorType& dst_vector) const;


 protected:
  /**
   * std::vector containing the DoF indices of the local vector.
   */
  std::vector<size_type> dof_indices;

  /**
   * Vector values at DoFs
   */
  dealii::Vector<value_type> local_vector;

  /**
   * If this flag is set to \c true, this object
   * will participate in the assembly to global vector.
   */
  bool active = false;
};


/* ************************************************** */
/**
 * Collection of copy data in cell and faces.
 */
/* ************************************************** */
template <typename CopyDataType>
struct CellCopyDataBox
{
  using size_type = types::DoFIndex;
  using value_type = typename CopyDataType::value_type;
  using CopyData = CopyDataType;
  using CellIterator = typename CopyDataType::CellIterator;
  constexpr static int dim = CopyDataType::dimension;
  constexpr static int faces_per_cell =
    dealii::GeometryInfo<dim>::faces_per_cell;

  /**
   * Constructor. Allocate space for all copy data.
   */
  CellCopyDataBox(size_type n);

  /**
   * Virtual destructor
   */
  virtual ~CellCopyDataBox() = default;

  /**
   * Reset all copy data to initial state.
   */
  void reset();

  /**
   * Assemble the copy data from local vector to global vector.
   */
  template <typename DstVectorType>
  void assemble(const dealii::AffineConstraints<value_type>& constraints,
                DstVectorType& rhs_vector) const;

  /** \name Local CopyData */
  //@{
  /**
   * Volume integration terms internal to the cell.
   */
  CopyData cell;

  /**
   * Face integration terms on the internal faces.
   */
  std::array<CopyData, faces_per_cell> interior_faces;

  /**
   * Face integration terms on the external faces.
   */
  std::array<CopyData, faces_per_cell> exterior_faces;
  //@}
};


FELSPA_NAMESPACE_CLOSE

/* ----------  Implementations  ------------ */
#include "src/cell_data.implement.h"
/* ----------------------------------------- */
#endif  // _FELSPA_FE_CELL_DATA_H_ //
