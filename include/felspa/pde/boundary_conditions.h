#ifndef _FELSPA_PDE_BOUNDARY_CONDITIONS_H_
#define _FELSPA_PDE_BOUNDARY_CONDITIONS_H_

#include <deal.II/base/point.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/component_mask.h>
#include <deal.II/grid/grid_tools.h>
#include <felspa/base/exceptions.h>
#include <felspa/base/felspa_config.h>
#include <felspa/base/function.h>
#include <felspa/base/types.h>
#include <felspa/mesh/mesh.h>

#include <functional>
#include <initializer_list>
#include <map>
#include <vector>

FELSPA_NAMESPACE_OPEN

template <int dim, typename NumberType>
class BCBookKeeper;

template <int dim, typename NumberType>
class BCFunction;


/* ************************************************** */
/**
 * Enumerate types of boundary condition
 */
/* ************************************************** */
enum class BCCategory : short
{
  dirichlet = 1,      //!< Dirichlet-type (Type I)   boundary condition
  neumann = 2,        //!< Neumann-type   (Type II)  boundary condition
  robin = 3,          //!< Robin-type     (Type III) boundary condition
  periodic = 4,       //!< Periodic                  boundary condition
  no_normal_flux = 5  //!< No normal flux            boundary constraints
};

std::ostream& operator<<(std::ostream& os, BCCategory bc_category);


/* ************************************************** */
/**
 * Geometrical description of the boundary condition
 */
/* ************************************************** */
template <int dim, typename NumberType>
class BCGeometry
{
  friend class BCBookKeeper<dim, NumberType>;

 public:
  using value_type = NumberType;
  using ptr_bc_fcn_type = const BCFunction<dim, NumberType>*;
  using size_type = dealii::types::boundary_id;

  /**
   * @brief virtual destructor
   */
  virtual ~BCGeometry() = default;


  virtual bool at_boundary(const dealii::Point<dim, value_type>& pt) const = 0;

  /**
   * Get the boundary id
   */
  size_type get_boundary_id() const;


 private:
  /**
   * This value can only be set by \c BCBookKeeper
   */
  size_type boundary_id = dealii::numbers::invalid_boundary_id;
};


/* ************************************************** */
/**
 * Base types for boundary conditions
 */
/* ************************************************** */
template <int dim, typename NumberType>
class BCFunctionBase : public dealii::Function<dim, NumberType>
{
 public:
  constexpr static int spacedim = dim;
  using value_type = NumberType;
  using typename dealii::Function<dim, NumberType>::time_type;
  using size_type = dealii::types::boundary_id;

  /**
   * Constructor
   */
  BCFunctionBase(
    BCCategory bc_category, unsigned int n_components = 1,
    const dealii::ComponentMask& component_mask = dealii::ComponentMask(),
    time_type initial_time = 0.0);


  /**
   * Destructor
   */
  virtual ~BCFunctionBase() = default;


  /**
   * Set physical time.
   */
  virtual void set_time(time_type curr_time);


  /**
   * Test if the function is synchronized to \c current_time
   */
  virtual bool is_synchronized(time_type current_time) const;


  /**
   * Obtain boundary category
   */
  BCCategory get_category() const { return category; }


  /**
   * Get the component mask object
   */
  const dealii::ComponentMask& get_component_mask() const;


 protected:
  /**
   * Category of boundary condition.
   */
  BCCategory category;


  /**
   * Component mask
   */
  dealii::ComponentMask component_mask;
};


/* ************************************************** */
/**
 * Individual boundary condition entry.
 */
/* ************************************************** */
template <int dim, typename NumberType>
class BCFunction : public BCFunctionBase<dim, NumberType>
{
  friend class BCBookKeeper<dim, NumberType>;

 public:
  constexpr static int spacedim = dim;
  using value_type = NumberType;
  using typename BCFunctionBase<dim, NumberType>::time_type;
  using size_type = dealii::types::boundary_id;


  /**
   * Constructor
   */
  BCFunction(
    BCCategory bc_category,
    unsigned int n_components = 1,
    const dealii::ComponentMask& component_mask = dealii::ComponentMask(),
    time_type initial_time = 0.0);


  /**
   * Set the geometry object
   *
   */
  void set_geometry(
    const std::shared_ptr<BCGeometry<dim, value_type>>& ptr_geom);


  /**
   * Obtain boundary id.
   */
  size_type get_boundary_id() const;


 protected:
  /**
   * (Back)-Shared pointer to the geometry definition.
   */
  std::shared_ptr<BCGeometry<dim, value_type>> ptr_geometry;
};


/* ************************************************** */
/**
 * A pair of periodic boundary conditions.
 * Derived from BCFunction to
 * make it compatible with other BC definition.
 */
/* ************************************************** */
template <int dim, typename NumberType>
class PeriodicBCFunction : public BCFunctionBase<dim, NumberType>
{
  friend class BCBookKeeper<dim, NumberType>;

 public:
  using value_type = NumberType;
  using typename BCFunctionBase<dim, NumberType>::time_type;

  /**
   * Constructor.
   */
  PeriodicBCFunction(
    unsigned int n_component = 1,
    const dealii::ComponentMask& component_mask = dealii::ComponentMask(),
    time_type initial_time = 0.0);


  void set_geometry(
    const std::shared_ptr<BCGeometry<dim, value_type>>& ptr_geom1,
    const std::shared_ptr<BCGeometry<dim, value_type>>& ptr_geom2,
    const dealii::Tensor<1, dim, value_type>& offset,
    const dealii::FullMatrix<value_type>& rotation,
    int direction = -1);


  void set_geometry(
    const std::shared_ptr<BCGeometry<dim, value_type>>& ptr_geom1,
    const std::shared_ptr<BCGeometry<dim, value_type>>& ptr_geom2,
    int direction);


  template <typename MeshType>
  auto collect_periodic_faces(const MeshType& mesh) const;


  template <typename MeshType>
  void collect_face_pairs(
    const MeshType& mesh,
    std::vector<
      dealii::GridTools::PeriodicFacePair<typename MeshType::cell_iterator>>&
      face_pairs) const;


 protected:
  /**
   * A pair of periodic boundaries geometry
   */
  std::pair<std::shared_ptr<BCGeometry<dim, value_type>>,
            std::shared_ptr<BCGeometry<dim, value_type>>>
    ptr_geometry_pair;


  /**
   * \name Boundary geometry parameters.
   */
  //@{
  int direction;

  dealii::Tensor<1, dim, value_type> offset_vector;

  dealii::FullMatrix<value_type> rotation_matrix;
  //@}
};


/* ************************************************** */
/**
 * A class to bookkeep boundary condition,
 * also serve as a communicator between
 * a boundary condition function and the triangulation.
 */
/* ************************************************** */
template <int dim, typename NumberType>
class BCBookKeeper : public TimedPhysicsObject<NumberType>
{
 public:
  using value_type = NumberType;
  using time_type = typename BCFunctionBase<dim, value_type>::time_type;
  using size_type = typename BCFunctionBase<dim, value_type>::size_type;
  using boundary_id_type = dealii::types::boundary_id;

  /** \name Basic object behavior */
  //@{
  /**
   * Construct a BCBookKeeper, not initialized
   */
  BCBookKeeper() = default;

  /**
   * Construct and intialize the mesh
   */
  BCBookKeeper(Mesh<dim, value_type>& triag);

  /**
   * Copy constructor is not permitted.
   */
  BCBookKeeper(const BCBookKeeper<dim, value_type>& rhs) = delete;

  /**
   * Copy assignment is not permitted.
   */
  BCBookKeeper<dim, value_type>& operator=(
    const BCBookKeeper<dim, value_type>&) = delete;

  /**
   * Initialize the boundary collection with a given mesh.
   */
  void initialize(Mesh<dim, value_type>& mesh);

  /**
   * Inform the triangulation of a BC function.
   * Also attach the geometry definition to the boundary condition.
   */
  void append(const std::shared_ptr<BCFunction<dim, value_type>>& ptr_bc_fcn);

  /**
   * Inform the triangulation of a periodic BC.
   */
  void append(
    const std::shared_ptr<PeriodicBCFunction<dim, value_type>>& ptr_bc_fcn);

  /**
   * Clean up the vector holding BCs
   */
  void clear();

  /**
   * Return \c true if the BookKeeper does not contain any bc.
   */
  bool empty() const;

  /**
   * Test if the object is initialized.
   */
  bool is_initialized() const;

  /**
   * Print to screen the BCs added to this container.
   */
  void print(std::ostream& os) const;
  //@}


  /** \name Accessing boundary function entries */
  //@{
  /**
   * Return \c true if a specific BC category exists.
   */
  bool has_category(BCCategory category) const;

  /**
   * Return \c true if the boundary id is present
   */
  bool has_boundary_id(boundary_id_type i) const;

  /**
   * Get all BC functions of a specific category
   */
  std::vector<const BCFunctionBase<dim, value_type>*> operator()(
    BCCategory category) const;

  /**
   * Get boundary function at given boundary id
   */
  std::vector<const BCFunctionBase<dim, value_type>*> operator()(
    boundary_id_type bdry_id) const;
  //@}


  /** \name Temporal-related Functions */
  //@{
  void set_time(time_type current_time) override;

  void advance_time(time_type time_step) override;

  time_type get_time() const override;

  bool is_synchronized() const;

  bool is_synchronized(time_type current_time) const;
  //@}


  /**
   * Return the boundary id defined at certain point.
   */
  dealii::types::boundary_id boundary_id_at_point(
    const dealii::Point<dim>&) const;


  /**
   * Get a pointer to the triangulation.
   */
  const Mesh<dim, value_type>* get_mesh_ptr() const
  {
    ASSERT(ptr_mesh != nullptr, ExcNullPointer());
    return ptr_mesh;
  }


  /** \name Exceptions */
  //@{
  DECL_EXCEPT_0(ExcEmptyBoundaryCellIterators,
                "boundary_cell_iterators is empty. Call initialize()");

  DECL_EXCEPT_1(ExcDuplicateGeometry,
                "Point (" << arg1 << ") has multiple BCGeometry defined",
                dealii::Point<dim>);

  DECL_EXCEPT_2(ExcInsertFail,
                "Fail to insert boundary function of type "
                  << arg1 << " into BCBookKeeper with Boundary ID " << arg2
                  << ". This is probably an internal error.",
                BCCategory, boundary_id_type);

  DECL_EXCEPT_1(ExcBdryIdNotInBCBookKeeper,
                "Boundary Id "
                  << arg1 << " is not found in the current BCBookKeeper object",
                boundary_id_type);

  DECL_EXCEPT_1(ExcCategoryNotInBCBookKeeper,
                "BC Category "
                  << arg1 << " is not found in the current BCBookKeeper object",
                BCCategory);
  //@}


 protected:
  /**
   * Cache cell iterators that contain boundary
   */
  void cache_boundary_faces();


  /**
   * Get the next available boundary id
   */
  size_type next_available_boundary_id() const;


  /**
   * If a boundary id already exists on the given geometry,
   * then return this boundary.
   * If not, return a fresh boundary id.
   * If the boundary id switch midway when we
   * iterate through bc_geometry,
   * then an exception will be thrown.
   * @return the boundary id and if the Geometry entry should be inserted
   */
  std::pair<size_type, bool> try_insert_geometry(
    const BCGeometry<dim, value_type>& bc_geometry);


  bool check_bdry_id_geometry_consistency(
    size_type bdry_id, const BCGeometry<dim, value_type>& bc_geom) const;


  bool point_has_unique_bdry_id(const dealii::Point<dim, value_type>& pt) const;


  /**
   * Pointer to the attached triangulation object
   */
  dealii::SmartPointer<Mesh<dim, value_type>> ptr_mesh;


  /**
   * All boundary iterators
   */
  std::vector<typename dealii::Triangulation<dim>::face_iterator>
    boundary_iterators;


  /**
   * An std::set of pointers to boundary geometry.
   * This is to make sure that boundary geometry is unique.
   */
  std::set<const BCGeometry<dim, value_type>*> ptrs_geometry_set;


  /**
   * Using boundary id to get the BCGeometry object.
   */
  std::map<size_type, std::vector<const BCFunctionBase<dim, value_type>*>>
    bdry_id_to_pbcs;


  /**
   * Boundary functions indexed by its category.
   */
  std::map<BCCategory,
           std::vector<std::shared_ptr<BCFunctionBase<dim, value_type>>>>
    category_to_pbcs;
};


/* ---------------------------------------- */
/**
 * This namespace provides examples of BCs
 */
/* ---------------------------------------- */
namespace bc
{
  /* ************************************************** */
  /**
   * Class describing the boundary condition
   * for a lid-driven cavity in a Stokes solver.
   */
  /* ************************************************** */
  template <int dim, typename NumberType>
  class LidDrivenCavity : public BCFunction<dim, NumberType>
  {
   public:
    using value_type = NumberType;


    LidDrivenCavity(value_type velo_magnitude_ = 1.0,
                    value_type lower_bound = 0.0, value_type upper_bound = 1.0);


    value_type value(const dealii::Point<dim>& pt,
                     unsigned int component) const override;


    /**
     * Test if the point is at the top boundary.
     * Helper function for at_boundary().
     */
    bool at_top_boundary(const dealii::Point<dim, value_type>& pt) const;


   protected:
    class Geometry;

    /**
     * Magnitude of the velocity
     */
    value_type velo_magnitude;

    value_type lower_bound;

    value_type upper_bound;
  };


  /* ************************************************** */
  /**
   * Boundary condition for a linear simple shear.
   * Max velocity at the top/bottom.
   * Velocity diminishes towards the center line.
   */
  /* ************************************************** */
  template <int dim, typename NumberType>
  class LinearSimpleShear : public BCFunction<dim, NumberType>
  {
   public:
    using value_type = NumberType;
    using point_type = dealii::Point<dim, value_type>;

    /**
     * Constructor.
     */
    LinearSimpleShear(const point_type& lower_,
                      const point_type& upper_,
                      value_type velocity_magnitude);

    /**
     * Boundary condition value.
     */
    value_type value(const point_type& pt,
                     unsigned int component) const override;


   protected:
    class Geometry;

    /**
     * Lower-left-close point
     */
    const point_type lower;


    /**
     * Upper-right-far side point
     */
    const point_type upper;


    /**
     * Magnitude of the shear velocity.
     */
    const value_type velocity_magnitude;


    /**
     * Center level, computed in constructor.
     */
    const value_type center_level;


    /**
     * Half thickness of the box, computed in constructor.
     */
    const value_type half_thickness;
  };


  /* ************************************************** */
  /**
   * MidOceanRift boundary condition.
   * Velocity reverses when x = 0.
   */
  /* ************************************************** */
  template <int dim, typename NumberType>
  class MidOceanRift : public BCFunction<dim, NumberType>
  {
   public:
    using value_type = NumberType;
    using point_type = dealii::Point<dim, value_type>;

    /**
     * Constructor.
     */
    MidOceanRift(value_type velo_magnitude = 1.0, value_type upper_bound = 1.0);


    /**
     * Boundary condition definitiion.
     */
    value_type value(const dealii::Point<dim>& pt,
                     unsigned int component) const override;


   protected:
    class Geometry;


    /**
     * Magnitude of the velocity.
     */
    value_type velo_magnitude;


    /**
     * Upper bound of the bounding box.
     */
    value_type upper_bound;
  };

}  // namespace bc

FELSPA_NAMESPACE_CLOSE
/* -------- Explicit Instantiations ----------*/
#include "src/boundary_conditions.implement.h"
/* -------------------------------------------*/
#endif /* _FELSPA_PDE_BOUNDARY_CONDITIONS_H_ */
