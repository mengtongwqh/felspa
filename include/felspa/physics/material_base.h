#ifndef _FELSPA_PHYSICS_MATERIAL_BASE_H_
#define _FELSPA_PHYSICS_MATERIAL_BASE_H_

#include <deal.II/base/quadrature.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/dofs/dof_handler.h>
#include <felspa/base/base_classes.h>
#include <felspa/base/exceptions.h>
#include <felspa/base/types.h>

FELSPA_NAMESPACE_OPEN

template <int dim, typename NumberType>
class MaterialAccessorBase;

template <typename MaterialType>
class MaterialAccessor;


/* ************************************************** */
/**
 * List of material parameters that have been implemented.
 */
/* ************************************************** */
enum class MaterialParameter
{
  density,
  viscosity,
};

std::ostream& operator<<(std::ostream& os, MaterialParameter parameter);


/* ************************************************** */
/**
 * Used for collecting kinematic information in the
 * current cell and pass them into material object
 * to evaluate material parameters.
 */
/* ************************************************** */
template <int dim, typename NumberType>
class PointsField
{
 public:
  using value_type = NumberType;
  using size_type = unsigned int;


  /**
   * Constructor
   */
  PointsField(size_type npts) : n_pts(npts)
  {
    ASSERT(n_pts > 0, ExcArgumentCheckFail());
  }


  /**
   * Quadrature points.
   */
  const std::vector<dealii::Point<dim, value_type>>* ptr_pts = nullptr;


  /**
   * Velocity at quadrature points.
   */
  const std::vector<dealii::Tensor<1, dim, value_type>>* ptr_velocities =
    nullptr;


  /**
   * No of points expected.
   */
  size_type size() const { return n_pts; }


 private:
  size_type n_pts;
};


/* ************************************************** */
/**
 * A base class for all materials.
 * All material definition inherits from this class
 * and provide the definitions for the virtual functions
 * in this class. In a multi-threaded application,
 * this class is responsible for taking in parameters passed
 * in from \c MaterialAccessor and evaluate the correct
 * material parameter while the state of the class
 * will remain const throughout the process.
 */
/* ************************************************** */
template <int dim, typename NumberType>
class MaterialBase : public TimedPhysicsObject<NumberType>
{
 public:
  constexpr static int dimension = dim;
  using value_type = NumberType;
  using typename TimedPhysicsObject<NumberType>::time_step_type;


  /**
   * Destructor.
   */
  virtual ~MaterialBase() = default;


  /**
   * Compute a scalar property of the material.
   * This applies to, e.g. density, viscosity, etc.
   */
  virtual void scalar_values(MaterialParameter,
                             const PointsField<dim, value_type>&,
                             std::vector<value_type>&) const
  {
    THROW(ExcUnimplementedVirtualFcn());
  }


  /**
   * Print the material information to \c std::ostream.
   */
  virtual void print(std::ostream& os) const = 0;


  /**
   * Return a \c const reference to the identifier string.
   */
  const std::string& get_label_string() const { return label_string; }


  /**
   * Generate a shared_ptr to an accessor.
   */
  virtual std::shared_ptr<MaterialAccessorBase<dim, value_type>>
  generate_accessor(const dealii::Quadrature<dim>& quadrature) const = 0;


 protected:
  /**
   * Forbid external construction.
   */
  MaterialBase(const std::string& id_string_) : label_string(id_string_)
  {
    ASSERT_NON_EMPTY(id_string_);
  }


  /**
   * Identifier string for this material.
   */
  std::string label_string;
};


/* ************************************************** */
/**
 * Accessing the material.
 * The existence of this class is to allow multi-threaded
 * access to the material. Since each accessor may access
 * different cells, putting them into the material class
 * will cause a race condition.
 */
/* ************************************************** */
template <int dim, typename NumberType>
class MaterialAccessorBase
{
 public:
  using value_type = NumberType;
  constexpr static int dimension = dim;


  /**
   * Virtual destructor.
   */
  virtual ~MaterialAccessorBase() = default;


  /**
   * After reiniting, accessor can store states
   * that are local to the current cell.
   */
  virtual void reinit(
    const typename dealii::DoFHandler<dim>::active_cell_iterator&)
  {}


  /**
   * Fill the vector with the computed material parameters.
   */
  virtual void eval_scalars(MaterialParameter mp,
                            const PointsField<dim, value_type>& pts_field,
                            std::vector<value_type>& val) = 0;

  /**
   * Return a const reference to the material that is pointed to.
   */
  const MaterialBase<dim, value_type>& get_material() const
  {
    ASSERT(ptr_material != nullptr, ExcNullPointer());
    return *ptr_material;
  }


 protected:
  /**
   * Constructor
   */
  MaterialAccessorBase(const MaterialBase<dim, value_type>& material,
                       const dealii::Quadrature<dim>& quad)
    : ptr_material(&material), quadrature(quad)
  {}


  /**
   * Copy constructor
   */
  MaterialAccessorBase(const MaterialAccessorBase<dim, value_type>&) = default;


  /**
   * Copy assignment
   */
  MaterialAccessorBase<dim, value_type>& operator=(
    const MaterialAccessorBase<dim, value_type>&) = default;


  /**
   * Pointer to the material that is accessed.
   */
  dealii::SmartPointer<const MaterialBase<dim, value_type>,
                       MaterialAccessorBase<dim, value_type>>
    ptr_material;


  /**
   * Type of quadrature formula needed at the cell.
   */
  dealii::Quadrature<dim> quadrature;
};


/* ************************************************** */
/**
 * Generic definition for MaterialAccessor.
 */
/* ************************************************** */
template <typename MaterialType>
class MaterialAccessor
  : public MaterialAccessorBase<MaterialType::dimension,
                                typename MaterialType::value_type>
{
 public:
  using base_type = MaterialAccessorBase<MaterialType::dimension,
                                         typename MaterialType::value_type>;
  using typename base_type::value_type;
  constexpr static int dimension = MaterialType::dimension, dim = dimension;


  /**
   * Constructor
   */
  MaterialAccessor(const MaterialType& material,
                   const dealii::Quadrature<dim>& quadrature)
    : base_type(material, quadrature)
  {}


  /**
   * Copy constructor
   */
  MaterialAccessor(const MaterialAccessor<MaterialType>& that) = default;


  /**
   * Copy assignment
   */
  MaterialAccessor<MaterialType>& operator=(
    const MaterialAccessor<MaterialType>& that) = default;


  /**
   * Get a scalar parameter.
   */
  void eval_scalars(MaterialParameter mp,
                    const PointsField<dim, value_type>& pfield,
                    std::vector<value_type>& vals) override
  {
    // for (auto& val : vals) val = this->ptr_material->eval_scalar(mp);
    this->ptr_material->scalar_values(mp, pfield, vals);
  }
};

FELSPA_NAMESPACE_CLOSE
#endif  // _FELSPA_PHYSICS_MATERIAL_BASE_H_ //
