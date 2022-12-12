#ifndef _FELSPA_LEVEL_SET_MATERIAL_STACK_H_
#define _FELSPA_LEVEL_SET_MATERIAL_STACK_H_

#include <felspa/level_set/level_set.h>
#include <felspa/physics/material_base.h>

#include <deque>

FELSPA_NAMESPACE_OPEN

/* -------------------- */
namespace ls
/* -------------------- */
{
  /* ************************************************** */
  /**
   * Stack of materials that will be composited by
   * multi-material simulators.
   */
  /* ************************************************** */
  template <int dim, typename NumberType>
  class MaterialStack : public MaterialBase<dim, NumberType>
  {
    /**
     * Allow \c MaterialCompositor to access the this class
     * to calculate material parameter compositing.
     */
    friend class MaterialAccessor<MaterialStack<dim, NumberType>>;

   public:
    using base_type = MaterialBase<dim, NumberType>;
    using size_type = types::MaterialIndex;
    using value_type = NumberType;
    using typename base_type::time_step_type;


    /**
     * Constructor.
     */
    MaterialStack(const std::string& id_string = "MaterialStack");


    /**
     * Constructor
     */
    MaterialStack(const std::shared_ptr<MaterialBase<dim, value_type>>&
                    p_background_material,
                  const std::string& id_string = "MaterialStack");


    /**
     * Deleted copy constructor
     */
    MaterialStack(const MaterialStack<dim, value_type>&);


    /**
     * Set the background material.
     */
    void set_background_material(
      const std::shared_ptr<MaterialBase<dim, value_type>>& p_material);


    /**
     * Push the material entry to the front / top of the stack.
     */
    template <typename LevelSetSimulator>
    void push(const std::shared_ptr<MaterialBase<dim, value_type>>& p_material,
              const std::shared_ptr<LevelSetSimulator>& p_level_set);


    /**
     * Clear up all records in this container.
     */
    void clear();


    /**
     * Advance time unconditionally.
     */
    void advance_time(time_step_type time_step) override;


    /**
     * Advance time adaptively.
     */
    time_step_type advance_time(time_step_type time_step,
                                bool compute_single_step);


    /**
     * Test if the background material has been set.
     */
    bool has_background_material() const;


    /**
     * Test if the material collection is synchronized.
     */
    bool is_synchronized() const;


    /**
     * Number of materials contained in the stack.
     * If we have n materials and 1 background material,
     * then we have n+1 materials.
     */
    size_type n_materials() const;


    /**
     * Return the corresponding accessor.
     */
    std::shared_ptr<MaterialAccessorBase<dim, value_type>> generate_accessor(
      const dealii::Quadrature<dim>& quadrature) const override;


    /**
     * Print the information of the container to the std::ostream.
     */
    void print(std::ostream& os) const override;


    /**
     * Obtain a material/level-set pair.
     * Remember the index starts from 1 to remind you that there is
     * hypothetical a background material located at index 0.
     * The materials are arranged from bottom to top.
     */
    std::pair<MaterialBase<dim, NumberType>&, SimulatorBase<dim, NumberType>&>
    operator[](size_type idx);


    /**
     * Same as above with \c const override.
     */
    std::pair<const MaterialBase<dim, NumberType>&,
              const SimulatorBase<dim, NumberType>&>
    operator[](size_type idx) const;


    /**
     * Exception to be thrown if background material is not defined
     */
    DECL_EXCEPT_0(ExcBackgroundMaterialUndefined,
                  "Background material in the MaterialStack must be defined "
                  "prior to this operation. This can be done by calling "
                  "set_background_material() on the MaterialStack.");


   private:
    /**
     * Struct making up each entry of this container
     */
    struct MaterialLevelSet;


    /**
     * The actual stack which is implemented in the form of \c std::deque
     */
    std::deque<MaterialLevelSet> material_lvsets;


    /**
     * Background material.
     */
    std::shared_ptr<MaterialBase<dim, value_type>> ptr_background_material =
      nullptr;
  };

}  // namespace ls


/* ************************************************** */
/**
 * This is a class when initialized with \c MaterialStack,
 * retrieve material parameters from it with the level set
 * values reinited for this cell.
 * MaterialStack will not be altered in this process.
 */
/* ************************************************** */
template <int dim, typename NumberType>
class MaterialAccessor<ls::MaterialStack<dim, NumberType>>
  : public MaterialAccessorBase<dim, NumberType>
{
 public:
  using value_type = NumberType;
  using material_type = ls::MaterialStack<dim, value_type>;
  using size_type = typename material_type::size_type;
  using ExcBackgroundMaterialUndefined =
    typename material_type::ExcBackgroundMaterialUndefined;

  /**
   *  Constructor
   */
  MaterialAccessor(const material_type& mat_stack,
                   const dealii::Quadrature<dim>& quadrature);


  /**
   * Copy constructor.
   * Must be implemented as this class participates in a thread-local object.
   */
  MaterialAccessor(const MaterialAccessor<material_type>& that);


  /**
   * Deleted copy assignment.
   * Copy assignment only works if both operands point to the same material
   * stack as well as the underlying triangulation.
   */
  MaterialAccessor<material_type>& operator=(
    const MaterialAccessor<material_type>&);


  /**
   * Reinitialize all fevalues
   */
  void reinit(const typename dealii::DoFHandler<dim>::active_cell_iterator&
                cell) override;


  /**
   * To evaluate scalars material parameters.
   */
  void eval_scalars(MaterialParameter parameters_type,
                    const PointsField<dim, value_type>& pts_field,
                    std::vector<value_type>& value_qpt) override;


  /**
   * To check if the object and the underlying \c MaterialStack
   * is consistent.
   */
  bool consistent_with_material_stack() const;


  /**
   * Failure that will be thrown if the \c consistent_with_material_stack()
   * check is failed.
   */
  DECL_EXCEPT_0(ExcMaterialStackInconsistent,
                "The Material-Level set pairs defined in the MaterialStack is "
                "not consistent with FEValues defined in the class. The "
                "MaterialStack must have been altered after the construction "
                "of this class, which is not permissible.");


 private:
  /**
   * Helper function to construct FEValues from MaterialStack.
   */
  void make_fe_values();


  /**
   * Push this material entry into material_*_cache container
   * so that their material properties will be evaluated
   * when calling eval_* functions.
   */
  void cache_material(const MaterialBase<dim, value_type>* p_material);


  /**
   * Clear up \c material_proportion vector to a pristine state and
   * also clear up material property cache.
   */
  void reset();


  /**
   * Downcasting the \c ptr_material to a pointer to \c MaterialStack.
   */
  const ls::MaterialStack<dim, value_type>* ptr_material_stack() const;


  /**
   * Pointer to the mesh used by all level set simulators.
   * If only background material is defined in the material stack,
   * then this pointer is null.
   */
  const dealii::SmartPointer<const Mesh<dim, NumberType>,
                             MaterialAccessor<material_type>>
    ptr_mesh = nullptr;


  /**
   * List of \c FEValues object.
   * Each one corresponds to one entry of level set simulator.
   */
  std::deque<dealii::FEValues<dim>> fevals;


  /**
   * For each quadrature point, pair the type of material(s) with
   * the proportion of this type of material.
   * The std::vector is of length [nqpt].
   */
  std::vector<std::map<const MaterialBase<dim, value_type>*, value_type>>
    material_proportion;


  /**
   * Cache materials who are present in this cell.
   * Their scalar material properties at quadrature points will be precomputed
   * in \c eval_scalars.
   */
  std::map<const MaterialBase<dim, value_type>*, std::vector<value_type>>
    material_scalars_cache;
};


FELSPA_NAMESPACE_CLOSE

/* --------- IMPLEMENTATION --------- */
#include "src/material_stack.implement.h"
/* ---------------------------------- */
#endif  // _FELSPA_LEVEL_SET_MATERIAL_STACK_H_ //
