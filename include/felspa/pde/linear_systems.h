#ifndef _FELSPA_PDE_LINEAR_SYSTEMS_H_
#define _FELSPA_PDE_LINEAR_SYSTEMS_H_

#include <deal.II/dofs/dof_faces.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/simple.h>
#include <felspa/base/felspa_config.h>
#include <felspa/base/numerics.h>
#include <felspa/linear_algebra/system_assembler.h>
#include <felspa/pde/boundary_conditions.h>
#include <felspa/pde/pde_base.h>

FELSPA_NAMESPACE_OPEN

namespace dg
{
  /* ************************************************** */
  /**
   * Specialization of \c LinearSystem for DG problems.
   */
  /* ************************************************** */
  template <int dim, typename NumberType = types::DoubleType>
  class DGLinearSystem : public LinearSystem<dim, NumberType>
  {
   public:
    constexpr const static int spacedim = dim;
    constexpr const static int dimension = dim;

    using typename LinearSystem<dim, NumberType>::matrix_type;
    using typename LinearSystem<dim, NumberType>::vector_type;
    using typename LinearSystem<dim, NumberType>::value_type;

    /**
     * Constructor
     */
    DGLinearSystem(const dealii::DoFHandler<dim>&);


    /**
     * Constructor
     */
    DGLinearSystem(const dealii::DoFHandler<dim>&, const dealii::Mapping<dim>&);


    /**
     * When populating, generate
     * sparsity pattern for DG discretization
     */
    void populate_system_from_dofs();


    /**
     * Solve the linear system resulting from DG assembly
     */
    void solve(vector_type&, const vector_type& rhs_vector,
               dealii::SolverControl&) const;
  };


  template <int dim, typename NumberType = types::DoubleType>
  class DGBlockLinearSystem : public BlockLinearSystem<dim, NumberType>
  {
   public:
    constexpr const static int spacedim = dim;
    constexpr const static int dimension = dim;
  };
}  // namespace dg


/* ************************************************** */
/**
 * @brief Linear System for a velocity-presssure mixed FEM problem
 * The matrix is of the form
 * \f[
 * \begin{bmatrix}
 * A & B \\
 * B^\top &  O
 * \end{bmatrix}
 * \f]
 * @tparam dim
 * @tparam NumberType
 * @tparam InnerPreconditioner Preconditioner for inner solve
 * @tparam OuterPreconditioner
 */
/* ************************************************** */
template <int dim, typename NumberType>
class MixedVPLinearSystem : public BlockLinearSystem<dim, NumberType>
{
 public:
  using value_type = NumberType;
  using base_type = BlockLinearSystem<dim, value_type>;
  using typename base_type::matrix_block_type;
  using typename base_type::matrix_type;
  using typename base_type::sparsity_type;
  using typename base_type::vector_block_type;
  using typename base_type::vector_type;

  constexpr const static int spacedim = dim;
  constexpr const static int dimension = dim;

  /**
   * @brief Constructor
   */
  MixedVPLinearSystem(const dealii::DoFHandler<dim>& dofh);


  /**
   * @brief Constructor
   */
  MixedVPLinearSystem(const dealii::DoFHandler<dim>& dofh,
                      const dealii::Mapping<dim>& mapping);

  /**
   * @brief Allocate linear system
   */
  void populate_system_from_dofs();


  /**
   * @brief Setup hanging node and boundary constraints
   */
  void setup_constraints(const BCBookKeeper<dim, value_type>& bcs);


  /**
   * @brief Set up the constraints and system object.
   * This combines \c setup_constraints() and
   * \c populate_system_from_dofs() and utilizes multitasking.
   */
  void setup_constraints_and_system(const BCBookKeeper<dim, value_type>& bc);


  /**
   * @brief Inform the linear system that the constraints need to be updated.
   * Called upon every mesh update.
   */
  void flag_constraints_for_update() { constraints_updated = false; }


  /**
   * @brief Return the status of for \constraints_updated flag
   */
  bool constraints_are_updated() const { return constraints_updated; }


 protected:
  virtual void setup_bc_constraints(
    const BCBookKeeper<dim, value_type>& bcs) = 0;


  /**
   * @brief Construct the sparsity for linear system LHS matrix
   */
  void make_matrix_sparsity();


  /**
   * @brief Flag for constraint update
   */
  bool constraints_updated = false;
};


FELSPA_NAMESPACE_CLOSE


/*------- IMPLEMENTATIONS -------*/
#include "src/linear_systems.implement.h"
/*-------------------------------*/
#endif  // _FELSPA_PDE_LINEAR_SYSTEMS_H_ //
