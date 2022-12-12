#ifndef _FELSPA_PDE_LDG_H_
#define _FELSPA_PDE_LDG_H_

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_system.h>
#include <felspa/base/control_parameters.h>
#include <felspa/base/felspa_config.h>
#include <felspa/linear_algebra/system_assembler.h>
#include <felspa/pde/linear_systems.h>

#include <array>

// We are in the process of rewriting
// the assembler of the LDG linear systems
// from the Meshworker framework
// to a workstream framework.
#define USE_MESHWORKER_ASSEMBLER


FELSPA_NAMESPACE_OPEN

// ----------
namespace dg
// ----------
{
  // forward declarations
  template <int dim, typename NumberType>
  class LDGGradientLinearSystem;
  template <int dim, typename NumberType>
  struct LDGAssemblerControl;

  namespace internal
  {
    namespace meshworker
    {
      template <typename LinsysType>
      class LDGGradientAssembler;

      template <typename LinsysType>
      class LDGDiffusionAssembler;
    }  // namespace meshworker

    namespace workstream
    {
      template <typename LinsysType>
      class LDGGradientAssembler;

      template <int dim, typename NumberType>
      class LDGDiffusionAssembler;
    }  // namespace workstream
  }    // namespace internal

#ifdef USE_MESHWORKER_ASSEMBLER

  template <typename LinsysType>
  using LDGGradientAssembler =
    internal::meshworker::LDGGradientAssembler<LinsysType>;

  template <typename LinsysType>
  using LDGDiffusionAssembler =
    internal::meshworker::LDGDiffusionAssembler<LinsysType>;

#else  // USE_MESHWORKER_ASSEMBLER //

  template <int dim, typename NumberType>
  using LDGDiffusionTerm = internal::impl1::LDGDuffusionTerm;

  template <int dim, typename NumberType>
  using LDGGradientAssembler =
    internal::workstream::LDGGradientAssembler<dim, NumberType>;

  template <int dim, typename NumberType>
  using LDGDiffusionAssembler =
    internal::workstream::LDGDiffusionAssembler<dim, NumberType>;

#endif  // USE_MESHWORKER_ASSEMBLER //


  /* ************************************************** */
  /**
   * Two forms of flux choices
   * Use internal/external flux ("up" the normal or "down" the normal)
   * to compute gradient
   * Use external/internal flux to compute actual diffusion
   */
  /* ************************************************** */
  enum class LDGFluxEnum
  {
    left,
    right,
    internal,
    external,
    alternating,
  };


  /**
   * Overload operator for \enum LDGFluxEnum
   * \param[in, out] os
   * \param[in] flux_type
   */
  std::ostream& operator<<(std::ostream& os, const LDGFluxEnum& flux_type);


  /* ************************************************** */
  /**
   * Control Parameters for LDGDiffusionTerm
   */
  /* ************************************************** */
  template <int dim, typename NumberType>
  struct LDGControl
  {
    LDGControl()
      : ptr_solver(std::make_shared<dealii::SolverControl>()),
        ptr_assembler(std::make_shared<LDGAssemblerControl<dim, NumberType>>())
    {}

    /** Control parameters for gradient solver */
    std::shared_ptr<dealii::SolverControl> ptr_solver;

    /** Control parameters for assembler */
    std::shared_ptr<LDGAssemblerControl<dim, NumberType>> ptr_assembler;
  };


  /* ************************************************** */
  /**
   * Assemble LDG artificial diffusion coefficient.
   * This will be an addon to an existing solver.
   * Therefore this is not a primary simulator
   * This will not carry info about tempo-discretization.
   */
  /* ************************************************** */
  template <int dim, typename NumberType = types::DoubleType>
  class LDGDiffusionTerm
    : public FESimulator<dim, FEDGType<dim>, DGLinearSystem<dim, NumberType>>
  {
   public:
    using base_type =
      FESimulator<dim, FEDGType<dim>, DGLinearSystem<dim, NumberType>>;
    using fe_type = typename base_type::fe_type;
    using typename base_type::vector_type;
    using value_type = NumberType;
    using gradient_vector_type =
      typename LDGGradientLinearSystem<dim, value_type>::vector_type;

    using bcs_type = BCBookKeeper<dim, value_type>;

    /** Constructor */
    LDGDiffusionTerm(const base_type& simulator);

    LDGDiffusionTerm(const LDGDiffusionTerm<dim, NumberType>&) = delete;

    LDGDiffusionTerm<dim, NumberType>& operator=(
      const LDGDiffusionTerm<dim, NumberType>&) = delete;

    /** \name Initialization */
    //@{
    void attach_control(
      const std::shared_ptr<LDGControl<dim, value_type>>& ptr_control);

    void initialize(const TimedSolutionVector<vector_type>& soln);

    void initialize(vector_type&& initial_values);

    void initialize(const vector_type& initial_values);
    //@}

    /**
     * Compute diffusion term and put it in the solution vector.
     */
    void compute_diffusion_term(const bcs_type& bcs);

    void export_solution() const;

   protected:
    /// \name Simulator Auxillary functions
    //@{
    /**
     * Distribute dofs for grad_dof_handler and
     * allocate space for the gradient linear system.
     * dof_handler is not touched since it is managed
     * by the primary simulator. Called whenever \c initialized
     * and by \c upon_mesh_update().
     */
    void allocate_assemble_system();

    /** Assemble mass matrix for gradient linear system */
    void assemble_gradient_mass_matrix();

    /** Assemble RHS vector for gradient approximation */
    void assemble_gradient_rhs();

    /** Solve the diffusion linear system with specified rhs */
    void solve_linear_system(vector_type& soln, const vector_type& rhs);

    /** Solve the linear system for gradient */
    void solve_grad_linear_system(gradient_vector_type& soln,
                                  const gradient_vector_type& rhs);
    //@}

    /** Pointer to the parent simulator */
    dealii::SmartPointer<const base_type> ptr_parent_simulator;

    /** Finite element object */
    dealii::FESystem<dim> grad_fe;

    /** DoFHandler object for FE System */
    dealii::DoFHandler<dim> grad_dof_handler;

    /** Pointer to a linear system to solve  */
    std::shared_ptr<LDGGradientLinearSystem<dim, value_type>>
      ptr_grad_linear_system;

    /** Intermediate gradient solution */
    gradient_vector_type solution_gradient;

    /** Pointer to control parameter class */
    std::shared_ptr<LDGControl<dim, value_type>> ptr_control;
  };


  /* ************************************************** */
  /**
   * class LDGGradientLinearSystem
   */
  /* ************************************************** */
  template <int dim, typename NumberType = types::DoubleType>
  class LDGGradientLinearSystem : public BlockLinearSystem<dim, NumberType>
  {
   public:
    constexpr static int dimension = dim;
    using base_type = BlockLinearSystem<dim, NumberType>;
    using typename base_type::matrix_type;
    using typename base_type::size_type;
    using typename base_type::vector_type;

    explicit LDGGradientLinearSystem(const dealii::DoFHandler<dim>& dofh)
      : BlockLinearSystem<dim, NumberType>(dofh)
    {}

    LDGGradientLinearSystem(const dealii::DoFHandler<dim>& dofh,
                            const dealii::Mapping<dim>& mapping)
      : BlockLinearSystem<dim, NumberType>(dofh, mapping)
    {}

    void populate_system_from_dofs();

    /**
     * Solution strategy for the multi-component mass matrix.
     */
    void solve(vector_type&, const vector_type&,
               dealii::SolverControl&) const;

    // using base_type::solve;
  };


  /* ************************************************** */
  /**
   * Control structure for the LDGAssembler
   */
  /* ************************************************** */
  template <int dim, typename NumberType = types::DoubleType>
  struct LDGAssemblerControl : public ControlBase
  {
    using value_type = NumberType;

    LDGAssemblerControl() : ControlBase("LDG_Assembler")
    {
      for (int idim = 0; idim < dim; ++idim) beta[idim] = 1.0;
    }

    void parse_parameters(dealii::ParameterHandler&) override;

    void declare_parameters(dealii::ParameterHandler&) override;

    /**
     * A vector that is not parallel to any internal element face
     * for computing alternating LDG flux.
     */
    dealii::Tensor<1, dim, value_type> beta;  // = {1.0, 1.0, 1.0}

    /** viscosity coefficient */
    value_type viscosity = 0.0;

    /** Penalty coefficient for imposing Dirichlet BC */
    value_type penalty_coeff;

    /**
     * Spatial dimension we are computing in
     */
    int dim_idx;
  };


  // ---------------
  namespace internal
  // ---------------
  {
    // -----------------
    namespace meshworker
    // -----------------
    {
      /* ************************************************** */
      /**
       * Assemble RHS for LDG gradient computations
       */
      /* ************************************************** */
      template <typename LinsysType>
      class LDGGradientAssembler : public MeshWorkerAssemblerBase<LinsysType>
      {
       public:
        constexpr static int dim = LinsysType::dimension;
        constexpr static int dimension = LinsysType::dimension;

        using base_type = MeshWorkerAssemblerBase<LinsysType>;
        using typename base_type::dof_info_t;
        using typename base_type::integration_info_t;
        using typename base_type::linsys_type;
        using typename base_type::local_vector_t;
        using typename base_type::value_type;
        using typename base_type::vector_type;
        using bcs_type = BCBookKeeper<dim, value_type>;

        /** Constructor */
        LDGGradientAssembler(linsys_type& linsys,
                             bool construct_mapping_adhoc = true)
          : MeshWorkerAssemblerBase<LinsysType>(linsys,
                                                construct_mapping_adhoc),
            ptr_control(nullptr)
        {}

        /** Initialize the control parameters */
        void attach_control(
          const std::shared_ptr<LDGAssemblerControl<dim, value_type>>&
            pcontrol);

        /**
         * Execute the assembly.
         * Using the rhs of the attached linear system as the default vector.
         */
        template <LDGFluxEnum flux_type, typename VectorType>
        void assemble(const VectorType& phi,
                      const bcs_type& bcs,
                      bool zero_out_rhs = true);

       protected:
        /**
         * Local integrator for each different choice of diffusion flux type.
         * The \tparam Dummy parameter is used because the class cannot be
         * explicitly specialised in a template class.
         */
        template <LDGFluxEnum flux_type, typename Dummy = void>
        class Integrator;

        /** Pointer to control parameters */
        std::shared_ptr<const LDGAssemblerControl<dim, value_type>> ptr_control;

       private:
        /** Forward declaration of the local integrator */
        class IntegratorBase;
      };


      /* ************************************************** */
      /**
       * Assembler RHS for the diffusion term
       */
      /* ************************************************** */
      template <typename LinsysType>
      class LDGDiffusionAssembler : public MeshWorkerAssemblerBase<LinsysType>
      {
       public:
        constexpr static int dim = LinsysType::dimension;
        constexpr static int dimension = LinsysType::dimension;

        using base_type = MeshWorkerAssemblerBase<LinsysType>;
        using typename base_type::linsys_type;
        using typename base_type::value_type;
        using typename base_type::vector_type;
        using local_integrator_type =
          dealii::MeshWorker::LocalIntegrator<dim, dim, value_type>;
        using typename base_type::dof_info_t;
        using typename base_type::integration_info_t;
        using typename base_type::local_vector_t;
        using gradient_vector_type = dealii::BlockVector<value_type>;
        using bcs_type = BCBookKeeper<dim, value_type>;

        /**
         * Constructor
         */
        LDGDiffusionAssembler(linsys_type& linsys,
                              bool construct_mapping_adhoc = true)
          : base_type(linsys, construct_mapping_adhoc), ptr_control(nullptr)
        {}


        /**
         * Initialize the control parameters
         */
        void attach_control(
          const std::shared_ptr<const LDGAssemblerControl<dim, value_type>>&
            pcontrol);


        /**
         * Execute assembly with the choice of \c flux_type to \c rhs_vector.
         * \note \c rhs_vector is not zeroed prior to assembly.
         */
        template <LDGFluxEnum flux_type>
        void assemble(const gradient_vector_type& gradients,
                      const bcs_type& bcs,
                      vector_type& rhs_vector);


        /**
         * Execute assembly with the choice of \c flux_type.
         * Use the RHS of the attached linear system as the target vector.
         */
        template <LDGFluxEnum flux_type>
        void assemble(const gradient_vector_type& gradient,
                      const bcs_type& bcs,
                      bool zero_out_rhs = true);


       protected:
        /**
         * Declaration for the local integrator
         */
        template <LDGFluxEnum flux_type, typename Dummy = void>
        class Integrator;


        /**
         * Pointer to control paramters.
         */
        std::shared_ptr<const LDGAssemblerControl<dim, value_type>> ptr_control;


       private:
        /**
         * Base class to the local integrator
         */
        class IntegratorBase;
      };
    }  // namespace meshworker


    // ------------------
    namespace workstream
    // ------------------
    {
      /* ************************************************** */
      /**
       * Refrain from using the dealii::MeshWorker framework
       * We have to use \c LinsysType  as the template parameter
       * because
       */
      /* ************************************************** */
      template <typename LinsysType>
      class LDGGradientAssembler : public AssemblerBase<LinsysType>
      {
       public:
        constexpr static int dim = LinsysType::dimension;
        using base_type = AssemblerBase<LinsysType>;
        using linsys_type = LinsysType;
        using number_type = typename LinsysType::value_type;
        using value_type = number_type;
        using bcs_type = BCBookKeeper<dim, value_type>;
        using vector_type = typename linsys_type::vector_type;

        /**
         * Constructor.
         */
        LDGGradientAssembler(linsys_type& linsys,
                             bool construct_mapping_adhoc = true);

        /**
         * Attach the control parameters to the assembler
         */
        void attach_control();


        /**
         * Assemble gradient linear system.
         */
        template <LDGFluxEnum flux_type>
        void assemble(const vector_type& phi, const bcs_type& bcs,
                      bool zero_out_rhs = true);

       protected:
        /**
         * Pointer to the control parameters
         */
        std::shared_ptr<const LDGAssemblerControl<dim, value_type>> ptr_control;
      };


      /* ************************************************** */
      /**
       * Assemble diffusion to the rhs.
       * Here we also have to consider
       * the inter-elemental penalty term.
       */
      /* ************************************************** */
      template <int dim, typename NumberType>
      class LDGDiffusionAssembler
        : public AssemblerBase<DGLinearSystem<dim, NumberType>>
      {
       public:
        using linsys_type = DGLinearSystem<dim, NumberType>;
        using number_type = NumberType;
        using value_type = number_type;

        /**
         * Constructor
         */

        LDGDiffusionAssembler();
      };
    }  // namespace workstream
  }    // namespace internal

}  // namespace dg

FELSPA_NAMESPACE_CLOSE
/* -------- Template Implementations --------- */
#include "src/ldg.implement.h"
/* ------------------------------------------- */
#endif  // _FELSPA_PDE_LDG_H_
