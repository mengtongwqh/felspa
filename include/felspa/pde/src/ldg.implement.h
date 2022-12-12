#ifndef _FELSPA_PDE_LDG_IMPLEMENT_H_
#define _FELSPA_PDE_LDG_IMPLEMENT_H_

#include <deal.II/meshworker/loop.h>
#include <felspa/pde/ldg.h>

FELSPA_NAMESPACE_OPEN

/* ------------------------------------------- */
namespace dg
/* ------------------------------------------- */
{
  // ----------------
  namespace internal
  // ----------------
  {
    // ------------------
    namespace meshworker
    // ------------------
    {
      /* ************************************************** */
      /**
       * Integrator Base class for \class LDGGradientAssembler
       */
      /* ************************************************** */
      template <typename LinsysType>
      class LDGGradientAssembler<LinsysType>::IntegratorBase
        : public dealii::MeshWorker::LocalIntegrator<dim, dim, value_type>
      {
       public:
        constexpr static int dim = LinsysType::dimension;
        constexpr static int dimension = LinsysType::dimension;

        using parent_class_t = LDGGradientAssembler<LinsysType>;
        using dof_info_t = typename parent_class_t::dof_info_t;
        using integration_info_t = typename parent_class_t::integration_info_t;
        using vector_type = typename parent_class_t::vector_type;
        using value_type = typename vector_type::value_type;
        using bcs_type = typename LDGGradientAssembler<LinsysType>::bcs_type;

        /**
         * Constructor. Initialize reference to \class FEFunctionSelector
         * and boundary conditions.
         */
        IntegratorBase(
          const FEFunctionSelector<dim, dim, value_type>& fe_fcn_selector_,
          const bcs_type& bcs_,
          const LDGAssemblerControl<dim, value_type>& control_)
          : dealii::MeshWorker::LocalIntegrator<dim, dim, value_type>(
              true, true, true),
            fe_fcn_selector(fe_fcn_selector_),
            bcs(bcs_),
            control(control_)
        {}

        /** Cell assembler */
        virtual void cell(dof_info_t& dinfo,
                          integration_info_t& cinfo) const override;

       protected:
        /**
         * Pointer to \class FEFunctionSelector in the parent class
         */
        const FEFunctionSelector<dim, dim, value_type>& fe_fcn_selector;

        /**
         * Collection of boundary conditions
         */
        const bcs_type& bcs;

        /**
         * Control Parameters
         */
        const LDGAssemblerControl<dim, value_type>& control;
      };


      /* ************************************************** */
      /**
       * Integrator class for \class LDGGradientAssembler.
       * This is a useless generic definition.
       */
      /* ************************************************** */
      template <typename LinsysType>
      template <LDGFluxEnum flux_type, typename Dummy>
      class LDGGradientAssembler<LinsysType>::Integrator
      {};


      /* ************************************* */
      /** Specialization for alternating flux  */
      /* ************************************* */
      template <typename LinsysType>
      template <typename Dummy>
      class LDGGradientAssembler<LinsysType>::Integrator<
        LDGFluxEnum::alternating, Dummy>
        : public LDGGradientAssembler<LinsysType>::IntegratorBase
      {
       public:
        using base_type =
          typename LDGGradientAssembler<LinsysType>::IntegratorBase;
        using typename base_type::dof_info_t;
        using typename base_type::integration_info_t;
        using typename base_type::value_type;
        using typename base_type::bcs_type;

        constexpr static int dim = base_type::dim;

        Integrator(
          const FEFunctionSelector<dim, dim, value_type>& fe_fcn_selector,
          const bcs_type& bcs,
          const LDGAssemblerControl<dim, value_type>& control)
          : base_type(fe_fcn_selector, bcs, control)
        {}

        virtual void cell(dof_info_t& dinfo,
                          integration_info_t& cinfo) const override;

        virtual void face(dof_info_t& dinfo_in,
                          dof_info_t& dinfo_ex,
                          integration_info_t& cinfo_in,
                          integration_info_t& cinfo_ex) const override;

        virtual void boundary(dof_info_t& dinfo,
                              integration_info_t& cinfo) const override;
      };


      /* ******************************* */
      /** Specialization for right flux  */
      /* ******************************* */
      template <typename LinsysType>
      template <typename Dummy>
      class LDGGradientAssembler<LinsysType>::Integrator<LDGFluxEnum::right,
                                                         Dummy>
        : public LDGGradientAssembler<LinsysType>::IntegratorBase
      {
       public:
        using base_type =
          typename LDGGradientAssembler<LinsysType>::IntegratorBase;
        using typename base_type::dof_info_t;
        using typename base_type::integration_info_t;
        using typename base_type::value_type;
        using typename base_type::bcs_type;

        constexpr static int dim = base_type::dimension;

        Integrator(
          const FEFunctionSelector<dim, dim, value_type>& fe_fcn_selector,
          const bcs_type& bcs,
          const LDGAssemblerControl<dim, value_type>& control)
          : base_type(fe_fcn_selector, bcs, control)
        {}

        virtual void face(dof_info_t& dinfo_in,
                          dof_info_t& dinfo_ex,
                          integration_info_t& cinfo_in,
                          integration_info_t& cinfo_ex) const override;

        virtual void boundary(dof_info_t& dinfo,
                              integration_info_t& cinfo) const override;
      };


      /* ****************************** */
      /** Specialization for left flux  */
      /* ****************************** */
      template <typename LinsysType>
      template <typename Dummy>
      class LDGGradientAssembler<LinsysType>::Integrator<LDGFluxEnum::left,
                                                         Dummy>
        : public LDGGradientAssembler<LinsysType>::IntegratorBase
      {
       public:
        using base_type =
          typename LDGGradientAssembler<LinsysType>::IntegratorBase;
        using typename base_type::dof_info_t;
        using typename base_type::integration_info_t;
        using typename base_type::value_type;
        using typename base_type::bcs_type;

        constexpr static int dim = base_type::dimension;
        
        Integrator(
          const FEFunctionSelector<dim, dim, value_type>& fe_fcn_selector,
          const bcs_type& bcs,
          const LDGAssemblerControl<dim, value_type>& control)
          : base_type(fe_fcn_selector, bcs, control)
        {}

        virtual void face(dof_info_t& dinfo_in,
                          dof_info_t& dinfo_ex,
                          integration_info_t& cinfo_in,
                          integration_info_t& cinfo_ex) const override;

        virtual void boundary(dof_info_t& dinfo,
                              integration_info_t& cinfo) const override;
      };


      /* ************************************************** */
      /**
       * Local integrator for \class LDGDifffusionIntegrator.
       * Assemble RHS for explicit diffusion computations.
       */
      /* ************************************************** */
      template <typename LinsysType>
      class LDGDiffusionAssembler<LinsysType>::IntegratorBase
        : public dealii::MeshWorker::LocalIntegrator<dim, dim, value_type>
      {
       public:
        constexpr static int dim = LinsysType::dimension;
        constexpr static int dimension = LinsysType::dimension;

        using parent_class_t = LDGGradientAssembler<LinsysType>;
        using dof_info_t = typename parent_class_t::dof_info_t;
        using integration_info_t = typename parent_class_t::integration_info_t;
        using vector_type = typename parent_class_t::vector_type;
        using local_vector_t = typename parent_class_t::local_vector_t;
        using value_type = typename vector_type::value_type;
        using bcs_type = typename LDGDiffusionAssembler<LinsysType>::bcs_type;

        IntegratorBase(
          const FEFunctionSelector<dim, dim, value_type>& fe_fcn_selector_,
          const bcs_type& bcs_,
          const LDGAssemblerControl<dim, value_type>& control_)
          : dealii::MeshWorker::LocalIntegrator<dim, dim, value_type>(
              true, true, true),
            fe_fcn_selector(fe_fcn_selector_),
            bcs(bcs_),
            control(control_)
        {
          ASSERT(control.viscosity > 0.0, ExcArgumentCheckFail());
        }

        virtual void cell(dof_info_t&, integration_info_t&) const override;

       protected:
        const FEFunctionSelector<dim, dim, value_type>& fe_fcn_selector;

        const bcs_type& bcs;

        const LDGAssemblerControl<dim, value_type>& control;
      };


      /* ************************************************** */
      /**
       * Integrator class for \class DiffusionAssembler.
       * This is a useless generic definition.
       */
      /* ************************************************** */
      template <typename LinsysType>
      template <LDGFluxEnum flux_type, typename Dummy>
      class LDGDiffusionAssembler<LinsysType>::Integrator
      {};


      /* ********************************** */
      /** Specialization for alternating flux  */
      /* ********************************** */
      template <typename LinsysType>
      template <typename Dummy>
      class LDGDiffusionAssembler<LinsysType>::Integrator<
        LDGFluxEnum::alternating, Dummy>
        : public LDGDiffusionAssembler<LinsysType>::IntegratorBase
      {
       public:
        using base_type =
          typename LDGDiffusionAssembler<LinsysType>::IntegratorBase;
        using typename base_type::dof_info_t;
        using typename base_type::integration_info_t;
        using typename base_type::value_type;
        using typename base_type::bcs_type;

        constexpr static int dim = base_type::dimension;

        Integrator(
          const FEFunctionSelector<dim, dim, value_type>& fe_fcn_selector,
          const bcs_type& bcs,
          const LDGAssemblerControl<dim, value_type>& control)
          : base_type(fe_fcn_selector, bcs, control)
        {}

        void face(dof_info_t& dinfo_in,
                  dof_info_t& dinfo_ex,
                  integration_info_t& cinfo_ex,
                  integration_info_t& cinfo_in) const override;

        void boundary(dof_info_t& dinfo_in,
                      integration_info_t& cinfo) const override;
      };


      /* ************************************************** */
      /**
       * \class LDGGradientAssembler function implementations
       */
      /* ************************************************** */
      /**
       * Attach control parameters by
       * initializing the ptr_control member.
       */
      template <typename LinsysType>
      void LDGGradientAssembler<LinsysType>::attach_control(
        const std::shared_ptr<LDGAssemblerControl<dim, value_type>>& pcontrol)
      {
        const auto beta_norm = pcontrol->beta.norm();
        if (beta_norm > 0.0)
          for (int idim = 0; idim < dim; ++idim)
            pcontrol->beta[idim] /= beta_norm;
        ptr_control = pcontrol;
      }

      /**
       * Assemble function for
       * \class LDGGradientAssembler.
       * The function will execute
       * assembly by calling the appropriate
       * LocalIntegrator type object.
       */
      template <typename LinsysType>
      template <LDGFluxEnum flux_type, typename VectorType>
      void LDGGradientAssembler<LinsysType>::assemble(const VectorType& phi,
                                                      const bcs_type& bcs,
                                                      bool zero_out_rhs)
      {
        ASSERT(
          ptr_control != nullptr,
          EXCEPT_MSG(
            "Control parameters are not initialized. Call attach_control() "
            "prior to assembly!"));

        using namespace dealii::MeshWorker;
        LOG_PREFIX("LDGGradientAssembler");
        felspa_log << "Assembling with " << flux_type << " flux..."
                   << std::endl;

        IntegrationInfoBox<dim> info_box;

        // push the previous solution into fe_fcn_selector
        this->fe_fcn_selector.reset();
        this->fe_fcn_selector.attach_info_box(info_box);

        this->fe_fcn_selector.add("soln",
                                  phi,
                                  {true, false, false},
                                  {true, false, false},
                                  {true, false, false});

        info_box.add_update_flags_cell(dealii::update_gradients);
        this->fe_fcn_selector.finalize(
          this->dof_handler().get_fe(), this->get_mapping(),
          &this->dof_handler().block_info(), vector_type());

        // construct local integrator
        Integrator<flux_type> integrator(this->fe_fcn_selector, bcs,
                                         *ptr_control);

        // construct assembler
        dealii::AnyData data;
        Assembler::ResidualSimple<vector_type> assembler;
        data.add<vector_type*>(&this->rhs(), "rhs");
        assembler.initialize(data);

        // execute assemly
        if (zero_out_rhs) this->ptr_linear_system->zero_out(false, true);

        if (std::is_same<dealii::Vector<value_type>, vector_type>::value) {
          // if single component, use dof_handler
          DoFInfo<dim> dof_info(this->dof_handler());
          integration_loop<dim, dim>(this->dof_handler().begin_active(),
                                     this->dof_handler().end(), dof_info,
                                     info_box, integrator, assembler);
        } else if (std::is_same<dealii::BlockVector<value_type>,
                                vector_type>::value) {
          // if dealing with multicomponent block vector, use block_info
          DoFInfo<dim> dof_info(this->dof_handler().block_info());
          integration_loop<dim, dim>(this->dof_handler().begin_active(),
                                     this->dof_handler().end(), dof_info,
                                     info_box, integrator, assembler);
        } else
          ASSERT(false, ExcNotImplemented());
      }


      /* ************************************************** */
      /**
       * Assemble function for \class LDGDiffusionAssembler.
       * The function will execute assembly by calling the appropriate
       * LocalIntegrator type object.
       */
      /* ************************************************** */
      /**
       * Initialize the control parameters
       */
      template <typename LinsysType>
      FELSPA_FORCE_INLINE void LDGDiffusionAssembler<LinsysType>::attach_control(
        const std::shared_ptr<const LDGAssemblerControl<dim, value_type>>&
          pcontrol)
      {
        ASSERT(pcontrol->viscosity > 0.0, ExcArgumentCheckFail());
        ptr_control = pcontrol;
      }


      /**
       * Assemble function.
       * The function will execute assembly
       * by calling the appropriate
       * LocalIntegrator type object.
       */
      template <typename LinsysType>
      template <LDGFluxEnum flux_type>
      void LDGDiffusionAssembler<LinsysType>::assemble(
        const gradient_vector_type& gradients,
        const bcs_type& bcs,
        vector_type& rhs_vector)
      {
        ASSERT(
          ptr_control != nullptr,
          EXCEPT_MSG(
            "Control parameters are not initialized. Call attach_control() "
            "prior to assembly!"));

        using namespace dealii::MeshWorker;
        LOG_PREFIX("LDGDiffusionAssembler");
        felspa_log << "Assembling with " << flux_type << " flux..."
                   << std::endl;

        IntegrationInfoBox<dim> info_box;
        DoFInfo<dim> dof_info(this->dof_handler());

        this->fe_fcn_selector.reset();
        this->fe_fcn_selector.attach_info_box(info_box);

        for (int idim = 0; idim < dim; ++idim) {
          std::string label = "grad" + util::int_to_string(idim, 2);
          this->fe_fcn_selector.add(label, gradients.block(idim),
                                    {true, false, false}, {true, false, false},
                                    {true, false, false});
        }

        // use the inherent numbering for dealii::AnyData
        info_box.add_update_flags_cell(dealii::update_gradients);
        this->fe_fcn_selector.finalize(this->dof_handler().get_fe(),
                                       this->get_mapping());

        // construct local integrator
        Integrator<flux_type> integrator(this->fe_fcn_selector, bcs,
                                         *ptr_control);

        // construct assembler
        Assembler::ResidualSimple<vector_type> assembler;
        dealii::AnyData data;
        data.add<vector_type*>(&rhs_vector, "rhs");
        assembler.initialize(data);

        // execute assembly
        integration_loop<dim, dim>(this->dof_handler().begin_active(),
                                   this->dof_handler().end(), dof_info,
                                   info_box, integrator, assembler);
      }


      template <typename LinsysType>
      template <LDGFluxEnum flux_type>
      void LDGDiffusionAssembler<LinsysType>::assemble(
        const gradient_vector_type& gradients, const bcs_type& bcs,
        bool zero_out_rhs)
      {
        if (zero_out_rhs) this->ptr_linear_system->zero_out(false, true);
        assemble<flux_type>(gradients, bcs, this->rhs());
      }
    }  // namespace meshworker


    // ------------------
    namespace workstream
    // ------------------
    {
      template <typename LinsysType>
      LDGGradientAssembler<LinsysType>::LDGGradientAssembler(
        linsys_type& linsys, bool construct_mapping_adhoc)
        : base_type(linsys, construct_mapping_adhoc), ptr_control(nullptr)
      {}


      // LDGGradientAssemble

    }  // namespace workstream
  }    // namespace internal
}  // namespace dg

FELSPA_NAMESPACE_CLOSE
#endif
