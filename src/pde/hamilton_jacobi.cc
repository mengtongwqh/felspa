#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/numerics/vector_tools.h>
#include <felspa/pde/advection.h>
#include <felspa/pde/hamilton_jacobi.h>

#include <functional>

FELSPA_NAMESPACE_OPEN

/* -------------------------------------------*/
namespace dg
/* -------------------------------------------*/
{
  /* ************************************************** */
  /*                class HJOperator                    */
  /* ************************************************** */

  template <int dim, typename NumberType>
  void HJOperator<dim, NumberType>::extract_cell_gradients(
    const dealii::FEValuesBase<dim>& feval,
    std::vector<dealii::Tensor<1, dim, value_type>>& lgrad,
    std::vector<dealii::Tensor<1, dim, value_type>>& rgrad) const
  {
    const auto nqpt = feval.n_quadrature_points;

    ASSERT(lgrad.size() == nqpt, ExcSizeMismatch(lgrad.size(), nqpt));
    ASSERT(rgrad.size() == nqpt, ExcSizeMismatch(rgrad.size(), nqpt));

    const auto& lgrad_vector =
      ptr_hj_simulator->get_local_gradients(LDGFluxEnum::left);
    const auto& rgrad_vector =
      ptr_hj_simulator->get_local_gradients(LDGFluxEnum::right);

    ASSERT(lgrad_vector.n_blocks() == dim,
           ExcSizeMismatch(lgrad_vector.n_blocks(), dim));
    ASSERT(rgrad_vector.n_blocks() == dim,
           ExcSizeMismatch(rgrad_vector.n_blocks(), dim));


    for (int idim = 0; idim < dim; ++idim) {
      ASSERT(lgrad_vector.block(idim).size() ==
               ptr_hj_simulator->get_dof_handler().n_dofs(),
             ExcInternalErr());
      ASSERT(rgrad_vector.block(idim).size() ==
               ptr_hj_simulator->get_dof_handler().n_dofs(),
             ExcInternalErr());

      std::vector<value_type> lgrad_component(nqpt);
      std::vector<value_type> rgrad_component(nqpt);

      feval.get_function_values(lgrad_vector.block(idim), lgrad_component);
      feval.get_function_values(rgrad_vector.block(idim), rgrad_component);

      for (unsigned int iq = 0; iq < nqpt; ++iq) {
        lgrad[iq][idim] = lgrad_component[iq];
        rgrad[iq][idim] = rgrad_component[iq];
      }  // iq-loop
    }    // idim-loop
  }


  /* ************************************************** */
  /*                class HJSimulator                   */
  /* ************************************************** */
  template <int dim, typename NumberType>
  HJSimulator<dim, NumberType>::HJSimulator(Mesh<dim, value_type>& triag,
                                            unsigned int fe_degree,
                                            const std::string& label)
    : FESimulator<dim, FEDGType<dim>, DGLinearSystem<dim, NumberType>,
                  TempoIntegrator<NumberType>>(triag, fe_degree, label),
      ptr_control(std::make_shared<HJControl<dim, value_type>>()),
      ptr_hj(nullptr)
  {
    this->tempo_integrator.attach_control(ptr_control->ptr_tempo);
    this->set_quadrature(dealii::QGauss<dim>(fe_degree + 2));
    ptr_diffusion_generator =
      std::make_shared<LDGDiffusionTerm<dim, value_type>>(*this);
  }


  template <int dim, typename NumberType>
  HJSimulator<dim, NumberType>::HJSimulator(const base_type& fe_simulator)
    : base_type(fe_simulator), ptr_hj(nullptr), ptr_diffusion_generator(nullptr)
  {
    this->tempo_integrator.attach_control(ptr_control->ptr_tempo);
    ptr_diffusion_generator =
      std::make_shared<LDGDiffusionTerm<dim, value_type>>(*this);
  }


  template <int dim, typename NumberType>
  HJSimulator<dim, NumberType>::HJSimulator(
    const HJSimulator<dim, NumberType>& that)
    : FESimulator<dim, FEDGType<dim>, DGLinearSystem<dim, NumberType>,
                  TempoIntegrator<NumberType>>(that),
      ptr_control(
        std::make_shared<HJControl<dim, value_type>>(*(that.ptr_control))),
      left_local_gradients({}),
      right_local_gradients({}),
      ptr_hj(nullptr)
  {
    this->tempo_integrator.attach_control(ptr_control->ptr_tempo);
    ptr_diffusion_generator =
      std::make_shared<LDGDiffusionTerm<dim, value_type>>(*this);
  }


  template <int dim, typename NumberType>
  HJSimulator<dim, NumberType>& HJSimulator<dim, NumberType>::operator=(
    const HJSimulator<dim, NumberType>& that)
  {
    if (this == &that) return *this;
    this->FESimulator<dim, FEDGType<dim>, DGLinearSystem<dim, NumberType>,
                      TempoIntegrator<NumberType>>::operator=(that);

    this->ptr_control =
      std::make_shared<HJControl<dim, value_type>>(*(that.ptr_control));

    left_local_gradients.reinit(0);
    right_local_gradients.reinit(0);

    ptr_hj = nullptr;
    ptr_diffusion_generator =
      std::make_shared<LDGDiffusionTerm<dim, value_type>>(*this);

    this->tempo_integrator.attach_control(ptr_control->ptr_tempo);
    return *this;
  }


  template <int dim, typename NumberType>
  void HJSimulator<dim, NumberType>::attach_control(
    const std::shared_ptr<HJControl<dim, value_type>>& pcontrol)
  {
    ptr_control = pcontrol;
    this->tempo_integrator.attach_control(ptr_control->ptr_tempo);
  }


  template <int dim, typename NumberType>
  auto HJSimulator<dim, NumberType>::get_local_gradients(LDGFluxEnum flux) const
    -> const gradient_vector_type&
  {
    switch (flux) {
      case LDGFluxEnum::left:
        return left_local_gradients;
      case LDGFluxEnum::right:
        return right_local_gradients;
      default:
        THROW(
          EXCEPT_MSG("Please specify a flux direction (left or right) to "
                     "retrieve local gradients."));
    }
  }


  template <int dim, typename NumberType>
  bool HJSimulator<dim, NumberType>::is_initialized() const
  {
    return this->initialized && this->dof_handler().has_active_dofs() &&
           ptr_hj && (this->solution->size() == this->dof_handler().n_dofs()) &&
           (ptr_control->ptr_ldg->ptr_assembler->viscosity > 0.0
              ? static_cast<bool>(ptr_diffusion_generator)
              : true);
  }


  template <int dim, typename NumberType>
  void HJSimulator<dim, NumberType>::discretize_function_to_vector(
    const ScalarFunction<dim, value_type>& fcn, vector_type& vect) const
  {
    const auto& f = static_cast<const dealii::Function<dim, value_type>&>(fcn);

    ASSERT(vect.size() == this->get_dof_handler().n_dofs(),
           ExcSizeMismatch(vect.size(), this->get_dof_handler().n_dofs()));

    if (this->get_fe().has_generalized_support_points())
      dealii::VectorTools::interpolate(this->get_dof_handler(), f, vect);
    else
      dealii::VectorTools::project(this->get_dof_handler(), this->constraints(),
                                   this->get_quadrature(), f, vect);
  }


  template <int dim, typename NumberType>
  void HJSimulator<dim, NumberType>::discretize_function_to_solution(
    const ScalarFunction<dim, value_type>& fcn)
  {
    discretize_function_to_vector(fcn, *this->solution);
  }


  template <int dim, typename NumberType>
  void HJSimulator<dim, NumberType>::init_hj_operator(
    const std::shared_ptr<HJOperator<dim, value_type>>& p_hj)
  {
    // attach the HJOperator
    ASSERT(p_hj, ExcNullPointer());
    ptr_hj = p_hj;

    // cache initial l/r gradients
    local_left_right_gradients();
    ptr_hj->initialize(*this);
  }


  template <int dim, typename NumberType>
  void HJSimulator<dim, NumberType>::do_initialize(
    bool use_independent_solution)
  {
    this->temporal_passive_members.clear();

    // zero out time
    this->reset_time();

    this->initialized = true;
    upon_mesh_update();

    // allocate system
    this->dof_handler().distribute_dofs(this->fe());
    allocate_assemble_system();

    // allocate solution
    if (!this->primary_simulator && use_independent_solution)
      this->solution.reinit(std::make_shared<vector_type>(), 0.0);
    if (this->solution.is_independent())
      this->solution->reinit(this->dof_handler().n_dofs());

    // initialize diffusion term
    ptr_diffusion_generator->attach_control(this->ptr_control->ptr_ldg);
    ptr_diffusion_generator->initialize(this->solution);
    ASSERT(
      this->dof_handler().n_dofs() == this->solution->size(),
      ExcSizeMismatch(this->dof_handler().n_dofs(), this->solution->size()));

    this->ptr_cfl_estimator =
      std::make_unique<CFLEstimator<HJSimulator<dim, NumberType>>>(*this);
  }


  template <int dim, typename NumberType>
  void HJSimulator<dim, NumberType>::initialize(
    const ScalarFunction<dim, value_type>& initial_condition,
    const std::shared_ptr<HJOperator<dim, value_type>>& p_hj,
    bool execute_mesh_refine,
    bool use_independent_solution)
  {
    do_initialize(use_independent_solution);

    MeshControl<value_type>& mesh_control = *this->ptr_control->ptr_mesh;

    // interpolate initial condition function to solution vector
    discretize_function_to_solution(initial_condition);

    if (p_hj) init_hj_operator(p_hj);

    this->initialized = true;

    if (execute_mesh_refine && this->ptr_mesh_refiner) {
      felspa_log << "Recursively refine mesh to level "
                 << mesh_control.max_level
                 << " to better resolve initial condition..." << std::endl;

      for (auto ilevel = mesh_control.min_level;
           ilevel <= mesh_control.max_level;
           ++ilevel) {
        this->refine_mesh(*this->ptr_control->ptr_mesh);
        discretize_function_to_solution(initial_condition);
      }
    }
  }


  template <int dim, typename NumberType>
  void HJSimulator<dim, NumberType>::initialize(
    const TimedSolutionVector<vector_type, time_step_type>& other_solution,
    const std::shared_ptr<HJOperator<dim, value_type>>& p_hj)
  {
    do_initialize(false);
    this->solution = other_solution;
    if (p_hj) init_hj_operator(p_hj);
  }


  template <int dim, typename NumberType>
  void HJSimulator<dim, NumberType>::initialize(
    const vector_type& initial_condition,
    const std::shared_ptr<HJOperator<dim, value_type>>& p_hj,
    bool use_independent_solution)
  {
    do_initialize(use_independent_solution);

    // set initial condition
    ASSERT_SAME_SIZE(initial_condition, this->get_solution_vector());
    *(this->solution) = initial_condition;

    if (p_hj) init_hj_operator(p_hj);
  }


  template <int dim, typename NumberType>
  void HJSimulator<dim, NumberType>::allocate_assemble_system()
  {
    ASSERT(this->dof_handler().has_active_dofs(), ExcDoFHandlerNotInit());
    if (this->primary_simulator) {
      this->linear_system().populate_system_from_dofs();
      assemble_mass_matrix();
    }

    left_local_gradients.reinit(dim);
    right_local_gradients.reinit(dim);
    for (int idim = 0; idim < dim; ++idim) {
      left_local_gradients.block(idim).reinit(this->dof_handler().n_dofs());
      right_local_gradients.block(idim).reinit(this->dof_handler().n_dofs());
    }
    this->left_local_gradients.collect_sizes();
    this->right_local_gradients.collect_sizes();

    ASSERT(this->linear_system().size() == this->dof_handler().n_dofs(),
           ExcInternalErr());

    this->mesh_update_detected = false;
  }

  template <int dim, typename NumberType>
  void HJSimulator<dim, NumberType>::upon_mesh_update()
  {
    base_type::upon_mesh_update();

    if (this->initialized) {
      ASSERT(this->dof_handler().has_active_dofs(), ExcDoFHandlerNotInit());
      this->constraints().clear();
      this->constraints().close();
    }
  }


  template <int dim, typename NumberType>
  void HJSimulator<dim, NumberType>::assemble_mass_matrix()
  {
    ASSERT(this->linear_system().is_populated(),
           EXCEPT_MSG("Linear system is not populated/allocated prior "
                      "to assembly. Call populate_system_from_dofs."));
    MassMatrixAssembler<dim, value_type, LinearSystem> assembler(
      this->linear_system());
    assembler.assemble();
  }


  template <int dim, typename NumberType>
  auto HJSimulator<dim, NumberType>::advance_time(time_step_type time_step,
                                                  bool compute_single_cycle)
    -> time_step_type
  {
    const auto& tempo_control = *ptr_control->ptr_tempo;

    ASSERT(time_step > 0.0, ExcBackwardSync(time_step));
    ASSERT(is_initialized(), ExcSimulatorNotInitialized());
    ASSERT(tempo_control.query_method().second == TempoCategory::exp,
           EXCEPT_MSG(
             "Hamilton-Jacobi Simulator only allow explicit time stepping."));

    time_step_type (TempoIntegrator<time_step_type>::*integrator)(
      HJSimulator<dim, NumberType>&, time_step_type);
    time_step_type cumulative_time = 0.0;

    switch (tempo_control.query_method().first) {
      case TempoMethod::rktvd1:
        integrator =
          &TempoIntegrator<time_step_type>::template advance_time_step<
            RungeKuttaTVD<1>, TempoCategory::exp>;
        break;
      case TempoMethod::rktvd2:
        integrator =
          &TempoIntegrator<time_step_type>::template advance_time_step<
            RungeKuttaTVD<2>, TempoCategory::exp>;
        break;
      case TempoMethod::rktvd3:
        integrator =
          &TempoIntegrator<time_step_type>::template advance_time_step<
            RungeKuttaTVD<3>, TempoCategory::exp>;
        break;
      default:
        THROW(ExcNotImplemented());
    }

    if (tempo_control.defined_auto_adjust()) {
      value_type max_cfl = tempo_control.get_cfl().second;
      ASSERT(max_cfl > 0.0, EXCEPT_MSG("Max CFL must be positive."));

      do {
        value_type suggest_time_step =
          max_cfl / this->max_velocity_over_diameter(this->get_time());
        value_type time_substep =
          std::min(suggest_time_step, time_step - cumulative_time);
        std::min(suggest_time_step, time_step - cumulative_time);

        ASSERT(time_step - cumulative_time > 0, ExcInternalErr());

        (this->tempo_integrator.*integrator)(*this, time_substep);
        cumulative_time += time_substep;
        if (compute_single_cycle) break;

        // apply_moment_limiter(this->get_dof_handler(), this->get_mapping(),
        //                      this->get_quadrature(), *this->solution);

        // apply_weno_limiter(this->mesh(), this->get_dof_handler(),
        //                    this->get_mapping(), this->get_quadrature(),
        //                    *this->solution, this->fe_degree());

      } while (!numerics::is_zero(cumulative_time - time_step));
    }

    else {
      ASSERT(compute_single_cycle, ExcArgumentCheckFail());
      (this->tempo_integrator.*integrator)(*this, time_step);
      cumulative_time += time_step;

      // apply_moment_limiter(this->get_dof_handler(), this->get_mapping(),
      //                      this->get_quadrature(), *this->solution);

      // apply_weno_limiter(this->mesh(), this->get_dof_handler(),
      //                    this->get_mapping(), this->get_quadrature(),
      //                    *this->solution, this->fe_degree());
    }

    return cumulative_time;
  }


  template <int dim, typename NumberType>
  auto HJSimulator<dim, NumberType>::max_velocity_over_diameter(
    time_step_type current_time) const -> value_type
  {
    UNUSED_VARIABLE(current_time);

    value_type velo_diam = ptr_cfl_estimator->estimate(
      this->get_dof_handler().begin_active(), this->get_dof_handler().end());

    ASSERT(this->mesh().get_info().min_diameter > 0.0, ExcInternalErr());
    if (ptr_control->ptr_ldg->ptr_assembler->viscosity > 0.0)
      return cfl_scaling(this->fe_degree(),
                         this->ptr_control->ptr_ldg->ptr_assembler->viscosity,
                         velo_diam, this->mesh().get_info().min_diameter);
    else
      return cfl_scaling(this->fe_degree(), velo_diam);
  }


  template <int dim, typename NumberType>
  void HJSimulator<dim, NumberType>::solve_linear_system(vector_type& phidot,
                                                         const vector_type& rhs)
  {
    ASSERT_SAME_SIZE(phidot, this->linear_system());
    ASSERT_SAME_SIZE(rhs, this->linear_system());
    ASSERT(this->linear_system().is_populated(), ExcLinSysNotInit());
    dealii::TimerOutput::Scope t(this->simulation_timer, "system_solution");
    this->linear_system().solve(phidot, rhs, *(ptr_control->ptr_solver));
  }


  template <int dim, typename NumberType>
  void HJSimulator<dim, NumberType>::local_left_right_gradients()
  {
    dealii::TimerOutput::Scope t(this->simulation_timer,
                                 "local_left_right_gradients");

    // initialize system assembler
    LDGGradientAssembler<DGLinearSystem<dim, value_type>> system_assembler(
      this->linear_system());
    std::shared_ptr<LDGAssemblerControl<dim, value_type>>
      ptr_ldg_assembly_control =
        std::make_shared<LDGAssemblerControl<dim, value_type>>();
    system_assembler.attach_control(ptr_ldg_assembly_control);

    ASSERT(
      this->dof_handler().n_dofs() == this->solution->size(),
      ExcSizeMismatch(this->dof_handler().n_dofs(), this->solution->size()));

    // LEFT FLUX //
    for (int idim = 0; idim < dim; ++idim) {
      ptr_ldg_assembly_control->dim_idx = idim;
      system_assembler.template assemble<LDGFluxEnum::left>(*(this->solution),
                                                            this->bcs);
      solve_linear_system(left_local_gradients.block(idim),
                          this->linear_system().get_rhs());
    }  // idim-loop

    // RIGHT FLUX //
    for (int idim = 0; idim < dim; ++idim) {
      ptr_ldg_assembly_control->dim_idx = idim;
      system_assembler.template assemble<LDGFluxEnum::right>(*(this->solution),
                                                             this->bcs);
      solve_linear_system(right_local_gradients.block(idim),
                          this->linear_system().get_rhs());
    }  // idim-loop

#ifdef DEBUG
    // vector_type grad_diff(left_local_gradients.size());
    // for (types::SizeType i = 0; i < left_local_gradients.size(); ++i)
    // grad_diff[i] =
    // left_local_gradients[i].norm() - right_local_gradients[i].norm();
    // std::cout << "Difference in left and right gradients norm ["
    // << *std::max_element(grad_diff.begin(), grad_diff.end()) << ", "
    // << *std::min_element(grad_diff.begin(), grad_diff.end()) << ']'
    // << std::endl;
#endif

    // limit the curvature
    if (this->get_time() > 0.0) post_local_gradient_limiting();
  }


  template <int dim, typename NumberType>
  typename HJSimulator<dim, NumberType>::vector_type
  HJSimulator<dim, NumberType>::explicit_time_derivative(
    time_step_type current_time, const vector_type& soln_prev_step)
  {
    UNUSED_VARIABLE(soln_prev_step);

    ASSERT(is_initialized(), ExcSimulatorNotInitialized());

    this->set_time_temporal_passive_members(current_time);
    if (this->mesh_update_detected) this->allocate_assemble_system();

    // use local discontinuous Galerkin to solve for gradients
    local_left_right_gradients();

    // assemble the Hamilton-Jacobi RHS
    {
      dealii::TimerOutput::Scope t(this->simulation_timer, "assemble_rhs");
      ASSERT(ptr_control->ptr_ldg->ptr_assembler->viscosity >= 0.0,
             ExcArgumentCheckFail());

      // assemble the Hamilton-Jacobi operator
      HJAssembler<dim, value_type> assembler(this->linear_system());
      assembler.assemble(*ptr_hj, this->get_quadrature());

      if (!numerics::is_zero(ptr_control->ptr_ldg->ptr_assembler->viscosity)) {
        ASSERT(ptr_diffusion_generator != nullptr,
               EXCEPT_MSG("The diffusion generator has not been allocated."));
        ptr_diffusion_generator->compute_diffusion_term(this->bcs);
      }  // if (viscosity > 0.0)
    }

    // run the linear system solver
    vector_type phidot(this->dof_handler().n_dofs());
    solve_linear_system(phidot, this->linear_system().get_rhs());

    return phidot;
  }


  template <int dim, typename NumberType>
  auto HJSimulator<dim, NumberType>::cfl_scaling(const unsigned int degree,
                                                 const value_type max_velo_diam)
    -> time_step_type
  {
    ASSERT(max_velo_diam >= 0.0, ExcUnexpectedValue(max_velo_diam));

    value_type coeff = AdvectSimulator<dim>::cfl_scaling(degree);
    ASSERT(coeff > 0.0, ExcUnexpectedValue(coeff));

    return coeff * max_velo_diam;
  }


  template <int dim, typename NumberType>
  auto HJSimulator<dim, NumberType>::cfl_scaling(const unsigned int degree,
                                                 const value_type viscosity,
                                                 const value_type max_velo_diam,
                                                 const value_type min_diam)
    -> time_step_type
  {
    // value_type coeff =
    // AdvectSolver<dim, TempoIntegratorType>::cfl_scaling(degree);
    ASSERT(max_velo_diam > 0.0, ExcUnexpectedValue(max_velo_diam));
    return pow(degree + 1, 2) * max_velo_diam +
           viscosity * std::pow(degree + 1, 4) / std::pow(min_diam, 2);
  }


  /* ------------------ */
  namespace internal
  /* ------------------ */
  {
    /* ************************************************** */
    /*            class HJAssemblerMeshWorker             */
    /* ************************************************** */
    template <int dim, typename NumberType>
    HJAssemblerMeshWorker<dim, NumberType>::HJAssemblerMeshWorker(
      linsys_type& linsys, bool construct_mapping_adhoc)
      : MeshWorkerAssemblerBase<DGLinearSystem<dim, NumberType>>(
          linsys, construct_mapping_adhoc)
    {}


    template <int dim, typename NumberType>
    void HJAssemblerMeshWorker<dim, NumberType>::assemble(
      const HJOperator<dim, value_type>& hj,
      const dealii::Quadrature<dim>& quadrature,
      bool zero_out_rhs)
    {
      UNUSED_VARIABLE(quadrature);
      using namespace dealii::MeshWorker;

      // construct the local integrator and info box
      const unsigned int n_gauss_pts =
        this->get_dof_handler().get_fe().degree + 1;

      HJLocalIntegrator local_integrator(hj);
      IntegrationInfoBox<dim> info_box;
      info_box.initialize_gauss_quadrature(n_gauss_pts, n_gauss_pts,
                                           n_gauss_pts);
      info_box.initialize_update_flags();
      info_box.add_update_flags_all(dealii::update_values);
      info_box.initialize(this->get_dof_handler().get_fe(),
                          this->get_mapping());

      // initialize IntegrationInfo data structure
      DoFInfo<dim> dof_info(this->dof_handler());

      // assembler for the RHS vector
      Assembler::ResidualSimple<vector_type> assembler;
      dealii::AnyData data;
      data.add<vector_type*>(&this->rhs(), "hj_rhs");
      assembler.initialize(data);

      // perform actual assembly
      if (zero_out_rhs) this->ptr_linear_system->zero_out(false, true);
      integration_loop<dim, dim>(this->dof_handler().begin_active(),
                                 this->dof_handler().end(), dof_info, info_box,
                                 local_integrator, assembler);
    }


    /**
     * This class is internal to \c HJAssemblerMeshWorker and is used to assist
     * the \c MeshWorker \c loop assembly process
     */
    template <int dim, typename NumberType>
    class HJAssemblerMeshWorker<dim, NumberType>::HJLocalIntegrator
      : public dealii::MeshWorker::LocalIntegrator<dim, dim, value_type>
    {
      using parent_class_t = HJAssemblerMeshWorker<dim, NumberType>;
      using dof_info_t = typename parent_class_t::dof_info_t;
      using vector_type = typename parent_class_t::vector_type;
      using local_vector_t = typename parent_class_t::local_vector_t;
      using integration_info_t = typename parent_class_t::integration_info_t;
      using value_type = typename vector_type::value_type;


     public:
      /**
       * Constructor.
       */
      HJLocalIntegrator(const HJOperator<dim, value_type>& hj)
        : dealii::MeshWorker::LocalIntegrator<dim, dim, value_type>(true, false,
                                                                    false),
          ptr_hj(&hj)
      {}

      /**
       * Cell Integrator
       */
      virtual void cell(dof_info_t& dinfo,
                        integration_info_t& cinfo) const override;


     private:
      /**
       * Pointer to Hamilton-Jacobi operator
       */
      const HJOperator<dim, value_type>* ptr_hj;
    };


    template <int dim, typename NumberType>
    void HJAssemblerMeshWorker<dim, NumberType>::HJLocalIntegrator::cell(
      dof_info_t& dinfo, integration_info_t& cinfo) const
    {
      using namespace dealii;

      const FEValuesBase<dim>& fe = cinfo.fe_values();
      const std::vector<double>& JxW = fe.get_JxW_values();
      local_vector_t& local_rhs = dinfo.vector(0).block(0);

      // values of the Hamilton-Jacobi vector at quadrature points
      std::vector<value_type> hj_at_qpt(fe.n_quadrature_points);
      ptr_hj->cell_values(fe, hj_at_qpt);

      for (unsigned int iq = 0; iq < fe.n_quadrature_points; ++iq)
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          local_rhs[i] -= fe.shape_value(i, iq) * hj_at_qpt[iq] *
                          static_cast<value_type>(JxW[iq]);
    }


    /* ************************************************** */
    /*            class HJAssemblerWorkStream             */
    /* ************************************************** */
    template <int dim, typename NumberType>
    HJAssemblerWorkStream<dim, NumberType>::HJAssemblerWorkStream(
      linsys_type& linear_system, bool construct_mapping_adhoc)
      : base_type(linear_system, construct_mapping_adhoc)
    {}


    template <int dim, typename NumberType>
    void HJAssemblerWorkStream<dim, NumberType>::assemble(
      const HJOperator<dim, value_type>& hj,
      const dealii::Quadrature<dim>& quadrature, bool zero_out_rhs)
    {
      using namespace dealii;
      using this_type = HJAssemblerWorkStream<dim, value_type>;
      ScratchData scratch(
        this->get_mapping(), this->get_dof_handler().get_fe(), quadrature,
        update_values | update_gradients | update_JxW_values, hj);
      CopyData copy(this->get_dof_handler().get_fe().dofs_per_cell);

      if (zero_out_rhs) this->ptr_linear_system->zero_out(false, true);

      dealii::WorkStream::run(this->dof_handler().begin_active(),
                              this->dof_handler().end(), *this,
                              &this_type::local_assembly,
                              &this_type::copy_local_to_global, scratch, copy);
    }


    template <int dim, typename NumberType>
    void HJAssemblerWorkStream<dim, NumberType>::local_assembly(
      const active_cell_iterator_type& cell, ScratchData& s, CopyData& c)
    {
      s.reinit(cell);
      cell->get_dof_indices(c.local_dof_indices);

      const auto nqpt = s.feval.get_quadrature().size();
      const auto ndof = s.feval.dofs_per_cell;

      c.local_rhs = 0.0;
      for (unsigned int iq = 0; iq < nqpt; ++iq)
        for (unsigned int idof = 0; idof < ndof; ++idof)
          c.local_rhs[idof] -=
            s.feval.shape_value(idof, iq) * s.hj_values[iq] * s.feval.JxW(iq);
    }


    template <int dim, typename NumberType>
    void HJAssemblerWorkStream<dim, NumberType>::copy_local_to_global(
      const CopyData& c)
    {
      this->constraints().distribute_local_to_global(
        c.local_rhs, c.local_dof_indices, this->rhs());
    }


    /* ---------------------------------------------------- *
     * HJAssemblerWorkStream<dim, NumberType>>::ScratchData
     * ---------------------------------------------------- */
    template <int dim, typename NumberType>
    class HJAssemblerWorkStream<dim, NumberType>::ScratchData
    {
     public:
      using active_cell_iterator_type =
        typename dealii::DoFHandler<dim>::active_cell_iterator;


      /**
       * @brief Constructor.
       */
      ScratchData(const dealii::Mapping<dim>& mapping,
                  const dealii::FiniteElement<dim>& finite_element,
                  const dealii::Quadrature<dim>& quadrature,
                  dealii::UpdateFlags update_flags,
                  const HJOperator<dim, value_type>& hj)
        : feval(mapping, finite_element, quadrature, update_flags),
          hj_values(quadrature.size()),
          ptr_hj(&hj)
      {}


      /**
       * Copy constructor
       */
      ScratchData(const ScratchData& that)
        : feval(that.feval.get_mapping(), that.feval.get_fe(),
                that.feval.get_quadrature(), that.feval.get_update_flags()),
          hj_values(that.hj_values),
          ptr_hj(that.ptr_hj)
      {}


      /**
       * @brief Reinit the object to a new cell.
       * This will reinit fevals and compute hj_values.
       */
      void reinit(const active_cell_iterator_type& cell)
      {
        feval.reinit(cell);
        ptr_hj->cell_values(feval, hj_values);
      }


      dealii::FEValues<dim> feval;

      std::vector<value_type> hj_values;

      const HJOperator<dim, value_type>* ptr_hj;
    };


    /* ------------------------------------------------- *
     * HJAssemblerWorkStream<dim, NumberType>>::CopyData
     * ------------------------------------------------- */
    template <int dim, typename NumberType>
    class HJAssemblerWorkStream<dim, NumberType>::CopyData
    {
     public:
      using size_type = types::DoFIndex;


      /**
       * @brief Constructor
       */
      CopyData(size_t n) : local_dof_indices(n), local_rhs(n) {}


      CopyData(const CopyData& that) = default;

      /**
       * DoF indices
       */
      std::vector<size_type> local_dof_indices;

      /**
       * local RHS vector that will be assembled into global matrix
       */
      dealii::Vector<value_type> local_rhs;
    };
  }  // namespace internal
}  // namespace dg

/* -------- Explicit Instantiations ----------*/
#include "hamilton_jacobi.inst"
/* -------------------------------------------*/

FELSPA_NAMESPACE_CLOSE
