#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/meshworker/local_integrator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <felspa/pde/ldg.h>

#include <type_traits>
#include <utility>

#include "felspa/base/felspa_config.h"

FELSPA_NAMESPACE_OPEN

/* -------------------------------------------*/
namespace dg
/* -------------------------------------------*/
{
  /* ************************************************** */
  /** Overload operator for \enum LDGFluxEnum */
  /* ************************************************** */

  std::ostream& operator<<(std::ostream& os, const LDGFluxEnum& flux_type)
  {
    switch (flux_type) {
      case LDGFluxEnum::left:
        os << "left";
        break;
      case LDGFluxEnum::right:
        os << "right";
        break;
      case LDGFluxEnum::internal:
        os << "internal";
        break;
      case LDGFluxEnum::external:
        os << "external";
        break;
      case LDGFluxEnum::alternating:
        os << "alternating";
        break;
      default:
        ASSERT(false, ExcInternalErr());
    }
    return os;
  }


  /* ************************************************** */
  /** \class LDGAssemblerControl */
  /* ************************************************** */

  template <int dim, typename NumberType>
  void LDGAssemblerControl<dim, NumberType>::declare_parameters(
    dealii::ParameterHandler& prm)
  {
    using namespace dealii;
    prm.enter_subsection(subsection_id_string);
    {
      prm.declare_entry(
        "penalty coefficient", "0.0", Patterns::Double(),
        "Penalty coefficient for imposing Dirichlet Boundary Conditions");
      prm.declare_entry("viscosity", "0.0", Patterns::Double(),
                        "Global viscosity coefficient for diffusion modelling");
    }
    prm.leave_subsection();
  }


  template <int dim, typename NumberType>
  void LDGAssemblerControl<dim, NumberType>::parse_parameters(
    dealii::ParameterHandler& prm)
  {
    prm.enter_subsection(subsection_id_string);
    {
      penalty_coeff = prm.get_double("penalty coefficient");
      viscosity = prm.get_double("viscosity");
    }
    prm.leave_subsection();
  }


  /* ************************************************** */
  /** \class LDGDiffusionTerm */
  /* ************************************************** */

  template <int dim, typename NumberType>
  LDGDiffusionTerm<dim, NumberType>::LDGDiffusionTerm(
    const base_type& simulator)
    : base_type(simulator),
      ptr_parent_simulator(&simulator),
      grad_fe(simulator.get_fe(), dim),
      grad_dof_handler(simulator.mesh()),
      ptr_grad_linear_system(
        std::make_shared<LDGGradientLinearSystem<dim, value_type>>(
          grad_dof_handler)),
      ptr_control(nullptr)

  {
    // this->solution.reinit(std::make_shared<vector_type>());
    // this->solution = simulator.get_wrapped_solution();
  }


  template <int dim, typename NumberType>
  void LDGDiffusionTerm<dim, NumberType>::attach_control(
    const std::shared_ptr<LDGControl<dim, value_type>>& pcontrol)
  {
    ASSERT(pcontrol, ExcNullPointer());
    ptr_control = pcontrol;
  }


  template <int dim, typename NumberType>
  FELSPA_FORCE_INLINE void LDGDiffusionTerm<dim, NumberType>::initialize(
    const TimedSolutionVector<vector_type>& soln)
  {
    this->solution = soln;
    allocate_assemble_system();
    this->initialized = true;
  }

  template <int dim, typename NumberType>
  FELSPA_FORCE_INLINE void LDGDiffusionTerm<dim, NumberType>::initialize(
    const vector_type& initial_values)
  {
    *(this->solution) = initial_values;
    allocate_assemble_system();
    this->initialized = true;
  }


  template <int dim, typename NumberType>
  FELSPA_FORCE_INLINE void LDGDiffusionTerm<dim, NumberType>::initialize(
    vector_type&& initial_values)
  {
    this->solution->swap(initial_values);
    allocate_assemble_system();
    this->initialized = true;
  }


  template <int dim, typename NumberType>
  void LDGDiffusionTerm<dim, NumberType>::assemble_gradient_mass_matrix()
  {
    MassMatrixAssembler<dim, NumberType, BlockLinearSystem> mass_assembler(
      *ptr_grad_linear_system);
    // ptr_grad_linear_system->zero_out(true, false);
    mass_assembler.assemble();

#ifdef DEBUG  // make sure all diagonals are nonzero
    const auto& linsys = ptr_grad_linear_system->get_matrix();
    for (size_t i = 0; i < linsys.n(); ++i)
      if (numerics::is_zero(linsys.diag_element(i))) {
        std::cerr << "ERROR: Diagonal element on row " << i << " / "
                  << linsys.n() << " is " << linsys.diag_element(i) << '!'
                  << std::endl;
        THROW(EXCEPT_MSG("Zero on diagonal detected"));
      }
#endif
  }


  template <int dim, typename NumberType>
  void LDGDiffusionTerm<dim, NumberType>::allocate_assemble_system()
  {
    LOG_PREFIX("LDGDiffusionTerm");

    felspa_log << "Allocating gradient system..." << std::endl;
    // distribute dofs
    this->grad_dof_handler.distribute_dofs(this->grad_fe);
    // count dofs per block/component
    this->ptr_grad_linear_system->count_dofs();
    // renumber here because linear_system is not allowed to modify dof_handler
    dealii::DoFRenumbering::component_wise(this->grad_dof_handler);
    // now allocate the system
    this->ptr_grad_linear_system->populate_system_from_dofs();

    felspa_log << "Assembling gradient mass matrix..." << std::endl;
    assemble_gradient_mass_matrix();

    // We are reusing the solution vector from the parent simulator
    // so it can never be independent.
    ASSERT(!this->solution.is_independent(), ExcInternalErr());
    // felspa_log << "Allocating solution vector..." << std::endl;
    // this->solution->reinit(this->dof_handler().n_dofs());

    // allocate gradient solution
    solution_gradient.reinit(dim);
    const auto& ndofs_component = ptr_grad_linear_system->get_component_ndofs();
    for (int idim = 0; idim < dim; ++idim)
      solution_gradient.block(idim).reinit(ndofs_component[idim]);
    solution_gradient.collect_sizes();

    this->mesh_update_detected = false;
  }


  template <int dim, typename NumberType>
  void LDGDiffusionTerm<dim, NumberType>::solve_linear_system(
    vector_type& soln, const vector_type& rhs)
  {
    ASSERT_SAME_SIZE(soln, this->linear_system());
    ASSERT_SAME_SIZE(rhs, this->linear_system());
    dealii::TimerOutput::Scope t(this->simulation_timer,
                                 "ldg_solve_diffusion_system");
    this->linear_system().solve(soln, rhs, *(this->ptr_control->ptr_solver));
  }


  template <int dim, typename NumberType>
  void LDGDiffusionTerm<dim, NumberType>::solve_grad_linear_system(
    gradient_vector_type& soln, const gradient_vector_type& rhs)
  {
    ASSERT_SAME_SIZE(soln, this->linear_system());
    ASSERT_SAME_SIZE(rhs, this->linear_system());
    dealii::TimerOutput::Scope t(this->simulation_timer,
                                 "ldg_solve_gradient_system");
    this->ptr_grad_linear_system->solve(soln, rhs,
                                        *this->ptr_control->ptr_solver);
  }


  template <int dim, typename NumberType>
  void LDGDiffusionTerm<dim, NumberType>::export_solution() const
  {
    using namespace dealii;

    // ASSERT(!ptr_parent_simulator->ptr_control->soln_label.empty(),
    // ExcInternalErr());
    const std::string solution_label = "diffusion_gradient.vtu";

    // auto& os = file.access_stream();

    // write vector output
    std::vector<std::string> solution_names(dim, "gradient");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim,
                     DataComponentInterpretation::component_is_part_of_vector);
    DataOut<dim> data_out;
    data_out.add_data_vector(grad_dof_handler, solution_gradient,
                             solution_names, interpretation);
    data_out.build_patches(this->fe_degree() + 1);

    std::ofstream output(solution_label);
    data_out.write_vtu(output);
    output.close();
  }


  template <int dim, typename NumberType>
  void LDGDiffusionTerm<dim, NumberType>::compute_diffusion_term(
    const bcs_type& bcs)
  {
    // assemble the mass matrix if there is an update in mesh configuration
    if (this->mesh_update_detected) allocate_assemble_system();

    ASSERT(this->dof_handler().n_dofs() ==
             ptr_parent_simulator->get_solution()->size(),
           ExcInternalErr());
    ASSERT(this->grad_dof_handler.n_dofs() == solution_gradient.size(),
           ExcInternalErr());
    ASSERT(this->ptr_control, ExcNullPointer());
    ASSERT(this->ptr_grad_linear_system->is_populated(), ExcLinSysNotInit());


    {  // assemble the gradient system
      grad_dof_handler.initialize_local_block_info();

      // copy parent simulator solution
      // *(this->solution) = ptr_parent_simulator->get_solution_vector();
      dealii::BlockVector<value_type> solution_repeated(
        this->grad_dof_handler.block_info().global());
      solution_repeated.block(0) = *(this->solution);
      solution_repeated.collect_sizes();

      for (int idim = 0; idim < dim; ++idim)
        ptr_control->ptr_assembler->beta[idim] = 1.0;

      // assemble the gradient system
      LDGGradientAssembler<LDGGradientLinearSystem<dim, value_type>>
        grad_assembler(*ptr_grad_linear_system);
      grad_assembler.attach_control(ptr_control->ptr_assembler);
      grad_assembler.template assemble<LDGFluxEnum::alternating>(
        solution_repeated, bcs, true);

      // solve the system and put gradients in gradient_solution
      ptr_grad_linear_system->solve(solution_gradient,
                                    ptr_grad_linear_system->get_rhs(),
                                    *(ptr_control->ptr_solver));
    }

    {  // assemble system to solve for diffusion
      LDGDiffusionAssembler<DGLinearSystem<dim, value_type>>
        diffusion_assembler(this->linear_system());
      diffusion_assembler.attach_control(ptr_control->ptr_assembler);
      diffusion_assembler.template assemble<LDGFluxEnum::alternating>(
        solution_gradient, bcs, false);  // directly add to the RHS
    }
  }


  /* ************************************************** */
  /** \class LDGGradientLinearSystem */
  /* ************************************************** */

  template <int dim, typename NumberType>
  void LDGGradientLinearSystem<dim, NumberType>::populate_system_from_dofs()
  {
    ASSERT(this->get_dof_handler().has_active_dofs(), ExcDoFHandlerNotInit());

    // allocate space for the matrix
    this->ndofs_per_component = std::move(
      dealii::DoFTools::count_dofs_per_fe_component(this->get_dof_handler()));
    dealii::BlockDynamicSparsityPattern dsp(dim, dim);

    for (int idim = 0; idim < dim; ++idim)
      for (int jdim = 0; jdim < dim; ++jdim)
        dsp.block(idim, jdim)
          .reinit(this->ndofs_per_component[idim],
                  this->ndofs_per_component[jdim]);
    dsp.collect_sizes();
    dealii::DoFTools::make_flux_sparsity_pattern(this->get_dof_handler(), dsp);
    this->sparsity_pattern.copy_from(dsp);
    this->matrix.reinit(this->sparsity_pattern);

    this->rhs.reinit(dim);
    for (int idim = 0; idim < dim; ++idim)
      this->rhs.block(idim).reinit(this->ndofs_per_component[idim]);
    this->rhs.collect_sizes();

    this->populated = true;
  }


  template <int dim, typename NumberType>
  void LDGGradientLinearSystem<dim, NumberType>::solve(
    vector_type& gradient_solution,
    const vector_type& rhs_vector,
    dealii::SolverControl& solver_control) const
  {
    using namespace dealii;
    SolverCG<vector_type> solver(solver_control);
    PreconditionJacobi<matrix_type> preconditioner;
    preconditioner.initialize(this->get_matrix());
    solver.solve(this->matrix, gradient_solution, rhs_vector, preconditioner);
  }


  /* ************************************************** */
  /** Base class for Local Integrators                  */
  /* ************************************************** */
  namespace internal
  {
    namespace meshworker
    {
      template <typename LinsysType>
      void LDGGradientAssembler<LinsysType>::IntegratorBase::cell(
        dof_info_t& dinfo, integration_info_t& cinfo) const
      {
        using namespace dealii;
        const FEValuesBase<dim>& fe = cinfo.fe_values();
        const auto& JxW = fe.get_JxW_values();
        const auto ndof = fe.dofs_per_cell;
        const auto nqpt = fe.n_quadrature_points;
        const auto soln_at_qpt =
          this->fe_fcn_selector.values("soln", AssemblyWorker::cell, cinfo)[0];

        auto& local_rhs = dinfo.vector(0).block(0);

        using nqpt_type = typename std::remove_const<decltype(nqpt)>::type;
        using ndof_type = typename std::remove_const<decltype(ndof)>::type;

        for (nqpt_type iq = 0; iq < nqpt; ++iq)
          for (ndof_type i = 0; i < ndof; ++i)
            local_rhs[i] -=
              fe.shape_grad(i, iq)[control.dim_idx] * soln_at_qpt[iq] * JxW[iq];
      }


      /* ************************************* */
      /** Specialization for alternating flux  */
      /* ************************************* */

      template <typename LinsysType>
      template <typename Dummy>
      void LDGGradientAssembler<LinsysType>::Integrator<
        LDGFluxEnum::alternating, Dummy>::cell(dof_info_t& dinfo,
                                               integration_info_t& cinfo) const
      {
        using namespace dealii;

        ASSERT(cinfo.finite_element().n_components() == dim,
               ExcWorksOnlyInSpaceDim(dim));
        ASSERT(cinfo.finite_element().element_multiplicity(0),
               EXCEPT_MSG("All components of FE must be of the same type."));

        const FEValuesBase<dim>& fe = cinfo.fe_values();
        const auto ndof = fe.dofs_per_cell;
        const auto nqpt = fe.n_quadrature_points;
        const auto& JxW = fe.get_JxW_values();
        const auto soln_qpt =
          this->fe_fcn_selector.values("soln", AssemblyWorker::cell, cinfo)[0];

        auto& local_rhs = dinfo.vector(0);

        using nqpt_type = typename std::remove_const<decltype(nqpt)>::type;
        using ndof_type = typename std::remove_const<decltype(ndof)>::type;

        for (nqpt_type iq = 0; iq < nqpt; ++iq)
          for (ndof_type i = 0; i < ndof; ++i)
            for (int idim = 0; idim < dim; ++idim)
              local_rhs.block(idim)[i] -=
                fe.shape_grad(i, iq)[idim] * soln_qpt[iq] * JxW[iq];
      }


      template <typename LinsysType>
      template <typename Dummy>
      void LDGGradientAssembler<LinsysType>::Integrator<
        LDGFluxEnum::alternating, Dummy>::face(dof_info_t& dinfo_in,
                                               dof_info_t& dinfo_ex,
                                               integration_info_t& cinfo_in,
                                               integration_info_t& cinfo_ex)
        const
      {
        using namespace dealii;

        const FEValuesBase<dim>& fe_face_in = cinfo_in.fe_values();
        const FEValuesBase<dim>& fe_face_ex = cinfo_ex.fe_values();

        const auto ndofs_in = fe_face_in.dofs_per_cell;
        const auto ndofs_ex = fe_face_ex.dofs_per_cell;
        const auto nqpt = fe_face_in.n_quadrature_points;
        const auto& JxW = fe_face_in.get_JxW_values();
        const auto& normals = fe_face_in.get_normal_vectors();
        const Tensor<1, dim, value_type>& beta = this->control.beta;

        auto& local_rhs_in = dinfo_in.vector(0);
        auto& local_rhs_ex = dinfo_ex.vector(0);

        const auto soln_qpt_in = this->fe_fcn_selector.values(
          "soln", AssemblyWorker::face, cinfo_in)[0];
        const auto soln_qpt_ex = this->fe_fcn_selector.values(
          "soln", AssemblyWorker::face, cinfo_ex)[0];

        using nqpt_type = typename std::remove_const<decltype(nqpt)>::type;
        using ndof_type = typename std::remove_const<decltype(ndofs_in)>::type;

        for (nqpt_type iq = 0; iq < nqpt; ++iq) {
          // central flux and beta flux
          value_type central_flux = 0.5 * (soln_qpt_in[iq] + soln_qpt_ex[iq]);
          value_type beta_flux =
            beta * normals[iq] * (soln_qpt_ex[iq] - soln_qpt_in[iq]);

          for (int idim = 0; idim < dim; ++idim) {
            for (ndof_type i = 0; i < ndofs_in; ++i)
              local_rhs_in.block(idim)[i] +=
                fe_face_in.shape_value(i, iq) * normals[iq][idim] *
                (central_flux + beta_flux) * JxW[iq];
            for (ndof_type i = 0; i < ndofs_ex; ++i)
              local_rhs_ex.block(idim)[i] -=
                fe_face_ex.shape_value(i, iq) * normals[iq][idim] *
                (central_flux + beta_flux) * JxW[iq];
          }  // idim-loop
        }    // iq-loop
      }


      template <typename LinsysType>
      template <typename Dummy>
      void LDGGradientAssembler<LinsysType>::Integrator<
        LDGFluxEnum::alternating, Dummy>::boundary(dof_info_t& dinfo,
                                                   integration_info_t& cinfo)
        const
      {
        using namespace dealii;
        const FEValuesBase<dim>& fe = cinfo.fe_values();
        auto& local_rhs = dinfo.vector(0);
        const auto ndof = fe.dofs_per_cell;
        const auto nqpt = fe.n_quadrature_points;
        const auto soln_at_qpt = this->fe_fcn_selector.values(
          "soln", AssemblyWorker::boundary, cinfo)[0];
        const auto& JxW = fe.get_JxW_values();
        const auto& normals = fe.get_normal_vectors();

        using nqpt_type = typename std::remove_const<decltype(nqpt)>::type;
        using ndof_type = typename std::remove_const<decltype(ndof)>::type;

        for (nqpt_type iq = 0; iq < nqpt; ++iq) {
          // const Point<dim> qpt = fe.quadrature_point(iq);
          // auto idx = this->bcs.boundary_id_at_point(qpt);
          int idx = 0;

          for (int idim = 0; idim < dim; ++idim) {
            if (idx == 0)  // no boundary condition is imposed
              for (ndof_type i = 0; i < ndof; ++i)
                // treat as a Neumann condition
                local_rhs.block(idim)[i] += fe.shape_value(i, iq) *
                                            normals[iq][idim] *
                                            soln_at_qpt[iq] * JxW[iq];
            else  // TODO implement BC
              ASSERT(false, ExcNotImplemented());
          }  // idim-loop
        }    // iq-loop
      }


      /* ********************************** */
      /** Specialization for left flux      */
      /* ********************************** */
      template <typename LinsysType>
      template <typename Dummy>
      void LDGGradientAssembler<LinsysType>::Integrator<
        LDGFluxEnum::left, Dummy>::face(dof_info_t& dinfo_in,
                                        dof_info_t& dinfo_ex,
                                        integration_info_t& cinfo_in,
                                        integration_info_t& cinfo_ex) const
      {
        using namespace dealii;

        const FEValuesBase<dim>& fe_face_in = cinfo_in.fe_values();
        const FEValuesBase<dim>& fe_face_ex = cinfo_ex.fe_values();
        const auto ndofs_in = fe_face_in.dofs_per_cell;
        const auto ndofs_ex = fe_face_ex.dofs_per_cell;
        const auto& JxW = fe_face_in.get_JxW_values();
        const auto& normals = fe_face_in.get_normal_vectors();
        const auto nqpt = fe_face_in.n_quadrature_points;

        auto& local_rhs_in = dinfo_in.vector(0).block(0);
        auto& local_rhs_ex = dinfo_ex.vector(0).block(0);

        const auto soln_at_qpt_in = this->fe_fcn_selector.values(
          "soln", AssemblyWorker::face, cinfo_in)[0];
        const auto soln_at_qpt_ex = this->fe_fcn_selector.values(
          "soln", AssemblyWorker::face, cinfo_ex)[0];

        using nqpt_type = typename std::remove_const<decltype(nqpt)>::type;
        using ndof_type = typename std::remove_const<decltype(ndofs_in)>::type;

        for (nqpt_type iq = 0; iq < nqpt; ++iq) {
          if (normals[iq][this->control.dim_idx] < 0.0) {
            // term to be added to the internal trial function
            for (ndof_type i = 0; i < ndofs_in; ++i)
              local_rhs_in[i] += normals[iq][this->control.dim_idx] *
                                 fe_face_in.shape_value(i, iq) *
                                 soln_at_qpt_ex[iq] * JxW[iq];
            // term to be added to the external trial function
            for (ndof_type i = 0; i < ndofs_ex; ++i)
              local_rhs_ex[i] -= normals[iq][this->control.dim_idx] *
                                 fe_face_ex.shape_value(i, iq) *
                                 soln_at_qpt_ex[iq] * JxW[iq];
          } else {
            // term to be added to the internal trial function
            for (ndof_type i = 0; i < ndofs_in; ++i)
              local_rhs_in[i] += normals[iq][this->control.dim_idx] *
                                 fe_face_in.shape_value(i, iq) *
                                 soln_at_qpt_in[iq] * JxW[iq];
            // term to be added to the external trial function
            for (ndof_type i = 0; i < ndofs_ex; ++i)
              local_rhs_ex[i] -= normals[iq][this->control.dim_idx] *
                                 fe_face_ex.shape_value(i, iq) *
                                 soln_at_qpt_in[iq] * JxW[iq];
          }  // normals[dim_idx] < 0
        }    // iqpt-loop
      }


      template <typename LinsysType>
      template <typename Dummy>
      void LDGGradientAssembler<LinsysType>::Integrator<
        LDGFluxEnum::left, Dummy>::boundary(dof_info_t& dinfo,
                                            integration_info_t& cinfo) const
      {
        using namespace dealii;

        const FEValuesBase<dim>& fe = cinfo.fe_values();
        const auto ndof = fe.dofs_per_cell;
        const auto nqpt = fe.n_quadrature_points;
        const auto& normals = fe.get_normal_vectors();
        const auto& JxW = fe.get_JxW_values();

        const auto soln_at_qpt = this->fe_fcn_selector.values(
          "soln", AssemblyWorker::boundary, cinfo)[0];

        auto& local_rhs = dinfo.vector(0).block(0);

        using nqpt_type = typename std::remove_const<decltype(nqpt)>::type;
        using ndof_type = typename std::remove_const<decltype(ndof)>::type;

        for (nqpt_type iq = 0; iq < nqpt; ++iq) {
          Point<dim> qpt = fe.quadrature_point(iq);
          const auto normal_comp = normals[iq][this->control.dim_idx];

          if (normal_comp < 0.0) {
            bool use_zero_neumann_bc = true;
            value_type bdry_val = 0.0;

            // const auto idx = this->bcs.boundary_id_at_point(qpt);
            const auto idx = dinfo.face->boundary_id();

            if (idx > 0 && this->bcs.has_boundary_id(idx)) {
              for (const auto& pbc : this->bcs(idx)) {
                if (pbc->get_category() == BCCategory::dirichlet) {
                  bdry_val = pbc->value(qpt);
                  use_zero_neumann_bc = false;
                }
              }
            }

            if (use_zero_neumann_bc) bdry_val = soln_at_qpt[iq];
            for (ndof_type i = 0; i < ndof; ++i)
              local_rhs[i] +=
                normal_comp * fe.shape_value(i, iq) * bdry_val * JxW[iq];

          } else {
            // normal_comp >= 0.0 //
            for (ndof_type i = 0; i < ndof; ++i)
              local_rhs[i] +=
                normal_comp * fe.shape_value(i, iq) * soln_at_qpt[iq] * JxW[iq];
          }
        }  // iq-loop
      }


      /* ******************************* */
      /** Specialization for right flux  */
      /* ******************************* */

      template <typename LinsysType>
      template <typename Dummy>
      void LDGGradientAssembler<LinsysType>::Integrator<
        LDGFluxEnum::right, Dummy>::face(dof_info_t& dinfo_in,
                                         dof_info_t& dinfo_ex,
                                         integration_info_t& cinfo_in,
                                         integration_info_t& cinfo_ex) const
      {
        using namespace dealii;

        const FEValuesBase<dim>& fe_face_in = cinfo_in.fe_values();
        const FEValuesBase<dim>& fe_face_ex = cinfo_ex.fe_values();
        const auto ndofs_in = fe_face_in.dofs_per_cell;
        const auto ndofs_ex = fe_face_ex.dofs_per_cell;
        const auto& JxW = fe_face_in.get_JxW_values();
        const std::vector<Tensor<1, dim>>& normals =
          fe_face_in.get_normal_vectors();
        const auto nqpt = fe_face_in.n_quadrature_points;

        auto& local_rhs_in = dinfo_in.vector(0).block(0);
        auto& local_rhs_ex = dinfo_ex.vector(0).block(0);

        const auto soln_at_qpt_in = this->fe_fcn_selector.values(
          "soln", AssemblyWorker::face, cinfo_in)[0];
        const auto soln_at_qpt_ex = this->fe_fcn_selector.values(
          "soln", AssemblyWorker::face, cinfo_ex)[0];

        using nqpt_type = typename std::remove_const<decltype(nqpt)>::type;
        using ndof_type = typename std::remove_const<decltype(ndofs_in)>::type;

        for (nqpt_type iq = 0; iq < fe_face_in.n_quadrature_points; ++iq) {
          if (normals[iq][this->control.dim_idx] < 0.0) {
            // term to be added to internal trial function
            for (ndof_type i = 0; i < ndofs_in; ++i)
              local_rhs_in[i] += normals[iq][this->control.dim_idx] *
                                 fe_face_in.shape_value(i, iq) *
                                 soln_at_qpt_in[iq] * JxW[iq];
            // term to be added to external trial function
            for (ndof_type i = 0; i < ndofs_ex; ++i)
              local_rhs_ex[i] -= normals[iq][this->control.dim_idx] *
                                 fe_face_ex.shape_value(i, iq) *
                                 soln_at_qpt_in[iq] * JxW[iq];
          } else {
            // terms to be added to internal trial function
            for (ndof_type i = 0; i < ndofs_in; ++i)
              local_rhs_in[i] += normals[iq][this->control.dim_idx] *
                                 fe_face_in.shape_value(i, iq) *
                                 soln_at_qpt_ex[iq] * JxW[iq];
            // terms to be added to external trial function
            for (ndof_type i = 0; i < ndofs_ex; ++i)
              local_rhs_ex[i] -= normals[iq][this->control.dim_idx] *
                                 fe_face_ex.shape_value(i, iq) *
                                 soln_at_qpt_ex[iq] * JxW[iq];
          }  // normals[this->dim_idx] < 0.0
        }    // qpt-loop
      }


      template <typename LinsysType>
      template <typename Dummy>
      void LDGGradientAssembler<LinsysType>::Integrator<
        LDGFluxEnum::right, Dummy>::boundary(dof_info_t& dinfo,
                                             integration_info_t& cinfo) const
      {
        using namespace dealii;
        const FEValuesBase<dim>& fe = cinfo.fe_values();
        const auto ndof = fe.dofs_per_cell;
        const auto nqpt = fe.n_quadrature_points;
        const auto& normals = fe.get_normal_vectors();
        const auto& JxW = fe.get_JxW_values();

        const auto soln_at_qpt = this->fe_fcn_selector.values(
          "soln", AssemblyWorker::boundary, cinfo)[0];

        using nqpt_type = typename std::remove_const<decltype(nqpt)>::type;
        using ndof_type = typename std::remove_const<decltype(ndof)>::type;

        auto& local_rhs = dinfo.vector(0).block(0);

        for (nqpt_type iq = 0; iq < fe.n_quadrature_points; ++iq) {
          Point<dim> qpt = fe.quadrature_point(iq);
          const auto normal_comp = normals[iq][this->control.dim_idx];

          if (normal_comp < 0.0) {
            for (ndof_type i = 0; i < ndof; ++i)
              local_rhs[i] +=
                normal_comp * fe.shape_value(i, iq) * soln_at_qpt[iq] * JxW[iq];
          } else {
            // normal_comp >= 0.0 //
            bool use_zero_neumann_bc = true;
            value_type bdry_val = 0.0;

            //  compute the boundary value
            // auto idx = this->bcs.boundary_id_at_point(qpt);
            const auto idx = dinfo.face->boundary_id();

            if (idx > 0 && this->bcs.has_boundary_id(idx)) {
              for (const auto& pbc : this->bcs(idx)) {
                if (pbc->get_category() == BCCategory::dirichlet) {
                  bdry_val = pbc->value(qpt);
                  use_zero_neumann_bc = false;
                }
              }  // loop thru all bcs at this boundary id
            }

            if (use_zero_neumann_bc) bdry_val = soln_at_qpt[iq];

            for (ndof_type i = 0; i < ndof; ++i)
              local_rhs[i] +=
                normal_comp * fe.shape_value(i, iq) * bdry_val * JxW[iq];
          }
        }  // iq-loop //
      }


      /* ************************************************** */
      /**
       * Local integrator for \class LDGDifffusionIntegrator.
       * Assemble RHS for explicit diffusion computations.
       */
      /* ************************************************** */

      template <typename LinsysType>
      void LDGDiffusionAssembler<LinsysType>::IntegratorBase::cell(
        dof_info_t& dinfo, integration_info_t& cinfo) const
      {
        using namespace dealii;

        const FEValuesBase<dim>& fe = cinfo.fe_values();
        const auto& JxW = fe.get_JxW_values();
        const auto nqpt = fe.n_quadrature_points;
        const auto ndof = fe.dofs_per_cell;
        const auto gradients = cinfo.values;

        using nqpt_type = typename std::remove_const<decltype(nqpt)>::type;
        using ndof_type = typename std::remove_const<decltype(ndof)>::type;

        std::vector<Tensor<1, dim, value_type>> grad_phi_qpt(nqpt);
        for (int idim = 0; idim < dim; ++idim)
          for (nqpt_type iq = 0; iq < nqpt; ++iq)
            grad_phi_qpt[iq][idim] = gradients[idim][0][iq];

        auto& local_rhs = dinfo.vector(0).block(0);
        const auto viscosity = this->control.viscosity;
        for (nqpt_type iq = 0; iq < nqpt; ++iq)
          for (ndof_type i = 0; i < ndof; ++i)
            local_rhs[i] -=
              viscosity * (fe.shape_grad(i, iq) * grad_phi_qpt[iq]) * JxW[iq];
      }


      /* ************************************************** */
      /** Specialization for alternating flux */
      /* ************************************************** */

      template <typename LinsysType>
      template <typename Dummy>
      void LDGDiffusionAssembler<LinsysType>::Integrator<
        LDGFluxEnum::alternating, Dummy>::face(dof_info_t& dinfo_in,
                                               dof_info_t& dinfo_ex,
                                               integration_info_t& cinfo_in,
                                               integration_info_t& cinfo_ex)
        const
      {
        using namespace dealii;

        const FEValuesBase<dim>& fe_in = cinfo_in.fe_values();
        const FEValuesBase<dim>& fe_ex = cinfo_ex.fe_values();
        const auto ndof_in = fe_in.dofs_per_cell;
        const auto ndof_ex = fe_ex.dofs_per_cell;
        const auto nqpt = fe_in.n_quadrature_points;
        const auto JxW = fe_in.get_JxW_values();
        const auto& normals = fe_in.get_normal_vectors();

        auto& local_rhs_in = dinfo_in.vector(0).block(0);
        auto& local_rhs_ex = dinfo_ex.vector(0).block(0);

        using nqpt_type = typename std::remove_const<decltype(nqpt)>::type;
        using ndof_type = typename std::remove_const<decltype(ndof_in)>::type;

        std::vector<Tensor<1, dim, value_type>> grad_qpt_in(nqpt),
          grad_qpt_ex(nqpt);

        for (int idim = 0; idim < dim; ++idim) {
          const auto grad_block_in = cinfo_in.values[idim][0];
          const auto grad_block_ex = cinfo_ex.values[idim][0];
          for (nqpt_type iq = 0; iq < nqpt; ++iq) {
            grad_qpt_in[iq][idim] = grad_block_in[iq];
            grad_qpt_ex[iq][idim] = grad_block_ex[iq];
          }  // iq-loop
        }    // idim-loop

        const auto viscosity = this->control.viscosity;
        const auto& beta = this->control.beta;

        for (nqpt_type iq = 0; iq < nqpt; ++iq) {
          const Tensor<1, dim, value_type> central_flux =
            0.5 * (grad_qpt_in[iq] + grad_qpt_ex[iq]);
          const Tensor<1, dim, value_type> beta_flux =
            normals[iq] * beta * (grad_qpt_ex[iq] - grad_qpt_in[iq]);

          for (ndof_type i = 0; i < ndof_in; ++i)
            local_rhs_in[i] += viscosity * fe_in.shape_value(i, iq) *
                               (central_flux - beta_flux) * normals[iq] *
                               JxW[iq];
          for (ndof_type i = 0; i < ndof_ex; ++i)
            local_rhs_ex[i] -= viscosity * fe_ex.shape_value(i, iq) *
                               (central_flux - beta_flux) * normals[iq] *
                               JxW[iq];
        }  // iq-loop
      }


      template <typename LinsysType>
      template <typename Dummy>
      void LDGDiffusionAssembler<LinsysType>::Integrator<
        LDGFluxEnum::alternating, Dummy>::boundary(dof_info_t& dinfo,
                                                   integration_info_t& cinfo)
        const
      {
        UNUSED_VARIABLE(cinfo);
        UNUSED_VARIABLE(dinfo);
      }

    }  // namespace meshworker
  }    // namespace internal

}  // namespace dg

/* -------- Explicit Instantiations ----------*/
#include "ldg.inst"
/* -------------------------------------------*/
FELSPA_NAMESPACE_CLOSE
