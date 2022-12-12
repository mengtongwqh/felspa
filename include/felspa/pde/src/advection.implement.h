#ifndef _FELSPA_PDE_ADVECTION_IMPLEMENTATION_H_
#define _FELSPA_PDE_ADVECTION_IMPLEMENTATION_H_

#include <felspa/fe/cell_data.h>
#include <felspa/pde/advection.h>

#include <array>

FELSPA_NAMESPACE_OPEN

/* -------------------------------------------*/
namespace dg
/* -------------------------------------------*/
{
  /* ************************************************** */
  /**
   * \class AdvectSimulator.
   * Function template Implementations
   */
  /* ************************************************** */

  template <int dim, typename NumberType>
  template <typename VeloFcnType>
  void AdvectSimulator<dim, NumberType>::do_initialize(
    const std::shared_ptr<VeloFcnType>& pvfield,
    const std::shared_ptr<source_term_type>& psource,
    bool use_independent_solution)
  {
    ptr_source_term = psource;
    ptr_velocity_field = pvfield;
    ptr_cfl_estimator = std::make_unique<CFLEstimator<VeloFcnType>>(*pvfield);

    this->temporal_passive_members.clear();
    this->add_temporal_passive_member(pvfield);
    if (psource != nullptr) this->add_temporal_passive_member(psource);

    felspa_log << "RHS vector will be assembled with "
               << control().assembly_framework << " method." << std::endl;

    switch (control().assembly_framework) {
      case AssemblyFramework::meshworker:
        // meshworker type asssembly not implemented for non TensorFunction type
        if constexpr (std::is_base_of<TensorFunction<1, dim, value_type>,
                                      VeloFcnType>::value)
          ptr_rhs_assembler = std::make_unique<
            AdvectAssembler<AssemblyFramework::meshworker, VeloFcnType>>(
            this->linear_system(), pvfield);
        else
          THROW(ExcNotImplemented());
        break;

      case AssemblyFramework::workstream:
        ptr_rhs_assembler = std::make_unique<
          AdvectAssembler<AssemblyFramework::workstream, VeloFcnType>>(
          this->linear_system(), pvfield);
        break;

      default:
        THROW(ExcNotImplemented());
    }

    // set all object members to time 0
    this->reset_time();

    // this is needed because the dof_handler.distribute_dofs()
    // will not be called if the simulator is not marked initialized.
    this->initialized = true;

    // distribute_dof and reset constraints
    upon_mesh_update();

    // allocate solution object
    if (!this->primary_simulator && use_independent_solution)
      this->solution.reinit(std::make_shared<vector_type>(), 0.0);

    // allocate solution vector
    if (this->solution.is_independent())
      this->solution->reinit(this->dof_handler().n_dofs());

    ASSERT(
      this->dof_handler().n_dofs() == this->solution->size(),
      ExcSizeMismatch(this->dof_handler().n_dofs(), this->solution->size()));
  }


  template <int dim, typename NumberType>
  template <typename VeloFcnType>
  void AdvectSimulator<dim, NumberType>::initialize(
    const ScalarFunction<dim, value_type>& initial_condition,
    const std::shared_ptr<VeloFcnType>& pvfield,
    const std::shared_ptr<source_term_type>& psource, bool execute_mesh_refine,
    bool use_independent_solution)
  {
    do_initialize(pvfield, psource, use_independent_solution);

    // set initial condition by interpolating function to solution points
    discretize_function_to_solution(initial_condition);

    // mesh refine since we have the closed-form initial condition
    MeshControl<value_type>& mesh_ctrl = *this->ptr_control->ptr_mesh;

    if (execute_mesh_refine && this->ptr_mesh_refiner) {
      felspa_log << "Recursively refine mesh to level " << mesh_ctrl.max_level
                 << " to better resolve initial condition..." << std::endl;

      // we need refine one more time over the maximum level
      // so that the updated min_diameter at maximum refinement level
      // will take effect.
      for (auto ilevel = mesh_ctrl.min_level; ilevel <= mesh_ctrl.max_level;
           ++ilevel) {
        this->refine_mesh(mesh_ctrl);
        discretize_function_to_solution(initial_condition);
      }
    }
  }


  template <int dim, typename NumberType>
  template <typename VeloFcnType>
  void AdvectSimulator<dim, NumberType>::initialize(
    const vector_type& initial_condition,
    const std::shared_ptr<VeloFcnType>& pvfield,
    const std::shared_ptr<source_term_type>& psource,
    bool use_independent_solution)
  {
    do_initialize(pvfield, psource, use_independent_solution);

    // set initial condition by copying solution vector
    ASSERT_SAME_SIZE(initial_condition, this->get_solution_vector());
    *(this->solution) = initial_condition;
  }


  template <int dim, typename NumberType>
  template <typename VeloFcnType>
  void AdvectSimulator<dim, NumberType>::initialize(
    vector_type&& initial_condition,
    const std::shared_ptr<VeloFcnType>& pvfield,
    const std::shared_ptr<source_term_type>& psource,
    bool use_independent_solution)
  {
    do_initialize(pvfield, psource, use_independent_solution);

    // set initial condition by swapping memory
    ASSERT_SAME_SIZE(initial_condition, *(this->solution));
    this->solution->swap(initial_condition);
  }


  /* ************************************************** */
  /**
   * AdvectAssembler Implementations
   * specialized for meshworker framework
   */
  /* ************************************************** */
  template <typename VeloFcnType>
  AdvectAssembler<AssemblyFramework::meshworker, VeloFcnType>::AdvectAssembler(
    linsys_type& linsys, const std::weak_ptr<const velocity_fcn_type>& p_vfield)
    : base_type(linsys), ptr_velocity_field(p_vfield)
  {
    const bool is_derived_from_tensor_function =
      std::is_base_of<TensorFunction<1, dim, value_type>, VeloFcnType>::value;

    // forbid construction if VeloFcnType is not derived from TensorFunction.
    ASSERT(
      is_derived_from_tensor_function,
      EXCEPT_MSG("Only implemented for velocity derived from TensorFunction."));
  }


  /**
   * The idea now is to pass velocity field and Boundary conditions
   * as function arguments, re-bind the arguments and pass to \c
   * MeshWorker
   */
  template <typename VeloFcnType>
  void AdvectAssembler<AssemblyFramework::meshworker, VeloFcnType>::assemble(
    const dealii::Quadrature<dim>& quadrature,
    const vector_type& solution,
    const bcs_type& bcs,
    const std::shared_ptr<const source_term_type>& p_source_term)
  {
    UNUSED_VARIABLE(quadrature);

    using this_type = std::remove_reference_t<decltype(*this)>;
    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    using std::placeholders::_4;

    dealii::MeshWorker::IntegrationInfoBox<dim> info_box;

    // Just in case that the assemble() is called multiple times on the
    // same assembler object
    this->fe_fcn_selector.reset();

    // initialize fe_fcn_selector so that FE functions
    // that are defined at dof nodes can be computed at
    // quadrature points
    this->fe_fcn_selector.attach_info_box(info_box);

    // solution from previous step will be computed at quadrature point
    this->fe_fcn_selector.add("soln_prev_step", solution, {true, false, false},
                              {true, false, false}, {true, false, false});

    info_box.add_update_flags_cell(dealii::update_gradients);
    this->fe_fcn_selector.finalize(this->dof_handler().get_fe(),
                                   this->get_mapping());

    // initialize the IntegrationInfo data structure
    dealii::MeshWorker::DoFInfo<dim, dim, value_type> dof_info(
      this->dof_handler());

    // Assembler for the RHS vector
    dealii::MeshWorker::Assembler::ResidualSimple<vector_type> assembler;
    dealii::AnyData data;
    data.add<vector_type*>(&this->rhs(), "rhs");
    assembler.initialize(data);
    assembler.initialize(this->constraints());

    // zero out the linear system and prepare for assembly
    this->ptr_linear_system->zero_out(false, true);

    auto cell_integrator = std::bind(&this_type::assemble_cell, *this, _1, _2,
                                     std::cref(p_source_term));
    auto boundary_integrator =
      std::bind(&this_type::assemble_boundary, *this, _1, _2, std::cref(bcs));
    auto face_integrator =
      std::bind(&this_type::assemble_face, *this, _1, _2, _3, _4);

    dealii::MeshWorker::loop<dim, dim,
                             dealii::MeshWorker::DoFInfo<dim, dim, value_type>,
                             dealii::MeshWorker::IntegrationInfoBox<dim, dim>>(
      this->dof_handler().begin_active(), this->dof_handler().end(), dof_info,
      info_box, cell_integrator, boundary_integrator, face_integrator,
      assembler);
  }


  template <typename VeloFcnType>
  void
  AdvectAssembler<AssemblyFramework::meshworker, VeloFcnType>::assemble_cell(
    dof_info_t& dinfo, integration_info_t& cinfo,
    const std::shared_ptr<const source_term_type>& p_source_term)
  {
    using namespace dealii;
    const FEValuesBase<dim>& fe_values = cinfo.fe_values();
    const std::vector<Point<dim>>& qpts = fe_values.get_quadrature_points();

    std::vector<Tensor<1, dim, value_type>> velo_at_qpt(qpts.size());
    std::vector<value_type> source_at_qpt(qpts.size(), 0.0);
    if (p_source_term) (*p_source_term)(qpts, source_at_qpt);

    auto sp_velo = ptr_velocity_field.lock();
    ASSERT(sp_velo != nullptr, ExcExpiredPointer());
    (*sp_velo)(qpts, velo_at_qpt);

    const std::vector<double>& JxW = fe_values.get_JxW_values();
    local_vector_t& local_rhs = dinfo.vector(0).block(0);

    // extract local soln
    const auto soln_at_qpt = this->fe_fcn_selector.values(
      "soln_prev_step", AssemblyWorker::cell, cinfo)[0];

    // advection operator
    for (unsigned int iqpt = 0; iqpt < fe_values.n_quadrature_points; ++iqpt)
      for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
        local_rhs[i] += velo_at_qpt[iqpt] * fe_values.shape_grad(i, iqpt) *
                        soln_at_qpt[iqpt] * JxW[iqpt];

    // source term
    if (source_at_qpt.size())
      for (unsigned int iqpt = 0; iqpt < fe_values.n_quadrature_points; ++iqpt)
        for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
          local_rhs[i] +=
            source_at_qpt[iqpt] * fe_values.shape_value(i, iqpt) * JxW[iqpt];
  }


  template <typename VeloFcnType>
  void AdvectAssembler<AssemblyFramework::meshworker,
                       VeloFcnType>::assemble_face(dof_info_t& dinfo_in,
                                                   dof_info_t& dinfo_ex,
                                                   integration_info_t& cinfo_in,
                                                   integration_info_t& cinfo_ex)
  {
    using namespace dealii;

    // face shape functions of this cell
    const FEValuesBase<dim>& fe_face_values_in = cinfo_in.fe_values();
    const unsigned int dofs_per_cell_in = fe_face_values_in.dofs_per_cell;
    // face shape functions of neighboring cell
    const FEValuesBase<dim>& fe_face_values_ex = cinfo_ex.fe_values();
    const unsigned int dofs_per_cell_ex = fe_face_values_ex.dofs_per_cell;

    const std::vector<Point<dim>>& qpts =
      fe_face_values_in.get_quadrature_points();
    std::vector<Tensor<1, dim, value_type>> velo_at_qpt(qpts.size());

    auto sp_velo = this->ptr_velocity_field.lock();
    ASSERT(sp_velo != nullptr, ExcExpiredPointer());

    // compute the function at quadrature points
    std::transform(
      qpts.cbegin(), qpts.cend(), velo_at_qpt.begin(),
      [&](const dealii::Point<dim, value_type>& pt) { return (*sp_velo)(pt); });

    // extract local soln
    const auto soln_at_qpt_in = this->fe_fcn_selector.values(
      "soln_prev_step", AssemblyWorker::face, cinfo_in)[0];
    const auto soln_at_qpt_ex = this->fe_fcn_selector.values(
      "soln_prev_step", AssemblyWorker::face, cinfo_ex)[0];

    // rhs, will be assembled into global rhs vector
    local_vector_t& local_rhs_in = dinfo_in.vector(0).block(0);
    local_vector_t& local_rhs_ex = dinfo_ex.vector(0).block(0);

    const std::vector<Tensor<1, dim>>& normals =
      fe_face_values_in.get_normal_vectors();
    const std::vector<double>& JxW = fe_face_values_in.get_JxW_values();

    for (unsigned int iqpt = 0; iqpt < fe_face_values_in.n_quadrature_points;
         ++iqpt) {
      // compute normal velocity
      const value_type normal_velocity = normals[iqpt] * velo_at_qpt[iqpt];

      // compute local face integrals
      if (normal_velocity > 0.0)  // outgoing flux
      {
        for (unsigned int i = 0; i < dofs_per_cell_in; ++i)
          local_rhs_in[i] -= normal_velocity *
                             fe_face_values_in.shape_value(i, iqpt) *
                             soln_at_qpt_in[iqpt] * JxW[iqpt];
        for (unsigned int i = 0; i < dofs_per_cell_ex; ++i)
          local_rhs_ex[i] += normal_velocity *
                             fe_face_values_ex.shape_value(i, iqpt) *
                             soln_at_qpt_in[iqpt] * JxW[iqpt];
      }  // normal_velocity > 0.0

      else  // incoming flux
      {
        for (unsigned int i = 0; i < dofs_per_cell_in; ++i)
          local_rhs_in[i] -= normal_velocity *
                             fe_face_values_in.shape_value(i, iqpt) *
                             soln_at_qpt_ex[iqpt] * JxW[iqpt];
        for (unsigned int i = 0; i < dofs_per_cell_ex; ++i)
          local_rhs_ex[i] += normal_velocity *
                             fe_face_values_ex.shape_value(i, iqpt) *
                             soln_at_qpt_ex[iqpt] * JxW[iqpt];
      }  // normal_velocity <= 0.0
    }    // iqpt-loop
  }


  template <typename VeloFcnType>
  void
  AdvectAssembler<AssemblyFramework::meshworker,
                  VeloFcnType>::assemble_boundary(dof_info_t& dinfo,
                                                  integration_info_t& cinfo,
                                                  const bcs_type& bcs)
  {
    using types::DoubleType;
    using namespace dealii;

    const FEValuesBase<dim>& fe = cinfo.fe_values();
    const std::vector<Point<dim>> qpts = fe.get_quadrature_points();
    std::vector<Tensor<1, dim, value_type>> velo_at_qpt(qpts.size());

    auto sp_velo = ptr_velocity_field.lock();
    ASSERT(sp_velo != nullptr, ExcExpiredPointer());

    std::transform(
      qpts.cbegin(), qpts.cend(), velo_at_qpt.begin(),
      [&](const dealii::Point<dim, value_type>& pt) { return (*sp_velo)(pt); });


    using bdry_id_type = dealii::types::boundary_id;
    const FEValuesBase<dim>& fe_face_values = cinfo.fe_values();

    const unsigned int dofs_per_cell = fe_face_values.dofs_per_cell;
    const std::vector<double>& JxW = fe_face_values.get_JxW_values();
    const std::vector<Tensor<1, dim>>& normals =
      fe_face_values.get_normal_vectors();

    // local matrix and vector
    local_vector_t& local_rhs = dinfo.vector(0).block(0);

    const auto soln_at_qpt = this->fe_fcn_selector.values(
      "soln_prev_step", AssemblyWorker::boundary, cinfo)[0];

    for (unsigned int iqpt = 0; iqpt < fe_face_values.n_quadrature_points;
         ++iqpt) {
      Point<dim> qpt = fe_face_values.quadrature_point(iqpt);
      DoubleType normal_velocity = velo_at_qpt[iqpt] * normals[iqpt];

      if (normal_velocity > 0)  // outflow boundary
      {
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          local_rhs[i] -= normal_velocity *
                          fe_face_values.shape_value(i, iqpt) *
                          soln_at_qpt[iqpt] * JxW[iqpt];
      }  // normal_velocity > 0.0

      else  // inflow boundary
      {
        bool use_zero_neumann_bc = true;  // default bc
        bdry_id_type idx = dinfo.face->boundary_id();

        if (idx > 0 && bcs.has_boundary_id(idx)) {
          for (const auto pbc : bcs(idx)) {
            if (pbc->get_category() == BCCategory::dirichlet) {
              // compute the Dirichlet boundary value
              auto bdry_val = pbc->value(qpt);
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                local_rhs[i] -= normal_velocity *
                                fe_face_values.shape_value(i, iqpt) * bdry_val *
                                JxW[iqpt];
              use_zero_neumann_bc = false;
            }
          }  // loop thru bcs at this id
        }    

        if (use_zero_neumann_bc)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            local_rhs[i] -= normal_velocity *
                            fe_face_values.shape_value(i, iqpt) *
                            soln_at_qpt[iqpt] * JxW[iqpt];
      }  // normal_velocity <= 0.0
    }    // iqpt-loop
  }


  /* ************************************************** */
  /**
   * AdvectAssembler Implementations
   * specialized for workstream
   */
  /* ************************************************** */

  template <typename VeloFcnType>
  void AdvectAssembler<AssemblyFramework::workstream, VeloFcnType>::assemble(
    const dealii::Quadrature<dim>& quadrature, const vector_type& soln,
    const bcs_type& bcs,
    const std::shared_ptr<const source_term_type>& ptr_source_term)
  {
    using namespace dealii;
    using AssemblerType = std::remove_reference_t<decltype(*this)>;

    ptr_bcs = &bcs;

    const UpdateFlags update_flags = update_values | update_gradients |
                                     update_JxW_values |
                                     update_quadrature_points;

    const auto& quad_gauss =
      dynamic_cast<const dealii::QGauss<dim>&>(quadrature);

    ScratchDataBox scratch_box(this->get_mapping(),
                               this->dof_handler().get_fe(),
                               quad_gauss,
                               update_flags,
                               this->ptr_velocity_field,
                               soln,
                               ptr_source_term);

    CopyDataBox copy_box(this->dof_handler().get_fe().dofs_per_cell);

    SyncedActiveIterators<dim> sync_iters_begin{
      this->dof_handler().begin_active()},
      sync_iters_end{this->dof_handler().end()};

    if constexpr (is_simulator<VeloFcnType>::value) {
      auto sp_velo = ptr_velocity_field.lock();
      ASSERT(sp_velo != nullptr, ExcExpiredPointer());
      sync_iters_begin.append(sp_velo->get_dof_handler().begin_active());
      sync_iters_end.append(sp_velo->get_dof_handler().end());
    }

    this->ptr_linear_system->zero_out(false, true);

    // call workstream::run to assemble with multiple threads
    WorkStream::run(
      sync_iters_begin, sync_iters_end, *this, &AssemblerType::local_assembly,
      &AssemblerType::copy_local_to_global, scratch_box, copy_box);
  }


  template <typename VeloFcnType>
  void
  AdvectAssembler<AssemblyFramework::workstream, VeloFcnType>::local_assembly(
    const synced_iterators_type& cell_sync, ScratchDataBox& scratch_box,
    CopyDataBox& copy_box)
  {
    using ScratchDataType = ScratchData;
    ASSERT(cell_sync.is_synchronized(), ExcInternalErr());

    const bcs_type& bcs = *ptr_bcs;

    const cell_iterator_type& cell = cell_sync;
    copy_box.reset();

    // CELL INTEGRATION //
    copy_box.cell.reinit(cell);
    cell_assembly(scratch_box.reinit_cell(cell_sync), copy_box.cell);
    copy_box.cell.set_active();

    for (unsigned int face_no = 0;
         face_no < dealii::GeometryInfo<dim>::faces_per_cell;
         ++face_no) {
      const bool has_periodic_neighbor = cell->has_periodic_neighbor(face_no);
      synced_iterators_type neighbor_sync(cell_sync);

      // BOUNDARY FACE INTEGRATION //
      if (cell->at_boundary(face_no) && !cell->has_periodic_neighbor(face_no)) {
        const ScratchDataType& bdry_scratch =
          scratch_box.reinit_face(cell_sync, face_no);
        CopyData& copy = copy_box.interior_faces[bdry_scratch.get_face_no()];

        copy.reinit(cell);
        boundary_assembly(bdry_scratch, copy, bcs);
        copy.set_active();
      }

      // INTERNAL FACE INTEGRATION //
      else {
        // TriaAccessor<CellDoFAccessor>
        // Note that this may not be an active cell iterator
        // (which is of the typename TriaActiveAccessor<CellDoFAccessor>)
        // because the neighbor may be more refined.
        const auto neighbor = cell->neighbor_or_periodic_neighbor(face_no);

        // In this case the neighbor is coarser //
        const bool neighbor_is_coarser =
          (has_periodic_neighbor &&
           cell->periodic_neighbor_is_coarser(face_no)) ||
          (!has_periodic_neighbor && cell->neighbor_is_coarser(face_no));

        if (neighbor_is_coarser) {
          ASSERT(!cell->has_children(), ExcInternalErr());
          ASSERT(!neighbor->has_children(), ExcInternalErr());


          const std::pair<unsigned int, unsigned int> neighbor_face_subface_no =
            has_periodic_neighbor
              ? cell->periodic_neighbor_of_coarser_periodic_neighbor(face_no)
              : cell->neighbor_of_coarser_neighbor(face_no);

          // scratch_box.reinit_faces_coarser_neighbor(cell_sync, face_no);
          neighbor_sync.synchronize_to(neighbor);
          const auto face_scratch_pair = scratch_box.reinit_faces(
            cell_sync, face_no, neighbor_sync, neighbor_face_subface_no.first,
            neighbor_face_subface_no.second);

          const ScratchDataType& scratch_in = face_scratch_pair.first;
          const ScratchDataType& scratch_ex = face_scratch_pair.second;
          CopyData& copy_in = copy_box.interior_faces[scratch_in.get_face_no()];
          CopyData& copy_ex = copy_box.exterior_faces[scratch_ex.get_face_no()];

          copy_in.reinit(cell);
          copy_ex.reinit(neighbor);
          face_assembly(scratch_in, scratch_ex, copy_in, copy_ex);
          copy_in.set_active();
          copy_ex.set_active();
        }

        // Both cells are on the same refinement level, assemble from
        // the one side where cell->id() is smaller
        else if (!neighbor->has_children() && cell->id() < neighbor->id()) {
          // Both will have the same refinement levels
          ASSERT(cell->level() == neighbor->level(), ExcInternalErr());
          // Don't expect any of the cells to have children
          ASSERT(!cell->has_children(), ExcInternalErr());
          ASSERT(!neighbor->has_children(), ExcInternalErr());

          const unsigned int neighbor_face_no =
            has_periodic_neighbor
              ? cell->periodic_neighbor_of_periodic_neighbor(face_no)
              : cell->neighbor_of_neighbor(face_no);

          // const std::pair<ScratchDataType&, ScratchDataType&>
          // face_scratch_pair = scratch_box.reinit_faces_same_level(cell_sync,
          // face_no);
          neighbor_sync.synchronize_to(neighbor);
          const std::pair<ScratchDataType&, ScratchDataType&>
            face_scratch_pair = scratch_box.reinit_faces(
              cell_sync, face_no, neighbor_sync, neighbor_face_no);

          const ScratchDataType& scratch_in = face_scratch_pair.first;
          const ScratchDataType& scratch_ex = face_scratch_pair.second;
          CopyData& copy_in = copy_box.interior_faces[scratch_in.get_face_no()];
          CopyData& copy_ex = copy_box.exterior_faces[scratch_ex.get_face_no()];

          copy_in.reinit(cell);
          copy_ex.reinit(neighbor);
          face_assembly(scratch_in, scratch_ex, copy_in, copy_ex);
          copy_in.set_active();
          copy_ex.set_active();
        }

        // In all other cases, skip and go ahead to the next face
        else
          continue;
      }  // if internal face
    }    // face_no-loop
  }


  template <typename VeloFcnType>
  void AdvectAssembler<AssemblyFramework::workstream, VeloFcnType>::
    copy_local_to_global(const CopyDataBox& copy_data_box)
  {
    // assemble RHS of the current cell
    copy_data_box.assemble(this->constraints(), this->rhs());
  }


  template <typename VeloFcnType>
  void AdvectAssembler<AssemblyFramework::workstream,
                       VeloFcnType>::cell_assembly(const ScratchData& s,
                                                   CopyData& copy)
  {
    using namespace dealii;
    const FEValuesBase<dim>& fe = s.fe_values();
    const auto ndof = fe.dofs_per_cell;
    const auto nqpt = fe.n_quadrature_points;
    auto& local_rhs = copy.vector();


    const std::vector<value_type>& JxW = fe.get_JxW_values();

    // advection operator
    for (unsigned int idof = 0; idof < ndof; ++idof)
      for (unsigned int iqpt = 0; iqpt < nqpt; ++iqpt)
        local_rhs[idof] += s.get_velocity()[iqpt] * fe.shape_grad(idof, iqpt) *
                           s.get_soln()[iqpt] * JxW[iqpt];

    // source term
    if (s.ptr_source_term.get())
      for (unsigned int idof = 0; idof < ndof; ++idof)
        for (unsigned int iqpt = 0; iqpt < nqpt; ++iqpt)
          local_rhs[idof] +=
            s.get_source()[iqpt] * fe.shape_value(idof, iqpt) * JxW[iqpt];
  }


  template <typename VeloFcnType>
  void AdvectAssembler<AssemblyFramework::workstream,
                       VeloFcnType>::face_assembly(const ScratchData& s_in,
                                                   const ScratchData& s_ex,
                                                   CopyData& copy_in,
                                                   CopyData& copy_ex)
  {
    using namespace dealii;
    const FEValuesBase<dim>& fe_in = s_in.fe_values();
    const FEValuesBase<dim>& fe_ex = s_ex.fe_values();

    const auto ndof_in = fe_in.dofs_per_cell;
    const auto ndof_ex = fe_ex.dofs_per_cell;
    const auto nqpt_in = fe_in.n_quadrature_points;

    const std::vector<Tensor<1, dim, value_type>>& normals_in =
      fe_in.get_normal_vectors();

    auto& local_rhs_in = copy_in.vector();
    auto& local_rhs_ex = copy_ex.vector();

    for (unsigned int iqpt = 0; iqpt < nqpt_in; ++iqpt) {
      const value_type normal_velocity =
        normals_in[iqpt] * s_in.get_velocity()[iqpt];

      if (normal_velocity > 0.0) {
        for (unsigned int idof = 0; idof < ndof_in; ++idof)
          local_rhs_in[idof] -= normal_velocity *
                                fe_in.shape_value(idof, iqpt) *
                                s_in.get_soln()[iqpt] * fe_in.JxW(iqpt);
        for (unsigned int idof = 0; idof < ndof_ex; ++idof)
          local_rhs_ex[idof] += normal_velocity *
                                fe_ex.shape_value(idof, iqpt) *
                                s_in.get_soln()[iqpt] * fe_ex.JxW(iqpt);
      }

      else {
        for (unsigned int idof = 0; idof < ndof_in; ++idof)
          local_rhs_in[idof] -= normal_velocity *
                                fe_in.shape_value(idof, iqpt) *
                                s_ex.get_soln()[iqpt] * fe_in.JxW(iqpt);
        for (unsigned int idof = 0; idof < ndof_ex; ++idof)
          local_rhs_ex[idof] += normal_velocity *
                                fe_ex.shape_value(idof, iqpt) *
                                s_ex.get_soln()[iqpt] * fe_ex.JxW(iqpt);
      }
    }
  }


  template <typename VeloFcnType>
  void AdvectAssembler<AssemblyFramework::workstream,
                       VeloFcnType>::boundary_assembly(const ScratchData& s,
                                                       CopyData& copy,
                                                       const bcs_type& bcs)
  {
    using namespace dealii;

    const FEValuesBase<dim>& fe = s.fe_values();
    const auto nqpt = fe.n_quadrature_points;
    const auto ndof = fe.dofs_per_cell;
    const auto& normals = fe.get_normal_vectors();
    auto& local_rhs = copy.vector();


    for (unsigned int iqpt = 0; iqpt < nqpt; ++iqpt) {
      const value_type normal_velocity = normals[iqpt] * s.get_velocity()[iqpt];

      if (normal_velocity > 0.0) {
        // outflow bdry //
        for (unsigned int idof = 0; idof < ndof; ++idof)
          local_rhs[idof] -= normal_velocity * fe.shape_value(idof, iqpt) *
                             s.get_soln()[iqpt] * fe.JxW(iqpt);
      } else {
        // inflow bdry //
        bool use_zero_neumann_bc = true;
        const auto bid = s.face().boundary_id();

        if (bid > 0 && bcs.has_boundary_id(bid)) {
          for (const auto& pbc : bcs(bid)) {
            if (pbc->get_category() == BCCategory::dirichlet) {
              auto bdry_val = pbc->value(fe.quadrature_point(iqpt));
              for (unsigned int idof = 0; idof < ndof; ++idof)
                local_rhs[idof] -= normal_velocity *
                                   fe.shape_value(idof, iqpt) * bdry_val *
                                   fe.JxW(iqpt);
              use_zero_neumann_bc = false;
            }
          }  // loop thru bc at this id
        }

        if (use_zero_neumann_bc)
          // default zero-gradient boundary conditions
          for (unsigned int idof = 0; idof < ndof; ++idof)
            local_rhs[idof] -= normal_velocity * fe.shape_value(idof, iqpt) *
                               s.get_soln()[iqpt] * fe.JxW(iqpt);
      }  // normal_velocity <= 0.0
    }    // iqpt-loop
  }


  /* ************************************************** */
  /**
   * Local results for each and every cell assembly entity.
   */
  /* ************************************************** */
  template <typename VeloFcnType>
  class AdvectAssembler<AssemblyFramework::workstream, VeloFcnType>::ScratchData
    : public CellScratchData<dim>
  {
    /**
     * Allow access from the assembler
     */
    friend class AdvectAssembler<AssemblyFramework::workstream, VeloFcnType>;

   public:
    using base_type = CellScratchData<dim>;
    using typename base_type::synced_iterators_type;

    /**
     * Constructor.
     */
    ScratchData(FEValuesEnum fevalenum,
                const std::weak_ptr<const VeloFcnType>& ptr_velocity_field_,
                const vector_type& vector,
                const std::shared_ptr<const source_term_type>& p_source)
      : base_type(fevalenum),
        ptr_velocity_field(ptr_velocity_field_),
        ptr_soln(&vector),
        ptr_source_term(p_source)
    {}


    /**
     * Copy Constructor
     */
    ScratchData(const ScratchData&) = default;


    /**
     * Initialize the cache vectors to appropriate size
     * defined by (pointers to)  \c FEValuesBase -family of objects.
     */
    void allocate()
    {
      if constexpr (is_simulator<VeloFcnType>::value) {
        ASSERT(this->ptrs_feval.size() == 2, ExcInternalErr());
        ASSERT(this->ptrs_feval[0]->n_quadrature_points ==
                 this->ptrs_feval[1]->n_quadrature_points,
               ExcInternalErr());
      }

      const auto nqpt = this->ptrs_feval[0]->n_quadrature_points;
      soln_at_qpt.resize(nqpt);
      source_at_qpt.resize(nqpt);
      velocity_at_qpt.resize(nqpt);
    }


    /**
     * \name Reinit functions.
     * These functions will reinit \c FEValuesBase -family objects
     * and also populate the cache vectors
     */
    //@{
    /** Reinit as cell data */
    void reinit(const synced_iterators_type& cell)
    {
      base_type::reinit(cell);
      compute_source_and_soln();
      compute_velocity();
    }

    /** Reinit as face data */
    void reinit(const synced_iterators_type& cell, const unsigned int face_no)
    {
      base_type::reinit(cell, face_no);
      compute_source_and_soln();
      compute_velocity();
    }

    /** Reinit as subface data */
    void reinit(const synced_iterators_type& cell, unsigned int face_no,
                unsigned int subface_no)
    {
      base_type::reinit(cell, face_no, subface_no);
      compute_source_and_soln();
      compute_velocity();
    }
    //@}


    /**
     * \name Getting cached vectors
     */
    //@{
    /**
     * Get solution vector.
     */
    const std::vector<value_type>& get_soln() const { return soln_at_qpt; }

    /**
     * Get source vector.
     */
    const std::vector<value_type>& get_source() const { return source_at_qpt; }

    /**
     * Get velocity vector
     */
    const std::vector<dealii::Tensor<1, dim, value_type>>& get_velocity() const
    {
      return velocity_at_qpt;
    }
    //@}


   protected:
    /**
     * Compute source terms values and
     * solution values at quadrature points
     */
    void compute_source_and_soln()
    {
      using dealii::Point;

      // interpolate solution to quadrature points
      this->ptrs_feval[0]->get_function_values(*ptr_soln, soln_at_qpt);

      // compute source term values
      const std::vector<Point<dim>>& qpts =
        this->ptrs_feval[0]->get_quadrature_points();
      ASSERT_SAME_SIZE(qpts, source_at_qpt);

      if (ptr_source_term)
        std::transform(qpts.cbegin(), qpts.cend(), source_at_qpt.begin(),
                       [this](const dealii::Point<dim, value_type>& pt) {
                         return (*this->ptr_source_term)(pt);
                       });
      else
        std::fill(source_at_qpt.begin(), source_at_qpt.end(), 0.0);
    }


    /**
     * Fill the velocity vector by values provided in the external simulator.
     * The veclocity field is computed by the \c VelocityExtractor class
     * specialized to the velocity field function.
     */
    void compute_velocity()
    {
      auto sp_velofield = ptr_velocity_field.lock();
      ASSERT(sp_velofield != nullptr, ExcExpiredPointer());

      if constexpr (is_simulator<VeloFcnType>::value)
        extract_cell_velocities(*sp_velofield, *this->ptrs_feval[1],
                                velocity_at_qpt);
      else
        extract_cell_velocities(*sp_velofield, *this->ptrs_feval[0],
                                velocity_at_qpt);
    }


    /** \name Pointers to objects to be cached */
    //@{
    /**
     * Pointer to velocity field
     */
    const std::weak_ptr<const VeloFcnType> ptr_velocity_field;

    /**
     * Const pointer to const solution vector
     */
    const vector_type* const ptr_soln;

    /**
     * Pointer to source term function
     */
    const std::shared_ptr<const source_term_type> ptr_source_term;
    //@}


    /** \name Cache vectors */
    //@{
    /**
     * Value of the solution of the previous time step on the quadrature point.
     */
    std::vector<value_type> soln_at_qpt;

    /**
     * Value of the source term at quadrature point.
     */
    std::vector<value_type> source_at_qpt;

    /**
     * Value of velocities at quadrature point.
     */
    std::vector<dealii::Tensor<1, dim, value_type>> velocity_at_qpt;
    //@}
  };


  /* ************************************************** */
  /**
   * Collection of scratch data on cell,
   * interior/exterior faces and subface.
   */
  /* ************************************************** */
  template <typename VeloFcnType>
  struct AdvectAssembler<AssemblyFramework::workstream,
                         VeloFcnType>::ScratchDataBox
    : public CellScratchDataBox<ScratchData>
  {
    using simulator_type = VeloFcnType;
    using value_type = typename VeloFcnType::value_type;
    using base_type = CellScratchDataBox<ScratchData>;

    /**
     * Constructor
     */
    template <template <int> class QuadratureType>
    ScratchDataBox(
      const dealii::Mapping<dim>& mapping,
      const dealii::FiniteElement<dim>& finite_element,
      const QuadratureType<dim>& quad,
      const dealii::UpdateFlags update_flags,
      const std::weak_ptr<const simulator_type>& ptr_velocity_field_,
      const vector_type& soln,
      const std::shared_ptr<const source_term_type>& psource)
      : CellScratchDataBox<ScratchData>(ptr_velocity_field_, soln, psource)
    {
      this->add(mapping, finite_element, quad, update_flags);

      // if we have a simulator, we need its FEValues
      if constexpr (is_simulator<VeloFcnType>::value) {
        auto sp_velo = ptr_velocity_field_.lock();
        ASSERT(sp_velo != nullptr, ExcExpiredPointer());
        this->add(sp_velo->get_mapping(), sp_velo->get_fe(), quad,
                  update_flags);
      }

      this->cell.allocate();
      this->face_ex.allocate();
      this->face_in.allocate();
      this->subface.allocate();
    }
  };


  /* ************************************************** */
  /**
   * This class contains the local RHS vector
   * and the corresponding dof indices.
   * Used as data member in \c CopyDataBox.
   * Used for assembling the local vector into the global vector.
   */
  /* ************************************************** */
  template <typename VeloFcnType>
  class AdvectAssembler<AssemblyFramework::workstream, VeloFcnType>::CopyData
  {
   public:
    using size_type = types::DoFIndex;
    using CellIterator = typename dealii::DoFHandler<dim>::active_cell_iterator;


    /**
     * Default constructor. Construct with empty vectors.
     */
    CopyData() = default;


    /**
     * Allocate memory for all data members and then
     * \c reset the state of the object.
     */
    void allocate(size_type n)
    {
      dof_indices.resize(n);
      local_vector.reinit(n);
      reset(true);
    }


    /**
     * Resetting local vector to zero, local dof indices to invalid
     * and active flag to \c false.
     */
    void reset(bool force = false)
    {
      if (active || force) {
        local_vector = 0.0;
        std::fill(dof_indices.begin(), dof_indices.end(),
                  constants::invalid_unsigned_int);
        active = false;
      }
    }


    /**
     * Reinit with cell iterator.
     * Fill the dof_indices with that of the current cell,
     * Setting local vector to zero and set active flag to \c true.
     */
    void reinit(const CellIterator& cell_iter)
    {
      ASSERT(
        dof_indices.size() == cell_iter->get_fe().dofs_per_cell,
        ExcSizeMismatch(dof_indices.size(), cell_iter->get_fe().dofs_per_cell));
      ASSERT(
        dof_indices.size() == cell_iter->get_fe().dofs_per_cell,
        ExcSizeMismatch(dof_indices.size(), cell_iter->get_fe().dofs_per_cell));
      cell_iter->get_dof_indices(dof_indices);
      local_vector = 0.0;
      active = true;
    }


    /**
     * Provide interface for accessing and modifying the vector.
     */
    dealii::Vector<value_type>& vector() { return local_vector; }


    /**
     * Return \c active flag of the current object.
     */
    bool is_active() const { return active; }


    /**
     * When set to be active,
     * this object will participate in assembly if called.
     */
    void set_active()
    {
      ASSERT_SAME_SIZE(dof_indices, local_vector);
      active = true;
    }


    /**
     * Assemble the local vector into the global vector.
     */
    template <typename ConstraintsType>
    void assemble(const ConstraintsType& constraints,
                  vector_type& dst_vector) const
    {
      if (active) {
#ifdef DEBUG
        // make sure that the data is suitable for copying to global
        bool no_invalid_dof_idx = std::accumulate(
          dof_indices.cbegin(), dof_indices.cend(), true,
          [](bool init, const size_t dof) {
            return init && dof != constants::invalid_unsigned_int;
          });
        ASSERT(no_invalid_dof_idx,
               EXCEPT_MSG("Some dof indices are not initialized."));
        ASSERT(!dof_indices.empty(),
               EXCEPT_MSG("DoF Indices vector is empty."));
        ASSERT_SAME_SIZE(dof_indices, local_vector);
#endif  // DEBUG

        constraints.distribute_local_to_global(local_vector, dof_indices,
                                               dst_vector);
      }
    }


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
   * Collection of copy data on each face (exterior and interior)
   * as well as the cell itself.
   */
  /* ************************************************** */
  template <typename VeloFcnType>
  struct AdvectAssembler<AssemblyFramework::workstream,
                         VeloFcnType>::CopyDataBox
  {
    using size_type = types::DoFIndex;
    using CellIterator = typename dealii::DoFHandler<dim>::active_cell_iterator;
    constexpr static int faces_per_cell =
      dealii::GeometryInfo<dim>::faces_per_cell;

    /**
     * Constructor. Allocate space for all copy data.
     */
    CopyDataBox(size_t n)
    {
      cell.allocate(n);
      for (auto& f : interior_faces) { f.allocate(n); }
      for (auto& f : exterior_faces) { f.allocate(n); }
    }


    /**
     * Reset all copy data to initial state.
     */
    void reset()
    {
      cell.reset();
      for (auto& f : interior_faces) { f.reset(); }
      for (auto& f : exterior_faces) { f.reset(); }
    }


    /**
     * Assemble the copy data from local vector to global vector.
     */
    template <typename ConstraintsType>
    void assemble(const ConstraintsType& constraints,
                  vector_type& rhs_vector) const
    {
      cell.assemble(constraints, rhs_vector);
      for (auto& face : interior_faces) face.assemble(constraints, rhs_vector);
      for (auto& face : exterior_faces) face.assemble(constraints, rhs_vector);
    }


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

}  // namespace dg

FELSPA_NAMESPACE_CLOSE

#endif  // _FELSPA_PDE_ADVECTION_IMPLEMENTATION_H_ //
