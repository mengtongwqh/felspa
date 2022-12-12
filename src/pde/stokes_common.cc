#include <felspa/pde/stokes_common.h>

FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/*                StokesControlBase                   */
/* ************************************************** */
template <typename NumberType>
StokesControlBase<NumberType>::StokesControlBase()
  : SimulatorControlBase<NumberType>()
{
  this->ptr_solver = std::make_shared<StokesSolverControl>();
}


template <typename NumberType>
void StokesControlBase<NumberType>::write_solver_statistics(
  const std::string& filename)
{
  auto& ctrl = static_cast<StokesSolverControl&>(*this->ptr_solver);
  ctrl.write_statistics(filename);
}

template <typename NumberType>
void StokesControlBase<NumberType>::set_material_parameters_to_export(
  const std::set<MaterialParameter>& parameters)
{
  material_parameters_to_export = parameters;
}


/* ************************************************** */
/*                StokesSolutionMethod                */
/* ************************************************** */
std::string to_string(StokesSolutionMethod solution_method)
{
  switch (solution_method) {
    case FC:
      return "FC";
    case SCR:
      return "SCR";
    case CompareTest:
      return "Compare";
    default:
      THROW(ExcInternalErr());
  }
}


/* ************************************************** */
/*                StokesSolverControl                 */
/* ************************************************** */
void StokesSolverControl::write_statistics(const std::string& filename)
{
  ExportFile file(filename + ".csv");
  std::ofstream& ofs = file.access_stream();

  // iterators
  auto it_cg_error = cg_error.begin();
  auto it_cg_timer = cg_timer.begin();
  auto it_n_cg_inner_iter = n_cg_inner_iter.begin();
  auto it_n_cg_outer_iter = n_cg_outer_iter.begin();

  auto it_gmres_error = gmres_error.begin();
  auto it_gmres_timer = gmres_timer.begin();
  auto it_n_gmres_iter = n_gmres_iter.begin();

  auto it_soln_diff_l2 = soln_diff_l2.begin();
  auto it_soln_diff_linfty = soln_diff_linfty.begin();

  auto N = std::max(cg_error.size(), gmres_error.size());

  // headers
  if (log_cg) ofs << "cg_error,cg_timer,n_cg_inner_iter,n_cg_outer_iter,";
  if (log_gmres) ofs << "gmres_error,gmres_timer,n_gmres_iter,";
  if (log_cg && log_gmres) ofs << "soln_diff_l2,soln_diff_linfty,";
  ofs << '\n';

  for (decltype(N) i = 0; i < N; ++i) {
    if (log_cg)
      ofs << *it_cg_error++ << ',' << *it_cg_timer++ << ','
          << *it_n_cg_inner_iter++ << ',' << *it_n_cg_outer_iter++ << ',';
    if (log_gmres)
      ofs << *it_gmres_error++ << ',' << *it_gmres_timer++ << ','
          << *it_n_gmres_iter++ << ',';
    if (log_cg && log_gmres)
      ofs << *it_soln_diff_l2++ << ',' << *it_soln_diff_linfty++ << ',';
    ofs << '\n';
  }
}


/* ------------------- */
namespace internal
/* ------------------- */
{
  /* ************************************************** */
  /*               StokesAssemblyScratch                */
  /* ************************************************** */
  template <int dim, typename NumberType>
  StokesAssemblyScratch<dim, NumberType>::StokesAssemblyScratch(
    const dealii::FiniteElement<dim>& fe, const dealii::Mapping<dim>& mapping,
    const dealii::Quadrature<dim>& quadrature,
    const dealii::UpdateFlags update_flags,
    const std::shared_ptr<const MaterialBase<dim, value_type>>& p_material)
    : fe_values(mapping, fe, quadrature, update_flags),
      ptr_material_accessor(p_material->generate_accessor(quadrature)),
      pts_field(fe_values.n_quadrature_points),
      sym_grad_v(fe.dofs_per_cell),
      div_v(fe.dofs_per_cell),
      p(fe.dofs_per_cell),
      v(fe.dofs_per_cell),
      grad_v(fe.dofs_per_cell),
      source(fe_values.n_quadrature_points),
      viscosity(fe_values.n_quadrature_points),
      density(fe_values.n_quadrature_points)
  {}


  template <int dim, typename NumberType>
  StokesAssemblyScratch<dim, NumberType>::StokesAssemblyScratch(
    const StokesAssemblyScratch<dim, NumberType>& that)
    : fe_values(that.fe_values.get_mapping(), that.fe_values.get_fe(),
                that.fe_values.get_quadrature(),
                that.fe_values.get_update_flags()),
      ptr_material_accessor(
        that.ptr_material_accessor->get_material().generate_accessor(
          that.fe_values.get_quadrature())),
      pts_field(that.pts_field),
      sym_grad_v(that.sym_grad_v),
      div_v(that.div_v),
      p(that.p),
      v(that.v),
      grad_v(that.grad_v),
      source(that.source),
      viscosity(that.viscosity),
      density(that.density)
  {}


  template <int dim, typename NumberType>
  void StokesAssemblyScratch<dim, NumberType>::reinit(
    const cell_iterator_type& cell)
  {
    fe_values.reinit(cell);
    ptr_material_accessor->reinit(cell);
    pts_field.ptr_pts = &fe_values.get_quadrature_points();
  }


  /* ************************************************** */
  /*                 StokesAssemblyCopy                 */
  /* ************************************************** */
  template <int dim, typename NumberType>
  StokesAssemblyCopy<dim, NumberType>::StokesAssemblyCopy(
    const dealii::FiniteElement<dim>& fe)
    : local_dof_indices(fe.dofs_per_cell),
      local_matrix(fe.dofs_per_cell, fe.dofs_per_cell),
      local_preconditioner(fe.dofs_per_cell, fe.dofs_per_cell),
      local_rhs(fe.dofs_per_cell)
  {}


  template <int dim, typename NumberType>
  StokesAssemblyCopy<dim, NumberType>::StokesAssemblyCopy(
    const StokesAssemblyCopy<dim, NumberType>& that)
    : local_dof_indices(that.local_dof_indices),
      local_matrix(that.local_matrix),
      local_preconditioner(that.local_preconditioner),
      local_rhs(that.local_rhs)
  {}


#if 0
  /* ************************************************** */
  /*             StokesMaterialExporter                 */
  /* ************************************************** */
  template <int dim, typename NumberType>
  StokesMaterialParameterExporter<dim, NumberType>::
    StokesMaterialParameterExporter(
      const StokesSimulator<dim, NumberType>& simulator,
      UpdateMethod update_method_)
    : ptr_simulator(&simulator),
      dof_handler(simulator.mesh()),
      mesh_update_detected(true),
      update_method(update_method_),
      pvd_collector(simulator.get_label_string() + "Matrl")
  {
    ASSERT(!ptr_simulator->get_control().material_parameters_to_export.empty(),
           ExcInternalErr());

    std::vector<bool> component_mask(dim + 1, false);
    component_mask[dim] = true;
    ptr_fe = &ptr_simulator->get_fe().get_sub_fe(component_mask);

    pvd_collector.set_file_path(ptr_simulator->pvd_collector.get_file_path());

    for (auto parameter :
         ptr_simulator->get_control().material_parameters_to_export) {
      if (auto [it, status] = parameter_vectors.insert(
            std::make_pair(parameter, dealii::Vector<value_type>()));
          !status)
        THROW(ExcInternalErr());
    }
  }


  template <int dim, typename NumberType>
  void StokesMaterialParameterExporter<dim, NumberType>::update_parameters(
    const dealii::Quadrature<dim>& quadrature)
  {
    switch (update_method) {
      case projection:
        update_by_projection(quadrature);
        break;
      case cell_mean:
        update_by_cell_mean(quadrature);
        break;
      default:
        THROW(ExcInternalErr());
    }
  }


  template <int dim, typename NumberType>
  void StokesMaterialParameterExporter<dim, NumberType>::update_by_cell_mean(
    const dealii::Quadrature<dim>& quadrature)
  {
    if (mesh_update_detected) dof_handler.distribute_dofs(*ptr_fe);

    using this_type = StokesMaterialParameterExporter<dim, NumberType>;

    const auto n_cells = ptr_simulator->mesh().n_active_cells();
    auto pmaterial = ptr_simulator->ptr_material.lock();
    ASSERT(pmaterial != nullptr, ExcExpiredPointer());

    ScratchData scratch(
      *ptr_fe, ptr_simulator->get_mapping(), quadrature,
      dealii::update_quadrature_points | dealii::update_JxW_values, pmaterial);
    CopyData copy(1);

    for (auto& it : parameter_vectors) {
      current_parameter = it.first;
      it.second.reinit(n_cells);
      dealii::WorkStream::run(dof_handler.begin_active(), dof_handler.end(),
                              *this, &this_type::local_assembly,
                              &this_type::copy_local_to_global, scratch, copy);
    }
  }


  template <int dim, typename NumberType>
  void StokesMaterialParameterExporter<dim, NumberType>::update_by_projection(
    const dealii::Quadrature<dim>& quadrature)
  {
    if (mesh_update_detected) {
      dof_handler.distribute_dofs(*ptr_fe);
      constraints.clear();
      dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      constraints.close();

      const auto n_dofs = dof_handler.n_dofs();
      sparsity.reinit(n_dofs, n_dofs, dof_handler.max_couplings_between_dofs());
      dealii::DoFTools::make_sparsity_pattern(dof_handler, sparsity,
                                              constraints, false);
      sparsity.compress();
      mass_matrix.reinit(sparsity);

      dealii::MatrixCreator::create_mass_matrix(
        ptr_simulator->get_mapping(), dof_handler, quadrature, mass_matrix,
        static_cast<const dealii::Function<dim>* const>(nullptr), constraints);

      rhs.reinit(n_dofs);
      mesh_update_detected = false;
    }

    for (auto& [param, vector] : parameter_vectors)
      vector.reinit(dof_handler.n_dofs());

    auto pmaterial = ptr_simulator->ptr_material.lock();
    ASSERT(pmaterial != nullptr, ExcExpiredPointer());

    ScratchData scratch(*ptr_fe, ptr_simulator->get_mapping(), quadrature,
                        dealii::update_quadrature_points |
                          dealii::update_values | dealii::update_JxW_values,
                        pmaterial);
    CopyData copy(ptr_fe->n_dofs_per_cell());

    dealii::PreconditionJacobi<dealii::SparseMatrix<value_type>> precond_jacobi;
    precond_jacobi.initialize(mass_matrix);

    for (auto& it : parameter_vectors) {
      rhs = 0.0;
      current_parameter = it.first;
      dealii::WorkStream::run(
        dof_handler.begin_active(), dof_handler.end(), *this,
        &StokesMaterialParameterExporter<dim, NumberType>::local_assembly,
        &StokesMaterialParameterExporter<dim, NumberType>::copy_local_to_global,
        scratch, copy);

      // solve for the projection
      dealii::SolverControl control;
      dealii::SolverCG<> cg_solver(control);
      cg_solver.solve(mass_matrix, it.second, rhs, precond_jacobi);
    }
  }


  template <int dim, typename NumberType>
  void StokesMaterialParameterExporter<dim, NumberType>::local_assembly(
    const typename dealii::DoFHandler<dim>::active_cell_iterator& cell,
    ScratchData& scratch, CopyData& copy)
  {
    scratch.reinit(cell);
    scratch.ptr_material_accessor->eval_scalars(
      current_parameter, scratch.pts_field, scratch.scalar_parameter);

    copy.local_rhs = 0.0;

    if (update_method == projection) {
      cell->get_dof_indices(copy.local_dof_indices);

      for (unsigned int iq = 0; iq < scratch.fe_values.n_quadrature_points;
           ++iq)
        for (unsigned int idof = 0; idof < scratch.fe_values.dofs_per_cell;
             ++idof)
          copy.local_rhs[idof] += scratch.fe_values.shape_value(idof, iq) *
                                  scratch.scalar_parameter[iq] *
                                  scratch.fe_values.JxW(iq);
    } else if (update_method == cell_mean) {
      copy.local_dof_indices[0] = cell->active_cell_index();
      for (unsigned int iq = 0; iq < scratch.fe_values.n_quadrature_points;
           ++iq)
        copy.local_rhs[0] +=
          scratch.scalar_parameter[iq] * scratch.fe_values.JxW(iq);
      copy.local_rhs[0] /= cell->measure();
    } else {
      THROW(ExcInternalErr());
    }
  }


  template <int dim, typename NumberType>
  void StokesMaterialParameterExporter<dim, NumberType>::copy_local_to_global(
    const CopyData& copy)
  {
    // assemble into the global vector
    if (update_method == projection)
      constraints.distribute_local_to_global(copy.local_rhs,
                                             copy.local_dof_indices, rhs);
    else if (update_method == cell_mean)
      parameter_vectors[current_parameter][copy.local_dof_indices[0]] =
        copy.local_rhs[0];
    else
      THROW(ExcInternalErr());
  }


  template <int dim, typename NumberType>
  void StokesMaterialParameterExporter<dim, NumberType>::export_parameters()
    const
  {
    using namespace dealii;
    using dealii::Utilities::int_to_string;
    const auto counter = pvd_collector.get_file_count() + 1;
    std::string master_file_name =
      pvd_collector.get_file_name() + '_' +
      int_to_string(counter, constants::max_export_numeric_digits);

    ExportFile export_file(pvd_collector.get_file_path() + master_file_name +
                           ".vtu");

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    for (const auto& it : parameter_vectors) {
      std::stringstream ss;
      ss << it.first;
      data_out.add_data_vector(it.second, ss.str());
    }
    data_out.build_patches();
    data_out.write_vtu(export_file.access_stream());

    pvd_collector.append_record(ptr_simulator->get_time(),
                                master_file_name + ".vtu");
  }


  /*
   * StokesMaterialParameterExporter::ScratchData
   */
  template <int dim, typename NumberType>
  StokesMaterialParameterExporter<dim, NumberType>::ScratchData::ScratchData(
    const dealii::FiniteElement<dim>& fe, const dealii::Mapping<dim>& mapping,
    const dealii::Quadrature<dim>& quadrature,
    const dealii::UpdateFlags update_flags,
    const std::shared_ptr<const MaterialBase<dim, value_type>>& p_material)
    : fe_values(mapping, fe, quadrature, update_flags),
      ptr_material_accessor(p_material->generate_accessor(quadrature)),
      pts_field(fe_values.n_quadrature_points),
      scalar_parameter(fe_values.n_quadrature_points)
  {}


  template <int dim, typename NumberType>
  StokesMaterialParameterExporter<dim, NumberType>::ScratchData::ScratchData(
    const ScratchData& that)
    : fe_values(that.fe_values.get_mapping(), that.fe_values.get_fe(),
                that.fe_values.get_quadrature(),
                that.fe_values.get_update_flags()),
      ptr_material_accessor(
        that.ptr_material_accessor->get_material().generate_accessor(
          that.fe_values.get_quadrature())),
      pts_field(that.pts_field),
      scalar_parameter(that.scalar_parameter)
  {}


  template <int dim, typename NumberType>
  void StokesMaterialParameterExporter<dim, NumberType>::ScratchData::reinit(
    const cell_iterator_type& cell)
  {
    fe_values.reinit(cell);
    ptr_material_accessor->reinit(cell);
    pts_field.ptr_pts = &fe_values.get_quadrature_points();
  }


  /**
   * StokesMaterialParameterExporter::CopyData
   */
  template <int dim, typename NumberType>
  StokesMaterialParameterExporter<dim, NumberType>::CopyData::CopyData(
    size_type n_dofs)
    : local_dof_indices(n_dofs), local_rhs(n_dofs)
  {}
#endif
}  // namespace internal


/* -------------------------------------------------- */
template class StokesControlBase<types::DoubleType>;

template class internal::StokesAssemblyScratch<2, types::DoubleType>;
template class internal::StokesAssemblyScratch<3, types::DoubleType>;
template class internal::StokesAssemblyCopy<2, types::DoubleType>;
template class internal::StokesAssemblyCopy<3, types::DoubleType>;
/* -------------------------------------------------- */

FELSPA_NAMESPACE_CLOSE