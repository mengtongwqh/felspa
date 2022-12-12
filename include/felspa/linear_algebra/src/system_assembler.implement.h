#ifndef _FELSPA_LINEAR_SYSTEM_SYSTEM_ASSEMBLER_IMPLEMENT_H_
#define _FELSPA_LINEAR_SYSTEM_SYSTEM_ASSEMBLER_IMPLEMENT_H_

#include <deal.II/fe/mapping_q.h>
#include <felspa/base/constants.h>
#include <felspa/linear_algebra/system_assembler.h>

FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/*                  class AssemblerBase               */
/* ************************************************** */
template <typename LinsysType>
AssemblerBase<LinsysType>::AssemblerBase(LinsysType& linsys,
                                         bool construct_mapping_adhoc)
  : ptr_linear_system(&linsys), ptr_mapping_adhoc(nullptr)
{
  if (construct_mapping_adhoc && linsys.ptr_mapping == nullptr) {
    // construct mapping if it is not there
    ASSERT(dof_handler().has_active_dofs(), ExcDoFHandlerNotInit());
    const unsigned int fe_degree = dof_handler().get_fe().degree;
    ptr_mapping_adhoc = std::make_shared<dealii::MappingQ<dim>>(fe_degree);
  }
}


template <typename LinsysType>
auto AssemblerBase<LinsysType>::get_mapping() const
  -> const dealii::Mapping<dim>&
{
  if (ptr_linear_system->ptr_mapping)
    return *(ptr_linear_system->ptr_mapping);
  else if (this->ptr_mapping_adhoc)
    return *ptr_mapping_adhoc;

  // nullptr in both linear system and
  // system assembler for mapping. We have an error.
  THROW(ExcInternalErr());
}

/* ************************************************** */
/*            class FEFunctionSelector                */
/* ************************************************** */
template <int dim, int spacedim, typename NumberType>
struct FEFunctionSelector<dim, spacedim, NumberType>::FunctionTypeCounter
{
  /** Counter for FE function value index */
  types::SizeType values = 0;

  /** Counter for FE function gradient index */
  types::SizeType gradients = 0;

  /** Counter for FE function hessian index */
  types::SizeType hessians = 0;

  /** Default constructor */
  FunctionTypeCounter() = default;

  /** Zeroing out all cumulative counters */
  void reset()
  {
    values = 0;
    gradients = 0;
    hessians = 0;
  }
};


template <int dim, int spacedim, typename NumberType>
template <typename VectorType>
void FEFunctionSelector<dim, spacedim, NumberType>::add(
  const std::string& label, const VectorType& data,
  const std::array<bool, 3>& cell_selector,
  const std::array<bool, 3>& face_selector,
  const std::array<bool, 3>& boundary_selector)
{
  // Only add if the vector label is unique
  ASSERT(fe_fcn_data.try_find(label) == constants::invalid_unsigned_int,
         ExcEntryAlreadyExists(label));

  fe_fcn_data.add<const VectorType*>(&data, label);

  // inform info_box of update flags
  info_box().cell_selector.add(label, cell_selector[0], cell_selector[1],
                               cell_selector[2]);
  info_box().face_selector.add(label, face_selector[0], face_selector[1],
                               face_selector[2]);
  info_box().boundary_selector.add(label, boundary_selector[0],
                                   boundary_selector[1], boundary_selector[2]);

  // tediously update index map
  assign_selector_flags(label, AssemblyWorker::cell, cell_selector);
  assign_selector_flags(label, AssemblyWorker::face, face_selector);
  assign_selector_flags(label, AssemblyWorker::boundary, boundary_selector);
}


template <int dim, int spacedim, typename NumberType>
template <typename VectorType>
void FEFunctionSelector<dim, spacedim, NumberType>::finalize(
  const dealii::FiniteElement<dim, spacedim>& fe,
  const dealii::Mapping<dim, spacedim>& mapping,
  const dealii::BlockInfo* block_info, const VectorType& dummy) const
{
  // initialize Gauss points for cell/face/boundary
  const unsigned int n_gauss_pts = fe.degree + 1;
  info_box().initialize_gauss_quadrature(n_gauss_pts, n_gauss_pts, n_gauss_pts);

  // set appropriate update flags
  info_box().initialize_update_flags();
  info_box().add_update_flags_all(dealii::update_quadrature_points);

  // attach fe_fcn_data to info_box
  info_box().initialize(fe, mapping, fe_fcn_data, dummy, block_info);
}


/* ************************************************** */
/*            class MassMatrixAssembler               */
/* ************************************************** */
template <typename LinsysType>
void MassMatrixAssemblerBase<LinsysType>::assembly_kernel(
  const dealii::MeshWorker::LocalIntegrator<dim, dim, value_type>&
    local_integrator)
{
  using namespace dealii;

  UpdateFlags update_flags = update_values | update_JxW_values;

  MeshWorker::IntegrationInfoBox<dim> info_box;
  info_box.add_update_flags_all(update_flags);
  const unsigned int n_gauss_pts = this->dof_handler().get_fe().degree + 1;
  info_box.initialize_gauss_quadrature(n_gauss_pts, n_gauss_pts, n_gauss_pts);
  info_box.initialize(this->dof_handler().get_fe(), this->get_mapping());
                      // &this->dof_handler().block_info());
  dof_info_t dof_info(this->dof_handler());

  MeshWorker::Assembler::MatrixSimple<typename LinsysType::matrix_type>
    assembler;
  assembler.initialize(this->matrix());
  MeshWorker::integration_loop<dim, dim>(this->dof_handler().begin_active(),
                                         this->dof_handler().end(), dof_info,
                                         info_box, local_integrator, assembler);
}


/* -------------------------- */
/* class MassMatrixIntegrator */
/* -------------------------- */
// Specialization for LinearSystem //
template <int dim, typename NumberType>
class MassMatrixAssembler<dim, NumberType, LinearSystem>::MassMatrixIntegrator
  : public dealii::MeshWorker::LocalIntegrator<dim, dim, NumberType>
{
  using parent_type = MassMatrixAssembler<dim, NumberType, LinearSystem>;

 public:
  using value_type = typename parent_type::value_type;
  using integration_info_t = typename parent_type::integration_info_t;
  using dof_info_t = typename parent_type::dof_info_t;
  using local_matrix_t = typename parent_type::local_matrix_t;

  /** Constructor */
  MassMatrixIntegrator()
    : dealii::MeshWorker::LocalIntegrator<dim, dim, value_type>(true, false,
                                                                false)
  {}

  /** Cell assembler */
  virtual void cell(dof_info_t& dinfo, integration_info_t& cinfo) const override
  {
    using namespace dealii;
    const FEValuesBase<dim>& fe = cinfo.fe_values();
    local_matrix_t& local_mass = dinfo.matrix(0).matrix;
    const std::vector<double>& JxW = fe.get_JxW_values();

    for (unsigned int iq = 0; iq < fe.n_quadrature_points; ++iq)
      for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
        for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
          local_mass(i, j) +=
            fe.shape_value(i, iq) * fe.shape_value(j, iq) * JxW[iq];
  }
};


// Specialization for BlockLinearSystem //
template <int dim, typename NumberType>
class MassMatrixAssembler<dim, NumberType,
                          BlockLinearSystem>::MassMatrixIntegrator
  : public dealii::MeshWorker::LocalIntegrator<dim, dim, value_type>
{
  using parent_type = MassMatrixAssembler<dim, NumberType, BlockLinearSystem>;

 public:
  using value_type = typename parent_type::value_type;
  using integration_info_t = typename parent_type::integration_info_t;
  using dof_info_t = typename parent_type::dof_info_t;


  /**
   * Constructor
   */
  MassMatrixIntegrator()
    : dealii::MeshWorker::LocalIntegrator<dim, dim, value_type>(true, false,
                                                                false)
  {}


  /**
   * Cell assembler
   */
  virtual void cell(dof_info_t& dinfo, integration_info_t& cinfo) const override
  {
    using namespace dealii;

    const FEValuesBase<dim>& fe = cinfo.fe_values();
    const auto nqpt = fe.n_quadrature_points;
    const auto ndof = fe.dofs_per_cell;
    auto& local_mass = dinfo.matrix(0).matrix;
    const std::vector<double>& JxW = fe.get_JxW_values();

    using nqpt_type = typename std::remove_const<decltype(ndof)>::type;
    using ndof_type = typename std::remove_const<decltype(nqpt)>::type;

#if 0
    const FEValuesExtractors::Vector psi(0);
    for (nqpt_type iq = 0; iq < nqpt; ++iq) {
      for (ndof_type i = 0; i < ndof; ++i) {
        Tensor<1, dim, value_type> psi_i = fe[psi].value(i, iq);
        for (ndof_type j = 0; j < ndof; ++j) {
          Tensor<1, dim, value_type> psi_j = fe[psi].value(j, iq);
          /// \todo: This is problematic....
          local_mass(i, j) += psi_i * psi_j * JxW[iq];
        }  // j-loop
      }    // i-loop
    }      // iq-loop

#else
    for (ndof_type i = 0; i < ndof; ++i) {
      const value_type* psi_i = &fe.shape_value(i, 0);
      const unsigned int component_i =
        fe.get_fe().system_to_component_index(i).first;

      for (ndof_type j = 0; j < ndof; ++j) {
        const value_type* psi_j = &fe.shape_value(j, 0);
        const unsigned int component_j =
          fe.get_fe().system_to_component_index(j).first;

        if (component_i == component_j)
          for (nqpt_type iq = 0; iq < nqpt; ++iq)
            local_mass(i, j) += psi_i[iq] * psi_j[iq] * JxW[iq];
      }  // j-loop
    }    // i-loop
#endif
  }
};


FELSPA_NAMESPACE_CLOSE
#endif /* _FELSPA_LINEAR_SYSTEM_SYSTEM_ASSEMBLER_IMPLEMENT_H_ */
