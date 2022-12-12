#ifndef _FELSPA_LINEAR_ALGEBRA_SYSTEM_ASSEMBLER_H_
#define _FELSPA_LINEAR_ALGEBRA_SYSTEM_ASSEMBLER_H_

#include <deal.II/dofs/block_info.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/local_integrator.h>
#include <deal.II/meshworker/loop.h>
#include <felspa/linear_algebra/linear_system.h>

#include <array>
#include <map>
#include <string>
#include <vector>

FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/**
 * Type of integrator function used in \c MeshWorker framework.
 */
/* ************************************************** */
enum class AssemblyWorker
{
  cell,
  face,
  boundary
};


/* ************************************************** */
/**
 * Type of integration framework.
 */
/* ************************************************** */
enum class AssemblyFramework
{
  serial,
  workstream,
  meshworker
};

std::ostream& operator<<(std::ostream& os, AssemblyFramework framework);


/* ************************************************** */
/**
 * An abstract type to construct an assembler object
 * for a linear system of spatial dimension \c dim and
 * linear system type \c LinsysType
 */
/* ************************************************** */
template <typename LinsysType>
class AssemblerBase : public dealii::Subscriptor
{
 public:
  constexpr static int dim = LinsysType::spacedim;
  using linsys_type = LinsysType;
  using matrix_type = typename LinsysType::matrix_type;
  using vector_type = typename LinsysType::vector_type;
  using constraints_type = typename LinsysType::constraints_type;
  const LinsysType& get_linear_system() const { return *ptr_linear_system; }


  /**
   * Getter for the \c dof_handler member
   */
  const dealii::DoFHandler<dim>& get_dof_handler() const
  {
    return dof_handler();
  }


  /**
   * Return true if mapping object is set for the attached linear system.
   */
  bool has_mapping() const { return ptr_linear_system->ptr_mapping; }


 protected:
  /** \name Basic object handling */
  //@{
  /** 
   * Constructor. 
   */
  AssemblerBase(LinsysType& linsys, bool construct_mapping_adhoc);

  /** 
   * Virtual destructor.  
   */
  virtual ~AssemblerBase() = default;
  //@}


  /* ----------------------------------- */
  /** \name Accessing Linear System Info */
  /* ----------------------------------- */
  //@{
  /** 
   * Access the RHS of the linear system
   */
  vector_type& rhs() { return ptr_linear_system->rhs; }

  /** 
   * Same above for const overload 
   */
  const vector_type& rhs() const { return ptr_linear_system->rhs; }

  /** 
   * Access the LHS of the linear system 
   */
  matrix_type& matrix() { return ptr_linear_system->matrix; }

  /** 
   * Same as above for const overload 
   */
  const matrix_type& matrix() const { return ptr_linear_system->matrix; }

  /** 
   * Access constraints object of the linear system 
   */
  constraints_type& constraints() { return ptr_linear_system->constraints; }

  /** 
   * Same as above for const overload 
   */
  const constraints_type& constraints() const
  {
    return ptr_linear_system->constraints;
  }

  /** 
   * Access the \c dof_handler 
   */
  const dealii::DoFHandler<dim>& dof_handler() const
  {
    ASSERT(ptr_linear_system->ptr_dof_handler != nullptr, ExcNullPointer());
    return *(ptr_linear_system->ptr_dof_handler);
  }

  /** 
   * Access the mapping object 
   */
  const dealii::Mapping<dim>& get_mapping() const;
  //@}


  /** Pointer to linear system object */
  dealii::SmartPointer<LinsysType> ptr_linear_system;


  /**
   * Pointer to mapping object.
   * If the linear system does not provide a mapping object,
   * we create an ad-hoc one in the constructor so that the assembler
   * can use it to construct cell \c FEValues object.
   */
  std::shared_ptr<const dealii::Mapping<dim>> ptr_mapping_adhoc;
};


/* ************************************************** */
/**
 * A bridge class between \c dealii::IntegrationInfoBox and
 * \c dealii::IntegrationInfo to simplify retrieving finite element function
 * values/gradient/hessians. \c FEFunctionSelector does this by attaching an
 * identifier string to each finite element function, and then retrieving the
 * function value/gradient/hessian in cell/face/boundary assembler with this
 * identifier string.
 */
/* ************************************************** */
template <int dim, int spacedim = dim, typename NumberType = types::DoubleType>
class FEFunctionSelector
{
 public:
  using value_type = NumberType;
  using intg_info_box_t = dealii::MeshWorker::IntegrationInfoBox<dim>;
  using intg_info_t = dealii::MeshWorker::IntegrationInfo<dim>;


  /**
   * Default constructor
   */
  FEFunctionSelector() : ptr_info_box(nullptr) {}


  /**
   * Constructor taking integration info box reference
   */
  FEFunctionSelector(intg_info_box_t& info_box);


  /**
   * Destructor
   */
  virtual ~FEFunctionSelector() { ptr_info_box = nullptr; }


  /**
   * Update pointer to integration info box
   */
  void attach_info_box(intg_info_box_t& info_box) { ptr_info_box = &info_box; }


  /**
   * Appending this vector to the \c dealii::AnyData.
   */
  template <typename VectorType>
  void add(const std::string& label, const VectorType& soln,
           const std::array<bool, 3>& cell_selector,
           const std::array<bool, 3>& face_selector,
           const std::array<bool, 3>& boundary_selector);


  /**
   * Call this function after all vectors have been appended. The function
   * will inform \c info_box to generate update flags. By default \c
   * update_quadrature_points is automatically added to the update flags.
   */
  template <typename VectorType = dealii::Vector<types::DoubleType>>
  void finalize(
    const dealii::FiniteElement<dim, spacedim>&,
    const dealii::Mapping<dim, spacedim>&,
    const dealii::BlockInfo* block_info = nullptr,
    const VectorType& dummy = dealii::Vector<types::DoubleType>()) const;


  /**
   * Clear all members in the object to the state when it is default
   * initialized. index and reset counter to zero.
   * Pointer to \c info_box is set to \p nullptr.
   */
  void reset();

  /* ----------------------------------------- */
  /** \name Retrieving Finite Element Function */
  /* ----------------------------------------- */
  //@{
  /**
   * Obtain finite element function values for each component at quadrature
   * points.
   */
  std::vector<std::vector<value_type>> values(const std::string& label,
                                              AssemblyWorker worker_type,
                                              const intg_info_t& cinfo) const;

  /**
   * Obtain finite element function gradient for 
   * each component at quadrature points.
   */
  std::vector<std::vector<dealii::Tensor<1, dim, value_type>>> gradients(
    const std::string& label, AssemblyWorker worker_type,
    const intg_info_t& cinfo) const;

  /**
   * Obtain finite element function hessians for each component at quadrature
   * points.
   */
  std::vector<std::vector<dealii::Tensor<2, dim, value_type>>> hessians(
    const std::string& label, AssemblyWorker worker_type,
    const intg_info_t& cinfo) const;
  //@}


  /** 
   * Exception: Finite element function does not exist
   */
  DECL_EXCEPT_2(ExcNotExist,
                "The " << arg1 << " of the finite element function " << arg2
                       << " does not exist",
                std::string, std::string);

  DECL_EXCEPT_1(
    ExcEntryAlreadyExists,
    "The vector data entry ["
      << arg1
      << "] already exists in the fe_fcn_selector. Vectors with the repeating "
         "label string will not be correctly computed during assembly.",
    std::string);


 private:
  /**
   * A counter struct to record the indices of each function type
   * (value/gradient/hessian) cumulatively.
   */
  struct FunctionTypeCounter;


  /**
   * Getter for \c info_box
   */
  intg_info_box_t& info_box() const { return *ptr_info_box; }


  void assign_selector_flags(const std::string&, AssemblyWorker,
                             const std::array<bool, 3>& flags);


  std::map<AssemblyWorker, FunctionTypeCounter> counter;


  std::map<std::string, std::map<AssemblyWorker, FunctionTypeCounter>> index;


  /**
   * Pointer to \c info_box object
   */
  intg_info_box_t* ptr_info_box;


  /**
   * Discrete finite element function data at dof nodes
   */
  dealii::AnyData fe_fcn_data;
};


/* ************************************************** */
/**
 * Base Assembler class using \c MeshWorker framework
 */
/* ************************************************** */
template <typename LinsysType>
class MeshWorkerAssemblerBase : public AssemblerBase<LinsysType>
{
 public:
  constexpr static int dim = LinsysType::spacedim;
  constexpr static int dimension = LinsysType::spacedim;

  /** 
   * Linear system type 
   */
  using linsys_type = LinsysType;

  /** 
   * Matrix type as used in \c LinearSystem class
   */
  using matrix_type = typename LinsysType::matrix_type;

  /** 
   * Vector type as used in \c LinearSystem class
   */
  using vector_type = typename LinsysType::vector_type;

  /** 
   * Value as contained in matrix and vector
   */
  using value_type = typename vector_type::value_type;

  using dof_info_t = dealii::MeshWorker::DoFInfo<dim, dim, value_type>;
  using integration_info_t = dealii::MeshWorker::IntegrationInfo<dim, dim>;
  using local_matrix_t = dealii::FullMatrix<value_type>;
  using local_vector_t = dealii::Vector<value_type>;
  using local_integrator_type =
    dealii::MeshWorker::LocalIntegrator<dim, dim, value_type>;


 protected:
  /**
   * Constructor
   */
  MeshWorkerAssemblerBase(LinsysType& linsys, bool construct_mapping_adhoc)
    : AssemblerBase<LinsysType>(linsys, construct_mapping_adhoc)
  {}


  /**
   * Destructor
   */
  virtual ~MeshWorkerAssemblerBase() = default;


  /**
   * Simplify retrieving data from \c global_data in \c IntegratorInfo by
   * invoking an identifier string
   */
  FEFunctionSelector<dim, dim, value_type> fe_fcn_selector;
};


/* ************************************************** */
/**
 * Assembler for a mass matrix using the
 * \c MeshWorker framework.
 */
/* ************************************************** */
template <typename LinsysType>
class MassMatrixAssemblerBase : public MeshWorkerAssemblerBase<LinsysType>
{
 public:
  constexpr static int dim = LinsysType::spacedim;
  using base_type = MeshWorkerAssemblerBase<LinsysType>;
  using linsys_type = LinsysType;
  using typename base_type::dof_info_t;
  using typename base_type::local_vector_t;
  using typename base_type::matrix_type;
  using typename base_type::value_type;
  using typename base_type::vector_type;


  /**
   * Assemble the mass matrix
   */
  void assembly_kernel(
    const dealii::MeshWorker::LocalIntegrator<dim, dim, value_type>&);


  /**
   * Assembly function
   */
  virtual void assemble() = 0;


 protected:
  /**
   * Constructor
   */
  MassMatrixAssemblerBase(LinsysType& linear_system,
                          bool construct_mapping_adhoc)
    : base_type(linear_system, construct_mapping_adhoc)
  {}
};


/*******************************/
/** Useless generic definition */
/*******************************/
template <int dim, typename NumberType,
          template <int, typename> class LinsysContainer>
class MassMatrixAssembler
  : public MassMatrixAssemblerBase<LinsysContainer<dim, NumberType>>
{
 public:
  using LinsysType = LinsysContainer<dim, NumberType>;


  MassMatrixAssembler(LinsysType&) = delete;
};


/***********************************/
/** Spcialization for LinearSystem */
/***********************************/
template <int dim, typename NumberType>
class MassMatrixAssembler<dim, NumberType, LinearSystem>
  : public MassMatrixAssemblerBase<LinearSystem<dim, NumberType>>
{
  using base_type = MassMatrixAssemblerBase<LinearSystem<dim, NumberType>>;
  class MassMatrixIntegrator;

 public:
  using typename base_type::dof_info_t;
  using typename base_type::integration_info_t;
  using typename base_type::linsys_type;
  using typename base_type::local_vector_t;
  using typename base_type::matrix_type;
  using typename base_type::value_type;
  using typename base_type::vector_type;


  /**
   * Constructor
   */
  MassMatrixAssembler(linsys_type& linear_system,
                      bool construct_mapping_adhoc = true)
    : base_type(linear_system, construct_mapping_adhoc)
  {}


  virtual void assemble() override
  {
    MassMatrixIntegrator local_integrator;
    this->assembly_kernel(local_integrator);
  }
};


/****************************************/
/** Spcialization for BlockLinearSystem */
/****************************************/
template <int dim, typename NumberType>
class MassMatrixAssembler<dim, NumberType, BlockLinearSystem>
  : public MassMatrixAssemblerBase<BlockLinearSystem<dim, NumberType>>
{
  using base_type = MassMatrixAssemblerBase<BlockLinearSystem<dim, NumberType>>;
  class MassMatrixIntegrator;

 public:
  using typename base_type::dof_info_t;
  using typename base_type::integration_info_t;
  using typename base_type::linsys_type;
  using typename base_type::local_vector_t;
  using typename base_type::matrix_type;
  using typename base_type::value_type;
  using typename base_type::vector_type;


  MassMatrixAssembler(linsys_type& linear_system,
                      bool construct_mapping_adhoc = true)
    : base_type(linear_system, construct_mapping_adhoc)
  {}


  virtual void assemble() override
  {
    MassMatrixIntegrator local_integrator;
    this->assembly_kernel(local_integrator);
  }
};

FELSPA_NAMESPACE_CLOSE
/* -------- Template Implementations ---------*/
#include "src/system_assembler.implement.h"
/* -------------------------------------------*/
#endif /* _FELSPA_LINEAR_ALGEBRA_SYSTEM_ASSEMBLER_H_ */
