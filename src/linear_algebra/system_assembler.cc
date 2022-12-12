
#include <felspa/linear_algebra/system_assembler.h>

FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/**
 * Displaying the AssemblyFramework to ostream.
 */
/* ************************************************** */
std::ostream& operator<<(std::ostream& os, AssemblyFramework framework)
{
  switch (framework) {
    case AssemblyFramework ::serial:
      os << "serial";
      break;
    case AssemblyFramework::workstream:
      os << "workstream";
      break;
    case AssemblyFramework::meshworker:
      os << "meshworker";
      break;
    default:
      THROW(ExcNotImplemented());
  }
  return os;
}


/* ************************************************** */
/**
 * \class FEFunctionSelector
 */
/* ************************************************** */

template <int dim, int spacedim, typename NumberType>
FEFunctionSelector<dim, spacedim, NumberType>::FEFunctionSelector(
  intg_info_box_t& info_box)
  : ptr_info_box(&info_box)
{
  counter[AssemblyWorker::cell] = FunctionTypeCounter{};
  counter[AssemblyWorker::face] = FunctionTypeCounter{};
  counter[AssemblyWorker::boundary] = FunctionTypeCounter{};
}


template <int dim, int spacedim, typename NumberType>
void FEFunctionSelector<dim, spacedim, NumberType>::assign_selector_flags(
  const std::string& label, AssemblyWorker worker_type,
  const std::array<bool, 3>& flags)
{
  index[label][worker_type].values =
    flags[0] ? counter[worker_type].values++ : constants::invalid_unsigned_int;
  index[label][worker_type].gradients = flags[1]
                                          ? counter[worker_type].gradients++
                                          : constants::invalid_unsigned_int;
  index[label][worker_type].hessians = flags[2]
                                         ? counter[worker_type].hessians++
                                         : constants::invalid_unsigned_int;
}


template <int dim, int spacedim, typename NumberType>
void FEFunctionSelector<dim, spacedim, NumberType>::reset()
{
  for (auto& i : counter) i.second.reset();
  index.clear();
  fe_fcn_data = dealii::AnyData();
  ptr_info_box = nullptr;
}


template <int dim, int spacedim, typename NumberType>
std::vector<std::vector<NumberType>>
FEFunctionSelector<dim, spacedim, NumberType>::values(
  const std::string& label, AssemblyWorker worker_type,
  const intg_info_t& cinfo) const
{
  unsigned int i = index.at(label).at(worker_type).values;
  ASSERT(i != constants::invalid_unsigned_int, ExcNotExist("values", label));
  return cinfo.values[i];
}


template <int dim, int spacedim, typename NumberType>
std::vector<std::vector<dealii::Tensor<1, dim, NumberType>>>
FEFunctionSelector<dim, spacedim, NumberType>::gradients(
  const std::string& label, AssemblyWorker worker_type,
  const intg_info_t& cinfo) const
{
  unsigned int i = index.at(label).at(worker_type).gradients;
  ASSERT(i != constants::invalid_unsigned_int, ExcNotExist("gradients", label));
  return cinfo.gradients[i];
}


template <int dim, int spacedim, typename NumberType>
std::vector<std::vector<dealii::Tensor<2, dim, NumberType>>>
FEFunctionSelector<dim, spacedim, NumberType>::hessians(
  const std::string& label, AssemblyWorker worker_type,
  const intg_info_t& cinfo) const
{
  unsigned int i = index.at(label).at(worker_type).hessians;
  ASSERT(i != constants::invalid_unsigned_int, ExcNotExist("hessians", label));
  return cinfo.hessians[i];
}

/* -------- Explicit Instantiations ----------*/
#include "system_assembler.inst"
/* -------------------------------------------*/
FELSPA_NAMESPACE_CLOSE
