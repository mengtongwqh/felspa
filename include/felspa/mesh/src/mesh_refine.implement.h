#ifndef _FELSPA_MESH_MESH_REFINE_IMPLEMENT_H_
#define _FELSPA_MESH_MESH_REFINE_IMPLEMENT_H_

#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <felspa/base/exceptions.h>
#include <felspa/base/log.h>
#include <felspa/mesh/mesh_refine.h>

FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/*                 MeshFlagOperator                   */
/* ************************************************** */
template <int dim, typename NumberType>
MeshFlagOperator<dim, NumberType>::MeshFlagOperator(Mesh<dim, value_type>& mesh)
  : ptr_mesh(&mesh)
{
  using namespace dealii;
  // cache refinement flags
  std::transform(ptr_mesh->active_cell_iterators().begin(),
                 ptr_mesh->active_cell_iterators().end(),
                 std::back_inserter(cached_refine_flags),
                 [](const TriaActiveIterator<CellAccessor<dim, dim>>& cell) {
                   return static_cast<bool>(cell->refine_flag_set());
                 });

  // cache coarsen flags
  std::transform(ptr_mesh->active_cell_iterators().begin(),
                 ptr_mesh->active_cell_iterators().end(),
                 std::back_inserter(cached_coarsen_flags),
                 [](const TriaActiveIterator<CellAccessor<dim, dim>>& cell) {
                   return static_cast<bool>(cell->coarsen_flag_set());
                 });

  ASSERT(cached_coarsen_flags.size() == this->ptr_mesh->n_active_cells(),
         ExcInternalErr());
  ASSERT(cached_refine_flags.size() == this->ptr_mesh->n_active_cells(),
         ExcInternalErr());

  // clean up all the flags
  std::for_each(ptr_mesh->active_cell_iterators().begin(),
                ptr_mesh->active_cell_iterators().end(),
                [](const TriaActiveIterator<CellAccessor<dim, dim>>& cell) {
                  cell->clear_coarsen_flag();
                  cell->clear_refine_flag();
                });
}


template <int dim, typename NumberType>
void MeshFlagOperator<dim, NumberType>::limit_level(int min_level,
                                                    int max_level) const
{
  ASSERT(min_level < max_level, ExcArgumentCheckFail());

  for (auto& cell : this->ptr_mesh->active_cell_iterators()) {
    if (cell->level() >= max_level && cell->refine_flag_set())
      cell->clear_refine_flag();
    if (cell->level() <= min_level && cell->coarsen_flag_set())
      cell->clear_coarsen_flag();
  }
}


template <int dim, typename NumberType>
void MeshFlagOperator<dim, NumberType>::prioritize_refinement() const
{
  auto it_rflag = cached_refine_flags.cbegin();
  auto it_cflag = cached_coarsen_flags.cbegin();

  for (auto& cell : this->ptr_mesh->active_cell_iterators()) {
    if (cell->refine_flag_set() || *it_rflag) {
      cell->clear_coarsen_flag();
      cell->set_refine_flag();
    } else if (cell->coarsen_flag_set() || *it_cflag) {
      cell->clear_refine_flag();
      cell->set_coarsen_flag();
    }
    ++it_rflag;
    ++it_cflag;
  }
}


template <int dim, typename NumberType>
template <typename OstreamType>
void MeshFlagOperator<dim, NumberType>::print_info(OstreamType& os)
{
  using namespace dealii;
  auto zero = static_cast<types::SizeType>(0);

  types::SizeType n_rflags_old = std::accumulate(
    cached_refine_flags.cbegin(), cached_refine_flags.cend(), zero);
  types::SizeType n_cflags_old = std::accumulate(
    cached_coarsen_flags.cbegin(), cached_coarsen_flags.cend(), zero);

  auto cell_begin = this->ptr_mesh->active_cell_iterators().begin();
  auto cell_end = this->ptr_mesh->active_cell_iterators().end();

  types::SizeType n_updated_rflags =
    std::accumulate(cell_begin,
                    cell_end,
                    zero,
                    [](types::SizeType count,
                       const TriaActiveIterator<CellAccessor<dim>>& cell) {
                      return cell->refine_flag_set() ? ++count : count;
                    });

  types::SizeType n_updated_cflags =
    std::accumulate(cell_begin,
                    cell_end,
                    zero,
                    [](types::SizeType count,
                       const TriaActiveIterator<CellAccessor<dim>>& cell) {
                      return cell->coarsen_flag_set() ? ++count : count;
                    });

  os << "Mesh flag status: Refine " << n_rflags_old << " --> "
     << n_updated_rflags << ",  Coarsen " << n_cflags_old << " --> "
     << n_updated_cflags << std::endl;
}


/* ************************************************** */
/*                 MeshRefiner                        */
/* ************************************************** */
template <int dim, typename NumberType>
void MeshRefiner<dim, NumberType>::append(
  const std::shared_ptr<SolutionTransferBase>& p_soln_trans)
{
  ASSERT(p_soln_trans != nullptr, ExcNullPointer());
  solution_transfers.push_back(p_soln_trans);
}


template <int dim, typename NumberType>
template <typename SimulatorType>
void MeshRefiner<dim, NumberType>::append(const SimulatorType& sim)
{
  ASSERT(&sim.mesh() == ptr_mesh, ExcMeshNotSame());
  solution_transfers.push_back(sim.ptr_solution_transfer);
}


template <int dim, typename NumberType>
void MeshRefiner<dim, NumberType>::run_coarsen_and_refine(
  bool run_soln_transfer)
{
  // If the mesh is not labelled with coarsening/refinement flag,
  // there is no point to do the coarsening/refinement.
  if (!has_update_pending()) return;

  LOG_PREFIX("MeshRefiner");

  const auto old_mesh_size = this->mesh().n_active_cells();
  this->mesh().prepare_coarsening_and_refinement();

  if (run_soln_transfer)
    for (auto& soln_trans : solution_transfers)
      soln_trans->prepare_for_coarsening_and_refinement();

  this->mesh().execute_coarsening_and_refinement();

  if (run_soln_transfer) {
    for (auto& soln_trans : solution_transfers) soln_trans->interpolate();
  }

  this->update_pending = false;

  felspa_log << "Mesh refinement/coarsening done: " << old_mesh_size << " --> "
             << this->mesh().n_active_cells() << " cells." << std::endl;

  if (run_soln_transfer)
    felspa_log << solution_transfers.size()
               << " solution vectors are transferred to new mesh config."
               << std::endl;
}


FELSPA_NAMESPACE_CLOSE
#endif  // _FELSPA_MESH_MESH_REFINE_IMPLEMENT_H_ //
