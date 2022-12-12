#ifndef _FELSPA_LEVEL_SET_MATERIAL_COMPOSITING_IMPLEMENT_H_
#define _FELSPA_LEVEL_SET_MATERIAL_COMPOSITING_IMPLEMENT_H_

#include <felspa/level_set/material_stack.h>

FELSPA_NAMESPACE_OPEN

/* -------------------- */
namespace ls
/* -------------------- */
{
  template <int dim, typename NumberType>
  struct MaterialStack<dim, NumberType>::MaterialLevelSet
  {
    using value_type = NumberType;

    /**
     * Constructor.
     */
    template <typename LevelSetSimulator>
    MaterialLevelSet(
      const std::shared_ptr<MaterialBase<dim, value_type>>& p_material,
      const std::shared_ptr<LevelSetSimulator>& p_level_set)
      : ptr_material(p_material),
        ptr_simulator(p_level_set),
        ptr_level_set(p_level_set.get())
    {
      ASSERT(p_material != nullptr, ExcNullPointer());
      ASSERT(p_level_set != nullptr, ExcNullPointer());

#ifdef DEBUG
      auto p_lvset =
        dynamic_cast<LevelSetBase<dim, value_type>*>(p_level_set.get());
      ASSERT(p_lvset != nullptr, ExcArgumentCheckFail());
#endif  // DEBUG //
    }

    /**
     * Forbid copy construction.
     */
    MaterialLevelSet(const MaterialLevelSet&) = delete;


    /**
     * Forbid copy assignment.
     */
    MaterialLevelSet& operator=(const MaterialLevelSet&) = delete;


    /**
     * Material parameters.
     */
    const std::shared_ptr<MaterialBase<dim, NumberType>> ptr_material;


    /**
     *
     */
    const std::shared_ptr<SimulatorBase<dim, NumberType>> ptr_simulator;


    /**
     * Pointer with
     */
    const LevelSetBase<dim, NumberType>* ptr_level_set;
  };


  template <int dim, typename NumberType>
  MaterialStack<dim, NumberType>::MaterialStack(const std::string& id_string)
    : MaterialBase<dim, NumberType>(id_string)
  {}


  template <int dim, typename NumberType>
  MaterialStack<dim, NumberType>::MaterialStack(
    const std::shared_ptr<MaterialBase<dim, value_type>>& p_background_material,
    const std::string& id_string)
    : MaterialBase<dim, NumberType>(id_string)
  {
    set_background_material(p_background_material);
  }


  template <int dim, typename NumberType>
  void MaterialStack<dim, NumberType>::set_background_material(
    const std::shared_ptr<MaterialBase<dim, value_type>>& p_material)
  {
    ASSERT(p_material != nullptr, ExcNullPointer());
    ASSERT(numerics::is_zero(p_material->get_time()),
           EXCEPT_MSG("Level set must be added at zero time"));

    ptr_background_material = p_material;
  }


  template <int dim, typename NumberType>
  template <typename LevelSetSimulator>
  void MaterialStack<dim, NumberType>::push(
    const std::shared_ptr<MaterialBase<dim, value_type>>& p_material,
    const std::shared_ptr<LevelSetSimulator>& p_level_set)
  {
    ASSERT(p_material != nullptr, ExcNullPointer());
    ASSERT(p_level_set != nullptr, ExcNullPointer());

    using numerics::is_zero;
    ASSERT(is_zero(p_level_set->get_time()),
           EXCEPT_MSG("Level set must be added at zero time"));
    ASSERT(is_zero(p_material->get_time()),
           EXCEPT_MSG("Material must be added at zero time"));

    material_lvsets.emplace_front(p_material, p_level_set);
  }


  template <int dim, typename NumberType>
  void MaterialStack<dim, NumberType>::clear()
  {
    material_lvsets.clear();
    ptr_background_material = nullptr;
    this->phsx_time = 0.0;
  }


  template <int dim, typename NumberType>
  void MaterialStack<dim, NumberType>::advance_time(time_step_type time_step)
  {
    ASSERT(is_synchronized(), ExcNotSynchronized());

    ptr_background_material->advance_time(time_step);

    for (auto& ml : material_lvsets) {
      // forward the material function to current time_step
      ml.ptr_simulator->advance_time(time_step);
      // evolve the corresponding level set function
      ml.ptr_material->advance_time(time_step);
    }

    ASSERT(is_synchronized(), ExcNotSynchronized());
    this->phsx_time += time_step;
  }


  template <int dim, typename NumberType>
  auto MaterialStack<dim, NumberType>::advance_time(time_step_type time_step,
                                                    bool compute_single_cycle)
    -> time_step_type
  {
    ASSERT(is_synchronized(), ExcNotSynchronized());

    if (numerics::is_zero(time_step)) return 0.0;
    time_step_type cumulative_time = 0.0;
    std::vector<time_step_type> dt_list;

    do {
      dt_list.push_back(time_step);

      for (auto& ml : material_lvsets)
        dt_list.push_back(ml.ptr_simulator->estimate_max_time_step());

      // take the smallest time step among all possible time steps
      time_step_type time_substep =
        *std::min_element(dt_list.begin(), dt_list.end());

      // advance all level set with this time step
      std::for_each(material_lvsets.begin(),
                    material_lvsets.end(),
                    [&](MaterialLevelSet& ml) {
                      ml.ptr_material->advance_time(time_substep);
                      ml.ptr_simulator->advance_time(time_substep);
                    });

      ptr_background_material->advance_time(time_substep);

      ASSERT(time_step >= 0.0, ExcInternalErr());
      time_step -= time_substep;
      cumulative_time += time_substep;
      this->phsx_time += time_substep;
    } while (!numerics::is_zero(time_step) && !compute_single_cycle);

    ASSERT(is_synchronized(), ExcNotSynchronized());

    return cumulative_time;
  }


  template <int dim, typename NumberType>
  bool MaterialStack<dim, NumberType>::has_background_material() const
  {
    return ptr_background_material != nullptr;
  }


  template <int dim, typename NumberType>
  auto MaterialStack<dim, NumberType>::operator[](size_type idx)
    -> std::pair<MaterialBase<dim, NumberType>&,
                 SimulatorBase<dim, value_type>&>
  {
    ASSERT(idx > 0 && idx <= material_lvsets.size(),
           ExcOutOfRange(idx, 1, material_lvsets.size() + 1));

    const size_type idx_rev = material_lvsets.size() - idx;

    return {*material_lvsets[idx_rev].ptr_material,
            *material_lvsets[idx_rev].ptr_simulator};
  }


  template <int dim, typename NumberType>
  auto MaterialStack<dim, NumberType>::operator[](size_type idx) const
    -> std::pair<const MaterialBase<dim, NumberType>&,
                 const SimulatorBase<dim, value_type>&>
  {
    // re-use the definition for non-const
    return const_cast<MaterialStack<dim, NumberType>*>(this)->operator[](idx);
  }


  template <int dim, typename NumberType>
  bool MaterialStack<dim, NumberType>::is_synchronized() const
  {
    ASSERT(has_background_material(),
           EXCEPT_MSG("Not initialized with background material"));

    if (!this->is_synced_with(*ptr_background_material)) return false;

    for (const auto& ml : material_lvsets)
      if (!this->is_synced_with(*ml.ptr_material) ||
          !this->is_synced_with(*ml.ptr_simulator))
        return false;

    return true;
  }


  template <int dim, typename NumberType>
  auto MaterialStack<dim, NumberType>::n_materials() const -> size_type
  {
    return (ptr_background_material == nullptr ? 0 : 1) +
           material_lvsets.size();
  }


  template <int dim, typename NumberType>
  std::shared_ptr<MaterialAccessorBase<dim, NumberType>>
  MaterialStack<dim, NumberType>::generate_accessor(
    const dealii::Quadrature<dim>& quadrature) const
  {
    return std::make_shared<
      MaterialAccessor<ls::MaterialStack<dim, value_type>>>(*this, quadrature);
  }


  template <int dim, typename NumberType>
  void MaterialStack<dim, NumberType>::print(std::ostream& os) const
  {
    os << "\n*** MaterialStack: " << this->get_label_string() << " ***"
       << std::endl;
    os << "-----------------------------------------" << std::endl;

    os << ">>> Materials from Top to Bottom <<<" << std::endl;
    for (const auto& i : material_lvsets) i.ptr_material->print(os);
    os << "-----------------------------------------" << std::endl;

    os << ">>> Background Material <<<" << std::endl;
    ptr_background_material->print(os);
    os << "-----------------------------------------\n" << std::endl;
  }

}  // namespace ls


/* ************************************************** */
/*                 MaterialCompositor                 */
/* ************************************************** */
template <int dim, typename NumberType>
MaterialAccessor<ls::MaterialStack<dim, NumberType>>::MaterialAccessor(
  const material_type& mat_stack, const dealii::Quadrature<dim>& quadrature)
  : MaterialAccessorBase<dim, NumberType>(mat_stack, quadrature),
    ptr_mesh(mat_stack.material_lvsets.empty()
               ? nullptr
               : &mat_stack.material_lvsets.begin()->ptr_simulator->get_mesh())
{
  make_fe_values();
}


template <int dim, typename NumberType>
MaterialAccessor<ls::MaterialStack<dim, NumberType>>::MaterialAccessor(
  const MaterialAccessor<ls::MaterialStack<dim, NumberType>>& that)
  : MaterialAccessorBase<dim, NumberType>(that), ptr_mesh(that.ptr_mesh)
{
  make_fe_values();
}


template <int dim, typename NumberType>
void MaterialAccessor<ls::MaterialStack<dim, NumberType>>::make_fe_values()
{
  ASSERT(fevals.empty(), ExcInternalErr());
  ASSERT(this->ptr_material != nullptr, ExcInternalErr());

  using namespace dealii;

  // construct fevals
  const UpdateFlags update_flags = update_values | update_quadrature_points;

  for (const auto& ml : ptr_material_stack()->material_lvsets) {
    // All level sets should point to the same mesh
    ASSERT(&ml.ptr_simulator->get_mesh() == ptr_mesh, ExcMeshNotSame());
    // Constructing FEValues corresponding to the level set simulator.
    fevals.emplace_back(ml.ptr_simulator->get_mapping(),
                        ml.ptr_simulator->get_fe(), this->quadrature,
                        update_flags);
  }
}


template <int dim, typename NumberType>
void MaterialAccessor<ls::MaterialStack<dim, NumberType>>::reinit(
  const typename dealii::DoFHandler<dim>::active_cell_iterator& cell)
{
  ASSERT(consistent_with_material_stack(), ExcMaterialStackInconsistent());

  // no point to do anything if no compositing is required:
  // everything is the background material.
  if (fevals.empty()) return;

  // Now we reinit every FEValues.
  using cell_iterator_t =
    typename dealii::DoFHandler<dim>::active_cell_iterator;
  using numerics::is_zero;

  const auto nqpt = this->quadrature.size();
  const auto nmatrl = ptr_material_stack()->material_lvsets.size();

  reset();  // note that material_proportion is has nqpt size.

  std::vector<std::vector<value_type>> lvset_qpt_matrl(nqpt);
  std::for_each(lvset_qpt_matrl.begin(), lvset_qpt_matrl.end(),
                [nmatrl](std::vector<value_type>& v) { v.resize(nmatrl); });

  auto ml = ptr_material_stack()->material_lvsets.begin();
  unsigned int imatrl = 0;
  for (auto& fe : fevals) {
    cell_iterator_t cell_iter(ptr_mesh, cell->level(), cell->index(),
                              &ml->ptr_simulator->get_dof_handler());
    fe.reinit(cell_iter);  // FEValues reinitialization

    // compute level set values at each quadrature point
    std::vector<value_type> lvset_qpt(nqpt);
    ml->ptr_level_set->extract_level_set_values(fe, lvset_qpt);

    // transpose into lvset_qpt_matrl
    for (unsigned int iqpt = 0; iqpt < nqpt; ++iqpt)
      lvset_qpt_matrl[iqpt][imatrl] = lvset_qpt[iqpt];

    ++ml;      // increment the material-level set iterator
    ++imatrl;  // increment material counter
  }            // loop through fevals


  // all proportions of materials sum to 1.0
  std::vector<value_type> proportion_sum(nqpt, 1.0);

  for (unsigned int iqpt = 0; iqpt < nqpt; ++iqpt) {
    unsigned int imat = 0;
    // as long as the proportion is not exhausted
    while (imat < nmatrl && !is_zero(proportion_sum[iqpt])) {
      const ls::LevelSetBase<dim, value_type>* pls =
        ptr_material_stack()->material_lvsets[imat].ptr_level_set;

      // compute heaviside
      value_type heaviside = pls->domain_identity(lvset_qpt_matrl[iqpt][imat]);
      ASSERT(heaviside >= 0.0, ExcUnexpectedValue<value_type>(heaviside));

      if (!is_zero(heaviside)) {
        // scale the proportion by heaviside
        value_type proportion = proportion_sum[iqpt] * heaviside;

        // at this qpt, this material will participate in
        // determining the material constant
        const MaterialBase<dim, value_type>* p_mat =
          ptr_material_stack()->material_lvsets[imat].ptr_material.get();
        bool status =
          material_proportion[iqpt].insert({p_mat, proportion}).second;
        UNUSED_VARIABLE(status);
        ASSERT(status, ExcInternalErr());


        cache_material(p_mat);
        proportion_sum[iqpt] -= proportion;

        ASSERT(proportion_sum[iqpt] > 0.0 ||
                 numerics::is_zero(proportion_sum[iqpt], nmatrl),
               ExcUnexpectedValue<value_type>(proportion_sum[iqpt]));
      }  // heaviside > 0.0

      ++imat;  // increment counter to the next material
    }          // material-loop


    if (!is_zero(proportion_sum[iqpt])) {
      // setting background material
      ASSERT(proportion_sum[iqpt] > 0.0, ExcInternalErr());

      const MaterialBase<dim, value_type>* p_mat =
        ptr_material_stack()->ptr_background_material.get();
      material_proportion[iqpt].insert({p_mat, proportion_sum[iqpt]});
      cache_material(p_mat);
    }

#ifdef DEBUG
    types::DoubleType cumsum = 0.0;
    for (const auto& matprop : material_proportion[iqpt])
      cumsum += matprop.second;
    ASSERT(is_zero(cumsum - 1.0), ExcInternalErr());
    ASSERT_NON_EMPTY(material_proportion[iqpt]);
#endif  // DEBUG //
  }  // iqpt-loop

#ifdef VERBOSE
  if (!fevals.empty()) {
    // grab the first material
    auto pls = ptr_material_stack()->material_lvsets[0].ptr_level_set;

    auto ctr = fevals.begin()->get_cell()->center();
    std::cout << "At point (" << ctr << "): ";
    for (const auto& lvset_mat : lvset_qpt_matrl)
      std::cout << " " << lvset_mat[0] << ' '
                << pls->domain_identity(lvset_mat[0]) << " |";
    std::cout << std::endl;
  }
#endif  // VERBOSE
}


template <int dim, typename NumberType>
void MaterialAccessor<ls::MaterialStack<dim, NumberType>>::eval_scalars(
  MaterialParameter parameter_type,
  const PointsField<dim, NumberType>& pts_field,
  std::vector<value_type>& value_qpt)
{
  ASSERT(ptr_material_stack()->has_background_material(),
         ExcBackgroundMaterialUndefined());
  ASSERT_SAME_SIZE(value_qpt, this->quadrature);

  const auto nqpt = this->quadrature.size();

  std::fill(value_qpt.begin(), value_qpt.end(), 0.0);

  // cache material parameters present in this cell
  for (auto& mat_scalar : material_scalars_cache) {
    mat_scalar.second.resize(nqpt);
    mat_scalar.first->scalar_values(parameter_type, pts_field,
                                    mat_scalar.second);
  }

  for (unsigned int iqpt = 0; iqpt < this->quadrature.size(); ++iqpt) {
    ASSERT_NON_EMPTY(material_proportion[iqpt]);

    for (const auto& mp_pair : material_proportion[iqpt]) {
      const MaterialBase<dim, value_type>* p_mat = mp_pair.first;
      ASSERT(p_mat != nullptr, ExcExpiredPointer());

      value_type mat_param = material_scalars_cache[p_mat][iqpt];
      value_type proportion = mp_pair.second;
      value_qpt[iqpt] += mat_param * proportion;
    }  // material_proportion-loop
  }    // iqpt-loop

#ifdef VERBOSE
  if (!fevals.empty()) {
    auto ctr = fevals.begin()->get_cell()->center();
    std::cout << "At point (" << ctr << "): ";
    for (const auto mat_prop : value_qpt) std::cout << mat_prop << ' ';
    std::cout << std::endl;
  }
#endif  // VERBOSE //
}


template <int dim, typename NumberType>
bool MaterialAccessor<
  ls::MaterialStack<dim, NumberType>>::consistent_with_material_stack() const
{
  const auto& mls = ptr_material_stack()->material_lvsets;

  // first check size
  if (mls.size() != fevals.size()) return false;

  // then compare each fevalues and material-levelset entry
  auto pml = ptr_material_stack()->material_lvsets.begin();

  for (auto pfe = fevals.begin(); pfe != fevals.end(); ++pfe, ++pml) {
    if (ptr_mesh != &pml->ptr_simulator->get_mesh() ||
        pml->ptr_simulator->get_fe() != pfe->get_fe())
      return false;
  }

  return true;
}


template <int dim, typename NumberType>
auto MaterialAccessor<ls::MaterialStack<dim, NumberType>>::ptr_material_stack()
  const -> const material_type*
{
  auto p_material =
    static_cast<const MaterialBase<dim, NumberType>*>(this->ptr_material);
  const auto ptr_material_stack =
    dynamic_cast<const ls::MaterialStack<dim, NumberType>*>(p_material);
  ASSERT(ptr_material_stack != nullptr, ExcNullPointer());
  return ptr_material_stack;
}


template <int dim, typename NumberType>
void MaterialAccessor<ls::MaterialStack<dim, NumberType>>::cache_material(
  const MaterialBase<dim, value_type>* p_mat)
{
  ASSERT(p_mat != nullptr, ExcNullPointer());
  material_scalars_cache.emplace(
    std::make_pair(p_mat, std::vector<value_type>()));
}


template <int dim, typename NumberType>
void MaterialAccessor<ls::MaterialStack<dim, NumberType>>::reset()
{
  material_proportion.clear();
  material_scalars_cache.clear();
  material_proportion.resize(this->quadrature.size());
}

FELSPA_NAMESPACE_CLOSE
#endif  // _FELSPA_LEVEL_SET_MATERIAL_COMPOSITING_IMPLEMENT_H_ //
