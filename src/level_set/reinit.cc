#include <felspa/level_set/reinit.h>

#include <algorithm>
#include <cmath>

#ifdef HAS_CPP_PARALLEL_ALGORITHM
#include <execution>
#endif  //  HAS_CPP_PARALLEL_ALGORITHM //

#define USE_LF_FLUX
// #define USE_TANH_SMOOTHING

FELSPA_NAMESPACE_OPEN

namespace
{
using dealii::Tensor;

template <int dim, typename NumberType>
NumberType godunov_gradient_norm(NumberType sign,
                                 const Tensor<1, dim, NumberType>& lgrad,
                                 const Tensor<1, dim, NumberType>& rgrad)
{
  using std::max, std::min;
  NumberType norm = 0.0;
  if (sign >= 0) {
    for (unsigned int idim = 0; idim < dim; ++idim) {
      NumberType lp = max(lgrad[idim], 0.0);
      NumberType rm = -min(rgrad[idim], 0.0);
      norm += max(lp * lp, rm * rm);
    }
  } else {
    for (unsigned int idim = 0; idim < dim; ++idim) {
      NumberType rp = max(rgrad[idim], 0.0);
      NumberType lm = -min(lgrad[idim], 0.0);
      norm += max(lm * lm, rp * rp);
    }
  }
  return std::sqrt(norm);
}

}  // namespace

/* -------------------------------------------*/
namespace dg
/* -------------------------------------------*/
{
/* ************************************************** */
/*           \class ReinitOperator                    */
/* ************************************************** */
template <int dim, typename NumberType>
void ReinitOperator<dim, NumberType>::initialize(
  const HJSimulator<dim, value_type>& hj_sim)
{
  this->ptr_hj_simulator = &hj_sim;
  cell_diameter = hj_sim.get_mesh().get_info().min_diameter;
  // cached_signum_values.clear();
  const auto& reinit_sim =
    dynamic_cast<const ReinitSimulatorLDG<dim, NumberType>&>(hj_sim);
  ptr_initial_solution_vector = &(reinit_sim.get_initial_solution_vector());
  // ASSERT_NON_EMPTY(*ptr_initial_solution_vector);

  ASSERT(cell_diameter > 0.0, ExcInternalErr());

  // const auto& soln_vector = hj_sim.get_solution_vector();
  // dealii::FEValues<dim> feval(
  //   hj_sim.get_mapping(), hj_sim.get_fe(), hj_sim.get_quadrature(),
  //   dealii::update_values | dealii::update_gradients);

  // cache the signum values of level set at quadrature points
  // for (const auto& cell : hj_sim.get_dof_handler().active_cell_iterators())
  // {
  //   feval.reinit(cell);
  //   const auto nqpt = feval.n_quadrature_points;

  //   std::vector<value_type> soln_at_qpt(nqpt);
  //   std::vector<dealii::Tensor<1, dim, value_type>> grad_at_qpt(nqpt);
  //   std::vector<value_type> signum_at_qpt(nqpt);
  //   feval.get_function_values(soln_vector, soln_at_qpt);
  //   feval.get_function_gradients(soln_vector, grad_at_qpt);

  //   for (unsigned int iq = 0; iq < nqpt; ++iq)
  //     signum_at_qpt[iq] =
  //       smooth_signum(soln_at_qpt[iq], grad_at_qpt[iq].norm());

  //   // std::vector<dealii::Tensor<1, dim, value_type>> lgrad_at_qpt(nqpt),
  //   //   rgrad_at_qpt(nqpt);
  //   // this->extract_cell_gradients(feval, lgrad_at_qpt, rgrad_at_qpt);
  //   // for (unsigned int iq = 0; iq < nqpt; ++iq) {
  //   //   signum_at_qpt[iq] =
  //   //     smooth_signum(soln_at_qpt[iq], lgrad_at_qpt[iq],
  //   rgrad_at_qpt[iq]);
  //   // }

  //   auto status = cached_signum_values.insert({cell, signum_at_qpt});
  //   ASSERT(status.second, ExcInternalErr());
  // }  // cell-loop
}


template <int dim, typename NumberType>
FELSPA_FORCE_INLINE bool ReinitOperator<dim, NumberType>::is_initialized() const
{
  return this->ptr_hj_simulator != nullptr &&
         ptr_initial_solution_vector != nullptr &&
         ptr_initial_solution_vector->size() > 0;
}


template <int dim, typename NumberType>
auto ReinitOperator<dim, NumberType>::smooth_signum(
  value_type phi0, const dealii::Tensor<1, dim, value_type>& lgrad,
  const dealii::Tensor<1, dim, value_type>& rgrad) const -> value_type
{
  // extract the correct gradient approximation by using Godunov scheme
  value_type grad_phi0 = godunov_gradient_norm(phi0, lgrad, rgrad);
  // value_type grad_phi0 = (0.5 *  (lgrad + rgrad)).norm();
  return smooth_signum(phi0, grad_phi0);
}

template <int dim, typename NumberType>
auto ReinitOperator<dim, NumberType>::smooth_signum(
  value_type phi0, value_type grad_phi0_norm) const -> value_type
{
#ifdef USE_TANH_SMOOTHING
  value_type alpha = 2 * grad_phi0_norm * cell_diameter;
  return std::tanh(constants::PI * phi0 / alpha);
#else
  return phi0 / std::sqrt(phi0 * phi0 + cell_diameter * cell_diameter *
                                          grad_phi0_norm * grad_phi0_norm);
  // return phi0 / std::sqrt(phi0 * phi0 +  cell_diameter * cell_diameter);
#endif  // USE_TANH_SMOOTHING
}


template <int dim, typename NumberType>
FELSPA_FORCE_INLINE auto ReinitOperator<dim, NumberType>::operator()(
  const value_type smooth_sign,
  const dealii::Tensor<1, dim, value_type>& lgrad,
  const dealii::Tensor<1, dim, value_type>& rgrad) const -> value_type
{
  using namespace std;

#ifdef USE_LF_FLUX
  // Local Lax-Friedrich flux

  dealii::Tensor<1, dim, value_type> avg_grad, min_abs_grad;

  for (unsigned int idim = 0; idim < dim; ++idim) {
    avg_grad[idim] = 0.5 * (lgrad[idim] + rgrad[idim]);
    // min_norm += min(lgrad[idim] * lgrad[idim], rgrad[idim] * rgrad[idim]);
    min_abs_grad[idim] = min(abs(lgrad[idim]), abs(rgrad[idim]));
  }

  value_type hj = smooth_sign * (avg_grad.norm() - 1.0);
  // min_norm = sqrt(min_norm);

  for (unsigned int idim = 0; idim < dim; ++idim) {
    value_type max_abs_grad = max(abs(lgrad[idim]), abs(rgrad[idim]));
    NumberType min_grad_norm = min_abs_grad.norm();

    // hj += 0.5 * abs(smooth_sign) *
    //       sqrt(min_abs_grad.norm_square() -
    //            min_abs_grad[idim] * min_abs_grad[idim] +
    //            max_abs_grad * max_abs_grad) *
    //       (lgrad[idim] - rgrad[idim]);
    hj += 0.5 * abs(smooth_sign) * (lgrad[idim] - rgrad[idim]);
  }

  AssertIsFinite(hj);
  return hj;

#else
  value_type hj_val = 0.0;
  // Godunov Scheme //
  if (smooth_sign >= 0) {
    for (unsigned int idim = 0; idim < dim; ++idim) {
      value_type lp = max(lgrad[idim], 0.0);
      value_type rm = -min(rgrad[idim], 0.0);
      hj_val += max(lp * lp, rm * rm);
    }
  } else {
    for (unsigned int idim = 0; idim < dim; ++idim) {
      value_type lm = -min(lgrad[idim], 0.0);
      value_type rp = max(rgrad[idim], 0.0);
      hj_val += max(lm * lm, rp * rp);
    }
  }
  return smooth_sign * (sqrt(hj_val) - 1.0);
#endif
}


template <int dim, typename NumberType>
auto ReinitOperator<dim, NumberType>::operator()(
  const value_type smooth_sign,
  const dealii::Tensor<1, dim, value_type>& soln_normal,
  const dealii::Tensor<1, dim, value_type>& left_grad,
  const dealii::Tensor<1, dim, value_type>& right_grad) const -> value_type
{
  dealii::Tensor<1, dim, value_type> avg_grad, upwind_grad, downwind_grad,
    normal;
  NumberType min_grad_norm_sq = 0.0, max_abs_grad = 0.0;

  for (unsigned int idim = 0; idim < dim; ++idim) {
    avg_grad[idim] = 0.5 * (left_grad[idim] + right_grad[idim]);
    normal[idim] = soln_normal[idim] / soln_normal.norm();
    NumberType local_max_grad =
      std::max(std::abs(left_grad[idim]), std::abs(right_grad[idim]));
    max_abs_grad = std::max(max_abs_grad, local_max_grad);
    min_grad_norm_sq += std::min(left_grad[idim] * left_grad[idim],
                                 right_grad[idim] * right_grad[idim]);

    if (smooth_sign * normal[idim] > 0) {
      upwind_grad[idim] = left_grad[idim];
      downwind_grad[idim] = right_grad[idim];
    } else {
      upwind_grad[idim] = right_grad[idim];
      downwind_grad[idim] = left_grad[idim];
    }
  }  // idim-loop

  value_type hj = smooth_sign * (avg_grad.norm() - 1.0);
  // if (std::abs(normal * upwind_grad) / upwind_grad.norm() < 0.1 ||
  //     std::abs(normal * downwind_grad / downwind_grad.norm()) < 0.1)
  //   std::cout << normal << ' ' << upwind_grad << ' ' << downwind_grad
  //             << std::endl;

  hj += 0.5 * std::abs(smooth_sign) * max_abs_grad /
        std::sqrt(min_grad_norm_sq) *
        (upwind_grad * normal - downwind_grad * normal);
  return hj;
}

template <int dim, typename NumberType>
auto ReinitOperator<dim, NumberType>::characteristic_velocity(
  value_type signum, const dealii::Tensor<1, dim, value_type>& lgrad,
  const dealii::Tensor<1, dim, value_type>& rgrad) const
  -> dealii::Tensor<1, dim, value_type>
{
  using namespace std;
  dealii::Tensor<1, dim, value_type> grad;

  if (signum >= 0) {
    for (int idim = 0; idim < dim; ++idim) {
      value_type lp = max(lgrad[idim], 0.0);
      value_type rm = -min(rgrad[idim], 0.0);
      grad[idim] = std::abs(lp) > std::abs(rm) ? lp : rm;
    }
  } else {
    for (int idim = 0; idim < dim; ++idim) {
      value_type lm = -min(lgrad[idim], 0.0);
      value_type rp = max(rgrad[idim], 0.0);
      grad[idim] = std::abs(lm) > std::abs(rp) ? lm : rp;
    }
  }

  if (!numerics::is_zero(grad.norm())) {
    grad *= 1.0 / grad.norm();
    grad *= signum;
  }

  return grad;
}


template <int dim, typename NumberType>
void ReinitOperator<dim, NumberType>::cell_values(
  const dealii::FEValuesBase<dim>& feval, std::vector<value_type>& hj_qpt) const
{
  using dealii::Tensor;
  const auto nqpt = feval.n_quadrature_points;

  ASSERT(is_initialized(), ExcNotInitialized());
  ASSERT(hj_qpt.size() == nqpt, ExcSizeMismatch(hj_qpt.size(), nqpt));

  std::vector<value_type> signum(nqpt), init_soln(nqpt);
  std::vector<Tensor<1, dim, value_type>> init_soln_grad(nqpt);

  feval.get_function_values(*ptr_initial_solution_vector, init_soln);
  feval.get_function_gradients(*ptr_initial_solution_vector, init_soln_grad);
  for (unsigned int iqpt = 0; iqpt != nqpt; ++iqpt)
    signum[iqpt] = smooth_signum(init_soln[iqpt], init_soln_grad[iqpt].norm());

  std::vector<Tensor<1, dim, value_type>> lgrad(nqpt), rgrad(nqpt);
  this->extract_cell_gradients(feval, lgrad, rgrad);

#if 1
  // Old method: apply flux dimension by dimension
  for (unsigned int iq = 0; iq < nqpt; ++iq)
    hj_qpt[iq] = (*this)(signum[iq], lgrad[iq], rgrad[iq]);
#else
  // new method: project onto the local level set normal vector
  std::vector<Tensor<1, dim, value_type>> soln_grad(nqpt);
  feval.get_function_gradients(this->ptr_hj_simulator->get_solution_vector(),
                               soln_grad);
  for (unsigned int iq = 0; iq < nqpt; ++iq) {
    hj_qpt[iq] = (*this)(signum[iq], soln_grad[iq], lgrad[iq], rgrad[iq]);
  }
#endif
}


template <int dim, typename NumberType>
void ReinitOperator<dim, NumberType>::cell_velocities(
  const dealii::FEValuesBase<dim>& feval,
  std::vector<dealii::Tensor<1, dim, value_type>>& hj_velo_qpt) const
{
  const auto nqpt = feval.n_quadrature_points;
  ASSERT(hj_velo_qpt.size() == nqpt, ExcSizeMismatch(hj_velo_qpt.size(), nqpt));

  std::vector<value_type> signum(nqpt), init_soln(nqpt);
  std::vector<Tensor<1, dim, value_type>> init_soln_grad(nqpt);

  feval.get_function_values(*ptr_initial_solution_vector, init_soln);
  feval.get_function_gradients(*ptr_initial_solution_vector, init_soln_grad);
  for (unsigned int iqpt = 0; iqpt != nqpt; ++iqpt)
    signum[iqpt] = smooth_signum(init_soln[iqpt], init_soln_grad[iqpt].norm());

  std::vector<dealii::Tensor<1, dim, value_type>> lgrad(nqpt), rgrad(nqpt);
  this->extract_cell_gradients(feval, lgrad, rgrad);

  for (unsigned int iq = 0; iq < nqpt; ++iq) {
    ASSERT(-1.0 <= signum[iq] && signum[iq] <= 1.0,
           ExcUnexpectedValue(signum[iq]));
    hj_velo_qpt[iq] = characteristic_velocity(signum[iq], lgrad[iq], rgrad[iq]);
  }
}


template <int dim, typename NumberType>
void ReinitOperator<dim, NumberType>::export_signum_values(
  ExportFile& file, const dealii::DoFHandler<dim>& dofh) const
{
  ASSERT(is_initialized(),
         EXCEPT_MSG("The HJ operator has not been initialized."));
  UNUSED_VARIABLE(file);
  UNUSED_VARIABLE(dofh);

  // dealii::DataOut<dim> data_out;
  // data_out.attach_dof_handler(dofh);
  // data_out.add_data_vector(signum_values, "Reinit_HJ_Opeartor");
  // data_out.build_patches();
  // data_out.write_vtk(file.access_stream());
}


/* ************************************************** */
/*              \class ReinitSimulatorLDG                */
/* ************************************************** */

template <int dim, typename NumberType>
ReinitSimulatorLDG<dim, NumberType>::ReinitSimulatorLDG(
  Mesh<dim, value_type>& triag,
  unsigned int fe_degree,
  const std::string& label)
  : HJSimulator<dim, NumberType>(triag, fe_degree, label)
{
  this->post_local_gradient_limiting.connect(boost::bind(
    &ReinitSimulatorLDG<dim, value_type>::apply_curvature_limiting, this));
}


template <int dim, typename NumberType>
ReinitSimulatorLDG<dim, NumberType>::ReinitSimulatorLDG(
  const fe_simulator_type& advect_sim)
  : HJSimulator<dim, NumberType>(advect_sim)
{
  this->post_local_gradient_limiting.connect(boost::bind(
    &ReinitSimulatorLDG<dim, value_type>::apply_curvature_limiting, this));
}


template <int dim, typename NumberType>
void ReinitSimulatorLDG<dim, NumberType>::initialize(
  const ScalarFunction<dim, value_type>& initial_condition,
  bool use_independent_solution)
{
  auto p_hj = std::make_shared<ReinitOperator<dim, value_type>>();
  HJSimulator<dim, value_type>::initialize(initial_condition, p_hj, false,
                                           use_independent_solution);
  initial_solution_vector = this->get_solution_vector();
}


template <int dim, typename NumberType>
void ReinitSimulatorLDG<dim, NumberType>::initialize(
  const TimedSolutionVector<vector_type, time_step_type>& other_soln)
{
  auto p_hj = std::make_shared<ReinitOperator<dim, value_type>>();
  HJSimulator<dim, value_type>::initialize(other_soln, p_hj);
  initial_solution_vector = this->get_solution_vector();
}


template <int dim, typename NumberType>
auto ReinitSimulatorLDG<dim, NumberType>::advance_time() -> time_step_type
{
  value_type max_velo_diam = this->max_velocity_over_diameter(this->get_time());
  const TempoControl<value_type>& tempo_control = *this->ptr_control->ptr_tempo;
  value_type max_cfl = tempo_control.get_cfl().second;

  ASSERT(max_cfl > 0.0, EXCEPT_MSG("Max CFL must be positive."));

  time_step_type time_step = max_cfl / max_velo_diam;

  ASSERT(time_step > 0.0, ExcInternalErr());
  ASSERT(tempo_control.query_method().second == TempoCategory::exp,
         EXCEPT_MSG("ReinitSimulatorLDG only allow explicit time stepping."));

  switch (tempo_control.query_method().first) {
    case TempoMethod::rktvd1:
      this->tempo_integrator
        .template advance_time_step<RungeKuttaTVD<1>, TempoCategory::exp>(
          *this, time_step);
      break;
    case TempoMethod::rktvd2:
      this->tempo_integrator
        .template advance_time_step<RungeKuttaTVD<2>, TempoCategory::exp>(
          *this, time_step);
      break;
    case TempoMethod::rktvd3:
      this->tempo_integrator
        .template advance_time_step<RungeKuttaTVD<3>, TempoCategory::exp>(
          *this, time_step);
      break;
    default:
      THROW(ExcNotImplemented());
  }

  // MinmodShockDetector<dim, vector_type> shock_detector(
  //   this->mesh(), this->get_solution_vector());
  // const std::vector<bool>& shock_cell_flags =
  //   shock_detector.detect_shock_cells(
  //     this->get_dof_handler().active_cell_iterators(), this->get_mapping(),
  //     this->get_fe(), this->get_quadrature());

  // // if constexpr (dim == 2) {
  // //   ExportFile svg_file("limited_cells.svg");
  // //   shock_detector.export_mesh(svg_file);
  // // }

  // // apply WENO limiter to the shock cells
  // std::vector<bool> saved_flags;
  // this->mesh().save_user_flags(saved_flags);
  // this->mesh().load_user_flags(shock_cell_flags);
  // auto has_user_flag = [&](const Mesh<dim, NumberType>& mesh) {
  //   for (const auto& cell : mesh.active_cell_iterators())
  //     if (cell->user_flag_set()) return true;
  //   return false;
  // };

  // // while (has_user_flag(this->mesh())) {
  // //   WENOLimiter<dim, vector_type> weno_limiter(*this->solution);
  // //   auto flagged_cell_range =
  // //     filter_iterators(this->get_dof_handler().active_cell_iterators(),
  // //                      dealii::IteratorFilters::UserFlagSet());
  // //   weno_limiter.apply_limiting(flagged_cell_range, this->get_mapping(),
  // //                               this->get_fe(), this->get_quadrature());
  // // }

  // WENOLimiter<dim, vector_type> weno_limiter(*this->solution);
  // auto flagged_cell_range =
  //   filter_iterators(this->get_dof_handler().active_cell_iterators(),
  //                    dealii::IteratorFilters::UserFlagSet());
  // weno_limiter.apply_limiting(flagged_cell_range, this->get_mapping(),
  //                             this->get_fe(), this->get_quadrature());

  // this->mesh().load_user_flags(saved_flags);

  // apply_moment_limiter(this->get_dof_handler(),
  //                      this->get_mapping(), this->get_quadrature(),
  //                      *this->solution);

  if constexpr (dim == 2)
    apply_weno_limiter(this->mesh(), this->get_dof_handler(),
                       this->get_mapping(), this->get_quadrature(),
                       *this->solution);
  return time_step;
}


template <int dim, typename NumberType>
void ReinitSimulatorLDG<dim, NumberType>::attach_control(
  const std::shared_ptr<ReinitControl<dim, value_type>>& pcontrol)
{
  base_type::attach_control(pcontrol);
}


template <int dim, typename NumberType>
auto ReinitSimulatorLDG<dim, NumberType>::gradient_deviation(
  const vector_type& ls_values, bool global_solve) const -> value_type
{
  using namespace dealii;

  const auto fe_degree = this->get_fe().degree;

  QGauss<dim> quadrature(fe_degree + 1);
  FEValues<dim> fe_values(this->get_fe(), quadrature,
                          update_gradients | update_JxW_values);

  value_type deviation_sum{0.0}, volume_sum{0.0};

  for (const auto& cell : this->dof_handler().active_cell_iterators()) {
    if (global_solve || cell->user_flag_set()) {
      fe_values.reinit(cell);

      const auto nqpt = fe_values.n_quadrature_points;
      const auto& JxW = fe_values.get_JxW_values();
      using nqpt_type = typename std::remove_const<decltype(nqpt)>::type;

      std::vector<Tensor<1, dim, value_type>> ls_grads(nqpt);
      fe_values.get_function_gradients(ls_values, ls_grads);

      value_type cell_deviation = 0.0;
      for (nqpt_type iq = 0; iq < nqpt; ++iq)
        cell_deviation += std::abs(ls_grads[iq].norm() - 1.0) * JxW[iq];

      deviation_sum += cell_deviation;
      volume_sum += cell->measure();
    }
  }  // cell-loop

  return deviation_sum / volume_sum;
}


template <int dim, typename NumberType>
auto ReinitSimulatorLDG<dim, NumberType>::run_iteration() -> size_type
{
  ASSERT(this->dof_handler().has_active_dofs(), ExcDoFHandlerNotInit());

  auto p_control =
    dynamic_cast<ReinitControl<dim, NumberType>*>(this->ptr_control.get());
  ASSERT(p_control != nullptr, ExcNullPointer());

  auto& additional_control = p_control->additional_control;
  if (additional_control.global_solve)
    return run_iteration_global(additional_control);
  else
    return run_iteration_fixed_width(additional_control);
}


template <int dim, typename NumberType>
auto ReinitSimulatorLDG<dim, NumberType>::run_iteration_global(
  AdditionalControl& c) -> size_type
{
  LOG_PREFIX("ReinitSimulatorLDG");

  size_type iteration_count{0};
  value_type cumulative_time{0.0};

  auto p_control =
    dynamic_cast<ReinitControl<dim, NumberType>*>(this->ptr_control.get());
  ASSERT(p_control != nullptr, ExcNullPointer());

  value_type tol =
    this->solution->linfty_norm() * p_control->additional_control.tolerance;

  value_type max_abs_lvset = std::accumulate(
    this->get_solution_vector().begin(), this->get_solution_vector().end(), 0.0,
    [](const value_type& a, const value_type& b) {
      return std::max(a, std::abs(b));
    });

  felspa_log << "Running global level set reinit interation up to "
             << max_abs_lvset << "..." << std::endl;

  // stop when the information
  // has propogated throughout the whole domain
  while (cumulative_time < 1.2 * max_abs_lvset &&
         iteration_count <= c.max_iter) {
    cumulative_time += advance_time();
    ++iteration_count;

#ifdef DEBUG
    value_type grad_deviation =
      gradient_deviation(this->get_solution_vector(), true);
    felspa_log << "At iteration " << iteration_count + 1 << '/' << c.max_iter
               << ", Deviation of gradient from unity  = " << grad_deviation
               << '/' << tol << std::endl;

    this->export_solutions();
#endif  // DEBUG //
  }

  return iteration_count;
}


template <int dim, typename NumberType>
auto ReinitSimulatorLDG<dim, NumberType>::run_iteration_fixed_width(
  AdditionalControl& c) -> size_type
{
  LOG_PREFIX("ReinitSimulatorLDG");
  ASSERT(c.band_width_coeff > 1.0,
         EXCEPT_MSG("The band width coefficient must be greater than 1.0"));

  value_type grad_deviation;
  value_type cumulative_time{0.0};
  size_type iteration_count{0};

  time_step_type band_width =
    c.band_width_coeff * this->ptr_mesh->get_info().min_diameter;

  // value_type tol =
  //   this->solution->linfty_norm() * additional_control.tolerance;

  while (cumulative_time < band_width && iteration_count <= c.max_iter) {
    grad_deviation = gradient_deviation(this->get_solution_vector(), false);

    // if (grad_deviation < tol) break;

    cumulative_time += advance_time();
    ++iteration_count;

    felspa_log << "At iteration " << iteration_count << '/' << c.max_iter
               << ", difference  = " << grad_deviation << std::endl;
    //  '/' << tol << std::endl;

    // export file
#ifdef DEBUG
    this->export_solutions();
#endif  // DEBUG //
  }

  if (iteration_count == c.max_iter)
    THROW(ExcNotEnoughStepsForWidth(band_width));
  else
    felspa_log << "Completed after " << iteration_count
               << " iterations with gradient deviation = " << grad_deviation
               << std::endl;

  return iteration_count;
}


template <int dim, typename NumberType>
auto ReinitSimulatorLDG<dim, NumberType>::get_initial_solution_vector() const
  -> const vector_type&
{
  return initial_solution_vector;
}


template <int dim, typename NumberType>
void ReinitSimulatorLDG<dim, NumberType>::apply_curvature_limiting()
{
  std::vector<bool> saved_flags;
  this->ptr_mesh->save_user_flags(saved_flags);

  // label the troubled cells
  CurvatureShockDetector<dim, NumberType> detector(
    this->mesh(), this->get_initial_solution_vector(),
    this->left_local_gradients, this->right_local_gradients);

  std::vector<bool> shock_flags = detector.detect_shock_cells(
    this->dof_handler().active_cell_iterators(), this->get_mapping(),
    this->get_fe(), this->get_quadrature());

  this->ptr_mesh->load_user_flags(shock_flags);
  unsigned int si_degree = std::min(2u, this->fe_degree());
  auto flagged_cell_range =
    dealii::filter_iterators(this->get_dof_handler().active_cell_iterators(),
                             dealii::IteratorFilters::UserFlagSet());

  // for (int idim = 0; idim != dim; ++idim) {
  //   WENOLimiter<dim, vector_type> lgrad_limiter(
  //     this->left_local_gradients.block(idim), si_degree, 0.001);
  //   WENOLimiter<dim, vector_type> rgrad_limiter(
  //     this->right_local_gradients.block(idim), si_degree, 0.001);

  //   lgrad_limiter.apply_limiting(flagged_cell_range, this->get_mapping(),
  //                                this->get_fe(), this->get_quadrature());
  //   rgrad_limiter.apply_limiting(flagged_cell_range, this->get_mapping(),
  //                                this->get_fe(), this->get_quadrature());
  // }

  // reload previously saved flags
  this->ptr_mesh->load_user_flags(saved_flags);
}


#ifdef FELSPA_BETTER_REINIT
/* ************************************************** */
/*           \class ReinitSimulatorLDGLDG                */
/* ************************************************** */
template <int dim, typename NumberType>
IPReinitSimmlator<dim, NumberType>::IPReinitSimulator(
  Mesh<dim, value_type>& triag, unsigned int fe_degree,
  const std::string& label)
  : base_type(triag, fe_degree, label),
    ptr_control(std::make_shared<TempoControl<dim, value_type>>()),

{
  this->tempo_integrator.attach_control(ptr_control);
  this->set_quadrature(dealii::QGauss<dim>(fe_degree + 2));
}


template <int dim, typename NumberType>
NumberType IPReinitSimulator<dim, NumberType>::hj_cell_operator(
  const NumberType phi0,
  const dealii::Tensor<1, dim, NumberType>& grad_phi0,
  const dealii::Tensor<1, dim, NumberType>& grad) const
{
  const min_diam = this->mesh().get_info().min_diameter;
  return smooth_signum(phi0, grad_phi0, min_diam) * (grad.norm() - 1);
}


template <int dim, typename NumberType>
dealii::Tensor<1, dim, NumberType>
IPReinitSimulator<dim, NumberType>::hj_face_operator(
  const NumberType phi0,
  const dealii::Tensor<1, dim, NumberType>& grad_phi0,
  const dealii::Tensor<1, dim, NumberType>& normal,
  const dealii::Tensor<1, dim, NumberType>& grad_int,
  const dealii::Tensor<1, dim, NumberType>& grad_ext,
  NumberType max_penalty_coeff)
{
  const min_diam = this->mesh().get_info().min_dimeter;
  NumberType convect_speed = smooth_signum()
}


template <int dim, typename NumberType>
IPReinitSimulator<dim, NumberType>::do_initialize()
{
  this->temporal_passive_members.clear();
  this->reset_time();
}

template <int dim, typename NumberType>
NumberType IPReinitSimulator<dim, NumberType>::smooth_signum(
  const NumberType phi0, const NumberType eta)
{
  ASSERT(!numerics::is_equal(eta, 0.0), ExcArgumentCheckFail(eta));
  return phi0 / std::sqrt(phi0 * phi0 + eta * eta);
}


template <int dim, typename NumberType>
NumberType IPReinitSimulator<dim, NumberType> smoooth_signum(
  const Numbertype phi0, const dealii::Tensor<1, dim, NumberType>& grad_phi0,
  const NumberType eta)
{
  ASSERT(!numerics::is_equal(eta, 0.0), ExcArgumentCheckFail(eta));
#ifdef USE_TANH_SMOOTHING
  NumberType alpha = 2 * grad_phi0.norm() * eta;
  return std::tanh(constants::PI * phi0 / alpha);
#else
  const NumberType grad_phi0_norm2 = grad_phi0.norm_square();
  return phi0 / std::sqrt(phi0 * phi0 + grad_phi0_norm2 * eta * eta);
#endif
}

#endif  // FELSPA_BETTER_REINIT

}  // namespace dg

/* -------- Explicit Instantiations ----------*/
template class dg::ReinitOperator<1>;
template class dg::ReinitOperator<2>;
template class dg::ReinitOperator<3>;

#ifdef FELSPA_BETTER_REINIT
template class dg::IPReinitSimulator<1>;
template class dg::IPReinitSimulator<2>;
template class dg::IPReinitSimulator<3>;
#else
template class dg::ReinitSimulatorLDG<1>;
template class dg::ReinitSimulatorLDG<2>;
template class dg::ReinitSimulatorLDG<3>;
#endif  // FELSPA_BETTER_REINIT
/* -------------------------------------------*/
FELSPA_NAMESPACE_CLOSE
