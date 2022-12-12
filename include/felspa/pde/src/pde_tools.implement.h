#ifndef _FELSPA_PDE_PDE_TOOLS_IMPLEMENT_H_
#define _FELSPA_PDE_PDE_TOOLS_IMPLEMENT_H_

#include <deal.II/base/thread_management.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <felspa/pde/pde_tools.h>

FELSPA_NAMESPACE_OPEN


/* ************************************************** */
/**
 * Function implementation for \c CFLEstimator.
 */
/* ************************************************** */
template <typename SimulatorType>
auto CFLEstimator<SimulatorType>::do_estimate(
  const cell_iterator_type& begin,
  const cell_iterator_type& end,
  const dealii::Mapping<dim>& mapping,
  const dealii::FiniteElement<dim>& fe) -> value_type
{
  using namespace dealii;
  using this_type = CFLEstimator<SimulatorType>;

  UpdateFlags update_flags =
    update_values | update_quadrature_points | update_gradients;
  QTrapezoid<dim> vertex_quad;
  ScratchData scratch(mapping, fe, vertex_quad, update_flags);
  CopyData copy;

  WorkStream::run(begin,
                  end,
                  *this,
                  &this_type::local_estimate,
                  &this_type::compare_local_to_global,
                  scratch,
                  copy);

  return this->velo_diam;
}


template <typename SimulatorType>
void CFLEstimator<SimulatorType>::local_estimate(const cell_iterator_type& cell,
                                                 ScratchData& s,
                                                 CopyData& c)
{
  if constexpr (is_simulator<SimulatorType>::value) {
    typename dealii::DoFHandler<dim>::active_cell_iterator cell_iter(
      &cell->get_triangulation(), cell->level(), cell->index(),
      &ptr_simulator->get_dof_handler());

    s.feval.reinit(cell_iter);
  } else {
    s.feval.reinit(cell);
  }

  // get magnitude of velocities at qpt
  extract_cell_velocity_magnitude(*ptr_simulator, s.feval, s.velocities);

  c.cell_velo_diam =
    *std::max_element(s.velocities.cbegin(), s.velocities.cend());
  c.cell_velo_diam = c.cell_velo_diam / cell->diameter();
}


template <typename SimulatorType>
void CFLEstimator<SimulatorType>::compare_local_to_global(const CopyData& copy)
{
  velo_diam = std::max(copy.cell_velo_diam, velo_diam);
}


/* ************************************************** */
/**
 *  Local ScratchData used for CFL estimation.
 */
/* ************************************************** */
template <typename SimulatorType>
struct CFLEstimator<SimulatorType>::ScratchData
{
  /**
   * Constructor
   */
  ScratchData(const dealii::Mapping<dim>& mapping,
              const dealii::FiniteElement<dim>& fe,
              const dealii::Quadrature<dim>& quad,
              dealii::UpdateFlags update_flags)
    : feval(mapping, fe, quad, update_flags),
      velocities(feval.n_quadrature_points)
  {}


  /**
   * Copy Constructor
   */
  ScratchData(const ScratchData& that)
    : feval(that.feval.get_mapping(), that.feval.get_fe(),
            that.feval.get_quadrature(), that.feval.get_update_flags()),
      velocities(that.velocities)
  {}


  /**
   * FEValues object
   */
  dealii::FEValues<dim> feval;

  /**
   * Magnitude of velocity
   */
  std::vector<value_type> velocities;
};


/* ************************************************** */
/**
 *  Local CopyData used for CFL estimation.
 */
/* ************************************************** */
template <typename SimulatorType>
struct CFLEstimator<SimulatorType>::CopyData
{
  /**
   * Value of \f$ \frac{velocity magnitude}{cell diameter} \f$
   * for the current cell.
   */
  value_type cell_velo_diam;
};


/* ************************************************** */
/**
 *  Compute velocity norm over the whole domain
 */
/* ************************************************** */
template <typename SimulatorType>
auto compute_velocity_norm(
  const SimulatorType& sim,
  const dealii::Quadrature<SimulatorType::dimension>& quad,
  Norm norm_type) -> typename SimulatorType::value_type
{
  ASSERT(norm_type == Norm::L2, ExcNotImplemented());
  using namespace dealii;
  using value_type = typename SimulatorType::value_type;
  constexpr int dim = SimulatorType::dimension;

  FEValues<dim> feval(sim.get_mapping(), sim.get_fe(), quad,
                      update_values | update_JxW_values);

  value_type velo_norm = 0.0;
  const unsigned int nqpt = feval.n_quadrature_points;

  for (const auto& cell : sim.get_dof_handler().active_cell_iterators()) {
    feval.reinit(cell);
    std::vector<Tensor<1, dim, value_type>> velocities_qpt(nqpt);
    extract_cell_velocities(sim, feval, velocities_qpt);
    std::vector<value_type> JxW = feval.get_JxW_values();

    for (unsigned int iq = 0; iq != nqpt; ++iq)
      velo_norm += JxW[iq] * velocities_qpt[iq].norm_square();
  }

  return std::sqrt(velo_norm);
}


namespace dg
{
  /* ************************************************** */
  /*                 ShockDetectorBase                  */
  /* ************************************************** */

  template <int dim, typename NumberType>
  ShockDetectorBase<dim, NumberType>::ShockDetectorBase(
    Mesh<dim, NumberType>& mesh_)
    : ptr_mesh(&mesh_)
  {}


  template <int dim, typename NumberType>
  const std::vector<bool>&
  ShockDetectorBase<dim, NumberType>::detect_shock_cells(
    dealii::IteratorRange<ActiveCellIterator> cell_range,
    const dealii::Mapping<dim, dim>& mapping,
    const dealii::FiniteElement<dim, dim>& fe,
    const dealii::Quadrature<dim>& quad)
  {
    LOG_PREFIX("ShockDetector");
    ScratchData scratch_data(mapping, fe, quad);

    // cache the user flags
    std::vector<bool> existing_flags;
    ptr_mesh->save_user_flags(existing_flags);
    ptr_mesh->clear_user_flags();

    for (auto cell : cell_range)
      if (cell_has_shock(cell, scratch_data)) cell->set_user_flag();

    // save troubled cells and reload pre-existing flags
    ptr_mesh->save_user_flags(shock_cell_flags);
    ptr_mesh->load_user_flags(existing_flags);
    felspa_log << std::accumulate(shock_cell_flags.begin(),
                                  shock_cell_flags.end(), 0,
                                  [](unsigned int a, bool b) {
                                    return a + static_cast<unsigned int>(b);
                                  })
               << " cells are marked as troubled cells." << std::endl;

    return shock_cell_flags;
  }


  template <int dim, typename NumberType>
  void ShockDetectorBase<dim, NumberType>::export_mesh(ExportFile& file) const
  {
    ASSERT(file.get_format() == ExportFileFormat::svg, ExcNotImplemented());

    std::vector<bool> tmp_flags;
    ptr_mesh->save_user_flags(tmp_flags);
    ptr_mesh->load_user_flags(shock_cell_flags);

    for (auto cell : ptr_mesh->active_cell_iterators())
      if (cell->user_flag_set())
        cell->set_material_id(1);
      else
        cell->set_material_id(0);


    dealii::GridOut grid_out;
    dealii::GridOutFlags::Svg svg_flags;
    svg_flags.coloring = dealii::GridOutFlags::Svg::material_id;
    svg_flags.draw_legend = true;
    svg_flags.draw_colorbar = true;
    grid_out.set_flags(svg_flags);
    grid_out.write_svg(*ptr_mesh, file.access_stream());

    ptr_mesh->clear_user_flags();
    ptr_mesh->load_user_flags(tmp_flags);
  }


  /* ************************************************** */
  /*         ShockDetectorBase :: ScratchData           */
  /* ************************************************** */
  template <int dim, typename NumberType>
  ShockDetectorBase<dim, NumberType>::ScratchData::ScratchData(
    const dealii::Mapping<dim, dim>& mapping,
    const dealii::FiniteElement<dim, dim>& fe,
    const dealii::Quadrature<dim>& quad)
    : feval_cell(mapping, fe, quad,
                 dealii::update_values | dealii::update_gradients |
                   dealii::update_quadrature_points |
                   dealii::update_JxW_values),
      feval_vertex(mapping, fe, dealii::QTrapezoid<dim>(),
                   dealii::update_values | dealii::update_quadrature_points |
                     dealii::update_JxW_values),
      feval_neighbor(mapping, fe, quad,
                     dealii::update_values | dealii::update_quadrature_points |
                       dealii::update_JxW_values)
  {}


  /* ************************************************** */
  /*                MinmodShockDetector                 */
  /* ************************************************** */
  template <int dim, typename VectorType>
  MinmodShockDetector<dim, VectorType>::MinmodShockDetector(
    Mesh<dim, NumberType>& mesh, const VectorType& soln, NumberType beta_coeff_)
    : base_type(mesh), ptr_solution_vector(&soln), beta_coeff(beta_coeff_)
  {}


  template <int dim, typename VectorType>
  bool MinmodShockDetector<dim, VectorType>::cell_has_shock(
    const ActiveCellIterator& cell, ScratchData& s) const
  {
    using namespace dealii;
    ASSERT(cell->is_active(), ExcInternalErr());

    s.feval_cell.reinit(cell);
    s.feval_vertex.reinit(cell);

    // cell solution average
    NumberType this_cell_avg =
      compute_cell_average(s.feval_cell, *this->ptr_solution_vector);

    // vertex solution
    std::vector<NumberType> soln_vertex(GeometryInfo<dim>::vertices_per_cell);
    s.feval_vertex.get_function_values(*ptr_solution_vector, soln_vertex);

    Point<dim, NumberType> cell_center = cell->center();

    for (unsigned int ivertex = 0;
         ivertex != dealii::GeometryInfo<dim>::vertices_per_cell;
         ++ivertex) {
      // skip the vertex if it is on the boundary
      // if (cell->vertex_iterator(ivertex)->at_boundary()) continue;
      bool skip_vertex = false;
      for (unsigned int iface = 0; iface != dim; ++iface) {
        // if any of the vertex-abutting faces is at the boundary
        if (cell
              ->face(MeshGeometryInfo<dim>::faces_around_vertex[ivertex][iface])
              ->at_boundary())
          skip_vertex = true;
      }

      if (skip_vertex) continue;

      const Point<dim, NumberType> vertex = cell->vertex(ivertex);
      Point<dim, NumberType> center_to_vertex(vertex - cell_center);
      NumberType directional_coeffs[dim];
      NumberType neighbor_cell_avg[dim];

      for (unsigned int iface = 0; iface != dim; ++iface) {
        ActiveCellIterator neighbor_cell = cell;

        Point<dim, NumberType> center_to_neighbor_center;
        unsigned int face =
          MeshGeometryInfo<dim>::faces_around_vertex[ivertex][iface];
        CellIterator neighbor_parent_cell = cell->neighbor(face);

        if (neighbor_parent_cell->has_children()) {
          // the neighboring cell is on a finer level
          // loop thru all finer cells and find the cell
          // closest to the vertex in distance
          NumberType dist_to_vertex = cell->diameter();
          CellIterator closest_child_cell = neighbor_parent_cell;
          for (unsigned int ichild = 0;
               ichild != neighbor_parent_cell->n_children();
               ++ichild) {
            // check distance to vertex
            CellIterator child_cell = neighbor_parent_cell->child(ichild);
            NumberType dist = (child_cell->center() - vertex).norm();
            if (dist < dist_to_vertex) {
              dist_to_vertex = dist;
              closest_child_cell = child_cell;
            }
          }  // ichild-loop
          neighbor_cell = closest_child_cell;
        } else {
          neighbor_cell = neighbor_parent_cell;
        }  // if (neighbor_cell->has_children())

        ASSERT(neighbor_cell->is_active(), ExcInternalErr());

        s.feval_neighbor.reinit(neighbor_cell);
        center_to_neighbor_center = neighbor_cell->center() - cell->center();
        directional_coeffs[iface] = center_to_neighbor_center *
                                    center_to_vertex /
                                    center_to_neighbor_center.norm_square();

        // Compute the cell average of the neighboring cell //
        neighbor_cell_avg[iface] =
          compute_cell_average(s.feval_neighbor, *ptr_solution_vector);
      }  // iface-loop

      // minmod troubled cell test //
      NumberType vertex_center_slope = soln_vertex[ivertex] - this_cell_avg;
      NumberType interp_slope = 0.0;
      for (unsigned iface = 0; iface != dim; ++iface)
        interp_slope += (neighbor_cell_avg[iface] - this_cell_avg) *
                        directional_coeffs[iface];
      if (vertex_center_slope !=
          minmod(vertex_center_slope, beta_coeff * interp_slope)) {
        // std::cout << vertex_center_slope << ' '
        //           << minmod(vertex_center_slope, beta_coeff * interp_slope)
        //           << std::endl;
        return true;
      }
    }  // ivertex_loop
    return false;
  }


  /* ************************************************** */
  /*              CurvatureShockDetector                */
  /* ************************************************** */
  template <int dim, typename NumberType>
  CurvatureShockDetector<dim, NumberType>::CurvatureShockDetector(
    Mesh<dim, NumberType>& mesh,
    const dealii::Vector<NumberType>& initial_solution,
    const dealii::BlockVector<NumberType>& left_grads,
    const dealii::BlockVector<NumberType>& right_grads)
    : base_type(mesh),
      ptr_initial_solution_vector(&initial_solution),
      ptr_left_gradients(&left_grads),
      ptr_right_gradients(&right_grads)
  {
    ASSERT_NON_EMPTY(initial_solution);
    for (int idim = 0; idim != dim; ++idim) {
      ASSERT_NON_EMPTY(left_grads.block(idim));
      ASSERT_NON_EMPTY(right_grads.block(idim));
    }
  }


  template <int dim, typename NumberType>
  bool CurvatureShockDetector<dim, NumberType>::cell_has_shock(
    const ActiveCellIterator& cell_iter, ScratchData& s) const
  {
    using namespace dealii;
    s.feval_cell.reinit(cell_iter);
    const unsigned int nqpt = s.feval_cell.n_quadrature_points;
    const NumberType diam = this->ptr_mesh->get_info().min_diameter;

    // compute left and right gradients
    std::vector<Tensor<1, dim, NumberType>> lgrad(nqpt), rgrad(nqpt);
    std::vector<NumberType> phi(nqpt), lcurv(nqpt, 0.0), rcurv(nqpt, 0.0);
    s.feval_cell.get_function_values(*ptr_initial_solution_vector, phi);

    for (int idim = 0; idim != dim; ++idim) {
      s.feval_cell.get_function_gradients(ptr_left_gradients->block(idim),
                                          lgrad);
      s.feval_cell.get_function_gradients(ptr_right_gradients->block(idim),
                                          rgrad);

      for (unsigned int iqpt = 0; iqpt != nqpt; ++iqpt) {
        lcurv[iqpt] += lgrad[iqpt][idim] * util::sign(phi[iqpt]);
        rcurv[iqpt] += rgrad[iqpt][idim] * util::sign(phi[iqpt]);
      }  // iqpt-loop
    }    // idim-loop

    for (unsigned int iqpt = 0; iqpt != nqpt; ++iqpt)
      if (-diam * lcurv[iqpt] > 1.0 | -diam * rcurv[iqpt] > 1.0) return true;
    return false;

    // return -diam * std::max(*std::max_element(lcurv.begin(), lcurv.end()),
    //                        *std::max_element(rcurv.begin(), rcurv.end())) >
    //        1.0;
  }


  /* ************************************************** */
  /*                  WENOLimiter                       */
  /* ************************************************** */

  template <int dim, typename VectorType>
  WENOLimiter<dim, VectorType>::WENOLimiter(
    VectorType& soln_vector, 
    unsigned int max_deriv_for_smoothness,
    NumberType neighbor_cell_gamma_)
    : old_soln_vector(soln_vector),
      ptr_soln_vector(&soln_vector),
      neighbor_cell_gamma(neighbor_cell_gamma_),
      max_smoothness_indicator_derivative(max_deriv_for_smoothness)
  {}


  template <int dim, typename VectorType>
  template <typename Iterator>
  void WENOLimiter<dim, VectorType>::apply_limiting(
    const dealii::IteratorRange<Iterator>& cells,
    const dealii::Mapping<dim>& mapping, const dealii::FiniteElement<dim>& fe,
    const dealii::Quadrature<dim>& quad)
  {
    LOG_PREFIX("WENOLimiter")
    ScratchData scratch(mapping, fe, quad);
    CopyData copy;

#define FORCE_SERIAL
#ifdef FORCE_SERIAL
    std::size_t counter = 0;
    for (const auto& cell : cells) {
      apply_limiting_to_cell(cell, scratch, copy);
      copy_local_to_global(copy);
      ++counter;
    }
    felspa_log << "Limiter is applied to " << counter << " cells." << std::endl;
#else
    dealii::WorkStream::run(
      cells, *this, &this_type::template apply_limiting_to_cell<Iterator>,
      &this_type::copy_local_to_global, scratch, copy);
#endif  // FORCE_SERIAL //
  }


  template <int dim, typename VectorType>
  auto WENOLimiter<dim, VectorType>::find_neighbor_cells(
    const ActiveCellIterator& cell) -> std::vector<ActiveCellIterator>
  {
    using namespace dealii;
    std::vector<ActiveCellIterator> neighbor_cells;

    for (unsigned int face_no = 0;
         face_no != dealii::GeometryInfo<dim>::faces_per_cell;
         ++face_no) {
      // boundary faces have no neighboring cell
      if (cell->face(face_no)->at_boundary()) continue;

      // if the neighbor cell is finer, then include all subcells
      if (cell->face(face_no)->has_children()) {
        for (unsigned int subface_no = 0;
             subface_no != cell->face(face_no)->n_children();
             ++subface_no) {
          ActiveCellIterator nb_cell =
            cell->neighbor_child_on_subface(face_no, subface_no);
          // if (!nb_cell->user_flag_set())
          neighbor_cells.push_back(nb_cell);
        }
      } else {
        ActiveCellIterator nb_cell = cell->neighbor(face_no);
        // if (!nb_cell->user_flag_set())
        neighbor_cells.push_back(nb_cell);
      }  // cell->face(face_no)->has_children()
    }    // face_no-loop

    return neighbor_cells;
  }


  template <int dim, typename VectorType>
  auto WENOLimiter<dim, VectorType>::compute_smoothness_indicator(
    const dealii::FEValues<dim>& feval_home,
    const dealii::FEValues<dim>& feval_target,
    unsigned int n_deriv) const -> NumberType
  {
    ASSERT(n_deriv <= 2, ExcNotImplemented());
    using dealii::Tensor;

    const NumberType volume = feval_target.get_cell()->measure();
    const auto nqpt_home = feval_home.n_quadrature_points;

    std::vector<Tensor<1, dim, NumberType>> grads(
      feval_target.n_quadrature_points);
    feval_target.get_function_gradients(old_soln_vector, grads);

    NumberType beta = 0.0;

    if (n_deriv >= 1)
      for (unsigned int iqpt = 0; iqpt != nqpt_home; ++iqpt)
        beta += grads[iqpt].norm_square() * feval_home.JxW(iqpt) * volume;

    if (n_deriv >= 2) {
      std::vector<Tensor<2, dim, NumberType>> hessians(
        feval_target.n_quadrature_points);
      feval_target.get_function_hessians(old_soln_vector, hessians);

      for (unsigned int iqpt = 0; iqpt != nqpt_home; ++iqpt) {
        NumberType local_sum = 0.0;

        for (unsigned int i = 0; i != dim; ++i)
          for (unsigned int j = i; j != dim; ++j)
            local_sum += hessians[i][j] * hessians[i][j];

        beta += local_sum * feval_home.JxW(iqpt) * std::pow(volume, 3);
      }  // iqpt-loop
    }    // n_deriv >= 2

    return beta;
  }


  template <int dim, typename VectorType>
  template <typename CellIteratorType>
  void WENOLimiter<dim, VectorType>::apply_limiting_to_cell(
    const CellIteratorType& home_cell, ScratchData& s, CopyData& c)
  {
    using namespace dealii;
    static dealii::UpdateFlags si_update_flags =
      dealii::update_values | dealii::update_gradients |
      dealii::update_hessians | dealii::update_JxW_values;

    std::vector<NumberType> weights;
    std::vector<std::vector<NumberType>> p_new_qpt_cell;

    // initialize home cell
    s.feval_home.reinit(home_cell);
    const auto n_qpt_home = s.feval_home.n_quadrature_points;
    const auto n_dof_home = s.feval_home.get_fe().n_dofs_per_cell();
    // initialize neighbor cells
    std::vector<ActiveCellIterator> neighbor_cells =
      find_neighbor_cells(home_cell);
    const auto n_neighbor_cells = neighbor_cells.size();

    std::vector<NumberType> neighbor_gammas;
    for (unsigned int i = 0; i != n_neighbor_cells; ++i)
      neighbor_gammas.push_back(neighbor_cell_gamma);

    // if (n_neighbor_cells < 2) {
    //   c.assemble = false;
    //   return;
    // }

    // cell average, weights and p_new values for home cell
    const NumberType home_cell_avg =
      compute_cell_average(s.feval_home, old_soln_vector);
    {
      NumberType gamma_home = 1.0 - std::accumulate(neighbor_gammas.begin(),
                                                    neighbor_gammas.end(), 0.0);
      NumberType denominator_home =
        epsilon +
        compute_smoothness_indicator(s.feval_home, s.feval_home,
                                     max_smoothness_indicator_derivative);
      weights.push_back(gamma_home / denominator_home / denominator_home);

      std::vector<NumberType> p_new_qpt_home(n_qpt_home, 0.0);
      s.feval_home.get_function_values(old_soln_vector, p_new_qpt_home);
      p_new_qpt_cell.push_back(std::move(p_new_qpt_home));
    }

    //  cell average, weights and p_new values for neighbor cells
    const std::vector<Point<dim, NumberType>> home_qpts =
      s.feval_home.get_quadrature_points();

    auto cell_gamma = neighbor_gammas.begin();
    for (const ActiveCellIterator& nb_cell : neighbor_cells) {
      // cell average of thie cell
      // s.feval_neighbor.reinit(nb_cell);
      // NumberType this_cell_avg =
      //   compute_cell_average(s.feval_neighbor, old_soln_vector);

      // generate FEValues based on quadrature of home cell
      // convert the homecell quadrature points to the reference coordinates
      // of neighbor cell. if transformation fail,
      // dealii::ExcTransformationFailed will be thrown
      std::vector<Point<dim, NumberType>> home_qpts_nb_ref;
      const Mapping<dim>& mapping = s.feval_home.get_mapping();
      for (const Point<dim, NumberType>& qpt_real : home_qpts) {
        home_qpts_nb_ref.push_back(
          mapping.transform_real_to_unit_cell(nb_cell, qpt_real));
      }

      Quadrature<dim> home_qpt_quad(home_qpts_nb_ref);
      FEValues<dim> feval_nb_home_qpt(mapping, s.feval_home.get_fe(),
                                      home_qpt_quad, si_update_flags);
      feval_nb_home_qpt.reinit(nb_cell);

      // compute weights by smoothness indicator
      NumberType denom = epsilon + compute_smoothness_indicator(
                                     s.feval_home, feval_nb_home_qpt,
                                     max_smoothness_indicator_derivative);
      weights.push_back((*cell_gamma++) / denom / denom);

      // compute the values of the limited polynomials
      // p_new = \sum_l w_l (p_l - avg_l + home_cell_avg)
      // at each quadrature point of the home cell
      std::vector<NumberType> p_new_qpt_nb(n_qpt_home, 0.0);
      feval_nb_home_qpt.get_function_values(old_soln_vector, p_new_qpt_nb);

      //
      NumberType vol{0.0}, fcn_integral{0.0};
      for (unsigned int iqpt = 0; iqpt != n_qpt_home; ++iqpt) {
        fcn_integral += p_new_qpt_nb[iqpt] * s.feval_home.JxW(iqpt);
        vol += s.feval_home.JxW(iqpt);
      }
      ASSERT(dealii::numbers::is_finite(fcn_integral),
             ExcUnexpectedValue(fcn_integral));
      ASSERT(::felspa::numerics::is_nearly_equal(vol, home_cell->measure()),
             ExcInternalErr());
      NumberType this_cell_avg = fcn_integral / home_cell->measure();

      for (auto& p : p_new_qpt_nb) p = p - this_cell_avg + home_cell_avg;
      p_new_qpt_cell.push_back(std::move(p_new_qpt_nb));
    }  // nb_cell-loop

    // normalize weights so that they sum to unity
    NumberType weights_sum = 0.0;
    for (NumberType weight : weights) weights_sum += weight;
    ASSERT(weights_sum > 0.0, ExcUnexpectedValue(weights_sum));
    for (NumberType& weight : weights) { weight /= weights_sum; }

    // now sum polynomials values up with the weights
    ASSERT_SAME_SIZE(weights, p_new_qpt_cell);
    std::vector<NumberType> p_new_qpt(n_qpt_home, 0.0);
    auto wt = weights.begin();
    auto p_new = p_new_qpt_cell.begin();
    for (; wt != weights.end(); ++wt, ++p_new)
      for (unsigned int iqpt = 0; iqpt != n_qpt_home; ++iqpt)
        p_new_qpt[iqpt] += (*p_new)[iqpt] * (*wt);


#ifdef DEBUG
    // check cell average is conserved after modification
    NumberType old_avg = 0.0, new_avg = 0.0;
    for (unsigned int iqpt = 0; iqpt != n_qpt_home; ++iqpt) {
      old_avg += p_new_qpt_cell[0][iqpt] * s.feval_home.JxW(iqpt);
      new_avg += p_new_qpt[iqpt] * s.feval_home.JxW(iqpt);
    }
    ASSERT(::FELSPA_NAMESPACE::numerics::is_equal(old_avg, new_avg),
           ExcInternalErr());
#endif  // DEBUG


    // compute limited expansion coeffs
    c.assemble = true;
    c.local_soln.resize(n_dof_home);
    c.local_dof_indices.resize(n_dof_home);
    home_cell->get_dof_indices(c.local_dof_indices);

    for (unsigned int idof = 0; idof != n_dof_home; ++idof) {
      NumberType p_new_dot_shape = 0.0;
      NumberType shape_dot_shape = 0.0;
      for (unsigned int iqpt = 0; iqpt != n_qpt_home; ++iqpt) {
        NumberType shape_val = s.feval_home.shape_value(idof, iqpt);
        p_new_dot_shape += p_new_qpt[iqpt] * shape_val * s.feval_home.JxW(iqpt);
        shape_dot_shape += shape_val * shape_val * s.feval_home.JxW(iqpt);
      }  // iqpt-loop
      c.local_soln[idof] = p_new_dot_shape / shape_dot_shape;
    }  // idof-loop

    // home_cell->clear_user_flag();
  }


  template <int dim, typename VectorType>
  void WENOLimiter<dim, VectorType>::copy_local_to_global(const CopyData& c)
  {
    ASSERT_SAME_SIZE(c.local_dof_indices, c.local_soln);
    if (c.assemble) {
      auto dof = c.local_dof_indices.begin();
      auto soln = c.local_soln.begin();
      for (; dof != c.local_dof_indices.end(); ++dof, ++soln) {
        (*ptr_soln_vector)[*dof] = *soln;
      }
    }
  }


  template <int dim, typename VectorType>
  WENOLimiter<dim, VectorType>::ScratchData::ScratchData(
    const dealii::Mapping<dim>& mapping, const dealii::FiniteElement<dim>& fe,
    const dealii::Quadrature<dim>& quad)
    : feval_home(mapping, fe, quad,
                 dealii::update_values | dealii::update_gradients |
                   dealii::update_hessians | dealii::update_quadrature_points |
                   dealii::update_JxW_values),
      feval_neighbor(mapping, fe, quad,
                     dealii::update_values | dealii::update_JxW_values)
  {}


  // template <int dim, typename NumberType>
  // WENOLimiter<dim, VectorType>:: ScratchData::ScratchData(const ScratchData&
  // that) :feval_home()


  /* ************************************************** */
  /*                   MomentLimiter                    */
  /* ************************************************** */
  template <int dim, typename VectorType>
  MomentLimiter<dim, VectorType>::MomentLimiter(VectorType& soln_vector,
                                                unsigned int degree)
    : old_soln_vector(soln_vector),
      ptr_soln_vector(&soln_vector),
      fe_degree(degree)
  {}


  template <int dim, typename VectorType>
  template <typename Iterator>
  void MomentLimiter<dim, VectorType>::apply_limiting(
    const dealii::IteratorRange<Iterator>& cells,
    const dealii::Mapping<dim>& mapping, const dealii::FiniteElement<dim>& fe,
    const dealii::Quadrature<dim>& quad)
  {
    ScratchData scratch(mapping, fe, quad);
    CopyData copy;
    for (const auto& cell : cells) {
      apply_limiting_to_cell(cell, scratch, copy);
      copy_local_to_global(copy);
    }
    AssertIsFinite(ptr_soln_vector->l2_norm());
  }


  template <int dim, typename VectorType>
  void MomentLimiter<dim, VectorType>::apply_limiting_to_cell(
    const ActiveCellIterator& home_cell, ScratchData& s, CopyData& c)
  {
    using dealii::Vector;
    ASSERT(dim == 2, ExcNotImplemented());

    // coefficients of the home cell
    const unsigned int n_dof_home = home_cell->get_fe().n_dofs_per_cell();
    ASSERT(std::pow(fe_degree + 1, dim) == n_dof_home,
           ExcUnexpectedValue(n_dof_home));

    s.feval_home.reinit(home_cell);
    c.local_soln.reinit(n_dof_home);
    c.local_dof_indices.resize(n_dof_home);
    home_cell->get_dof_indices(c.local_dof_indices);
    home_cell->get_dof_values(old_soln_vector, c.local_soln);
    AssertIsFinite(c.local_soln.l2_norm());
    ASSERT(c.local_soln.l2_norm() > 1.0e-8,
           ExcUnexpectedValue(c.local_soln.l2_norm()));

    // obtain the backward and forward cell coeffs in each spatial dimension
    std::vector<Vector<NumberType>> forward_coeffs(dim);
    std::vector<Vector<NumberType>> backward_coeffs(dim);
    for (int idim = 0; idim != dim; ++idim) {
      for (unsigned int iface = 0; iface != 2; ++iface) {
        unsigned int face_no =
          MeshGeometryInfo<dim>::faces_normal_to_axis[idim][iface];
        auto face = home_cell->face(face_no);

        if (face->at_boundary()) continue;  // skip the boundary face

        if (face->has_children()) {
          // if the neighbor cell is finer
          // CellIterator nb_cell = home_cell->neighbor()->parent();
          THROW(ExcNotImplemented());

        } else {
          ActiveCellIterator nb_cell = home_cell->neighbor(face_no);
          if (home_cell->level() == nb_cell->level()) {
            // the neighbor cell is on the same level
            // directly obtain the coefficients from neighbor
            Vector<NumberType> nb_dof_values(
              nb_cell->get_fe().n_dofs_per_cell());
            nb_cell->get_dof_values(old_soln_vector, nb_dof_values);
            AssertIsFinite(nb_dof_values.l2_norm());
            ASSERT(nb_dof_values.l2_norm() > 1.0e-8,
                   ExcUnexpectedValue(nb_dof_values.l2_norm()));

            if (iface == 0)  // backward face //
              backward_coeffs[idim] = std::move(nb_dof_values);
            else if (iface == 1)  // forward face //
              forward_coeffs[idim] = std::move(nb_dof_values);
            else
              THROW(ExcInternalErr());

          } else {
            // the neighbor cell is coarser,
            // in which case we are to reconstruct a subcell
            ASSERT(home_cell->level() == nb_cell->level() + 1,
                   ExcInternalErr());
            THROW(ExcNotImplemented());
          }  // if (home_cell->level() == nb_cell->level())
        }    // cell->face(face_no)->has_children()
      }      // iface-loop
    }        // idim-loop

    // with coefficients of the forward and backward cells
    // for each spatial orientation obtained, execute limiting
    limit_coefficients(c.local_soln, backward_coeffs, forward_coeffs);
  }


  template <int dim, typename VectorType>
  void MomentLimiter<dim, VectorType>::copy_local_to_global(
    const CopyData& c) const
  {
    ASSERT_SAME_SIZE(c.local_dof_indices, c.local_soln);
    auto dof = c.local_dof_indices.begin();
    auto soln = c.local_soln.begin();
    for (; dof != c.local_dof_indices.end(); ++dof, ++soln) {
      AssertIsFinite(*soln);
      (*ptr_soln_vector)[*dof] = *soln;
    }
  }


  template <int dim, typename VectorType>
  void MomentLimiter<dim, VectorType>::limit_coefficients(
    dealii::Vector<NumberType>& home_coeffs,
    const std::vector<dealii::Vector<NumberType>>& backward_coeffs,
    const std::vector<dealii::Vector<NumberType>>& forward_coeffs) const
  {
    if constexpr (dim == 2)
      limit_coefficients_2d(home_coeffs, backward_coeffs, forward_coeffs);
    else
      THROW(ExcNotImplemented());
  }


  template <int dim, typename VectorType>
  void MomentLimiter<dim, VectorType>::limit_coefficients_2d(
    dealii::Vector<NumberType>& home_coeffs,
    const std::vector<dealii::Vector<NumberType>>& backward_coeffs,
    const std::vector<dealii::Vector<NumberType>>& forward_coeffs) const
  {
    static const std::vector<NumberType> minmod_coeffs = {0.5 / std::sqrt(3),
                                                          0.5 / std::sqrt(15)};

    for (int ideg = fe_degree; ideg >= 1; --ideg) {
      for (int jdeg = ideg; jdeg >= 0; --jdeg) {
        std::vector<std::array<int, dim>> deg_combinations = {{ideg, jdeg},
                                                              {jdeg, ideg}};

        // if ideg-jdeg is symmetric, limit only once
        bool skip_duplicate = (ideg == jdeg);
        bool limiter_not_active = true;

        for (const std::array<int, dim>& ijdeg : deg_combinations) {
          if (skip_duplicate) {
            skip_duplicate = false;
            continue;
          }

          const unsigned int home_idx = idx(ijdeg);
          std::vector<NumberType> minmod_candidates;
          minmod_candidates.push_back(home_coeffs[home_idx]);

          // consider each coordinate direction
          for (int idim = 0; idim != dim; ++idim) {
            if (ijdeg[idim] == 0) continue;

            std::array<int, dim> nbdeg = ijdeg;
            nbdeg[idim] -= 1;
            const unsigned int nb_idx = idx(nbdeg);

            if (backward_coeffs[idim].size()) {
              NumberType diff =
                home_coeffs[nb_idx] - backward_coeffs[idim][nb_idx];
              minmod_candidates.push_back(diff * minmod_coeffs[nbdeg[idim]]);
            }

            if (forward_coeffs[idim].size()) {
              NumberType diff =
                forward_coeffs[idim][nb_idx] - home_coeffs[nb_idx];
              minmod_candidates.push_back(diff * minmod_coeffs[nbdeg[idim]]);
            }
          }  // idim-loop

          NumberType limited_val = minmod(minmod_candidates);

          limiter_not_active &=
            numerics::is_equal(limited_val, home_coeffs[home_idx]);
          AssertIsFinite(limited_val);
          home_coeffs[home_idx] = limited_val;
        }  // deg_combinations-loop

        // stop limiting if limiter is not active on this level
        if (limiter_not_active) return;
      }  // jdeg-loop
    }    // ideg-loop
  }


  template <int dim, typename VectorType>
  unsigned int MomentLimiter<dim, VectorType>::idx(
    const std::array<int, dim>& nd_idx) const
  {
    unsigned int i = 0;
    for (int idim = 0; idim != dim; ++idim) {
      ASSERT(0 <= nd_idx[idim] && nd_idx[idim] <= static_cast<int>(fe_degree),
             ExcUnexpectedValue(nd_idx[idim]));
      i += nd_idx[idim] * std::pow(fe_degree + 1, idim);
    }
    return i;
  }


  // template <int dim, typename VectorType>
  // auto MomentLimiter<dim, VectorType>::get_coeffs_coarser_neighbor()
  //   -> std::vector<NumberType>
  // {}


  // template <int dim, typename VectorType>
  // auto MomentLimiter<dim, VectorType>::get_coeffs_finer_neighbor()
  //   -> std::vector<NumberType>
  // {}

  /* ************************************************** */
  template <int dim, typename VectorType>
  typename VectorType::value_type compute_cell_average(
    const dealii::FEValues<dim>& feval, const VectorType& soln_vector)
  {
    using NumberType = typename VectorType::value_type;
    NumberType soln_integral = 0.0;
    std::vector<NumberType> soln_qpt(feval.n_quadrature_points);
    feval.get_function_values(soln_vector, soln_qpt);

    for (unsigned int iq = 0; iq != feval.n_quadrature_points; ++iq)
      soln_integral += soln_qpt[iq] * feval.JxW(iq);

    return soln_integral / feval.get_cell()->measure();
  }


  /* ************************************************** */
  template <int dim, typename VectorType>
  void apply_weno_limiter(Mesh<dim>& mesh,
                          const dealii::DoFHandler<dim>& dofh,
                          const dealii::Mapping<dim>& mapping,
                          const dealii::Quadrature<dim>& quadrature,
                          VectorType& soln_vector)
  {
    std::vector<bool> saved_flags;
    mesh.save_user_flags(saved_flags);

    // label troubled cells
    MinmodShockDetector<dim, VectorType> shock_detector(mesh, soln_vector);
    const std::vector<bool>& shock_cell_flags =
      shock_detector.detect_shock_cells(dofh.active_cell_iterators(), mapping,
                                        dofh.get_fe(), quadrature);
    // WENOLimiter
    unsigned int max_deriv = std::min(2u, dofh.get_fe().degree);
    mesh.load_user_flags(shock_cell_flags);
    WENOLimiter<dim, VectorType> weno_limiter(soln_vector, max_deriv, 0.001);
    auto flagged_cell_range = filter_iterators(
      dofh.active_cell_iterators(), dealii::IteratorFilters::UserFlagSet());

    weno_limiter.apply_limiting(flagged_cell_range, mapping, dofh.get_fe(),
                                quadrature);

    // reload previous flags
    mesh.load_user_flags(saved_flags);
  }


  /* ************************************************** */
  template <int dim, typename VectorType>
  void apply_moment_limiter(const dealii::DoFHandler<dim>& dofh,
                            const dealii::Mapping<dim>& mapping,
                            const dealii::Quadrature<dim>& quadrature,
                            VectorType& soln_vector)
  {
    // this limiter will simply be applied to all cells
    MomentLimiter<dim, VectorType> moment_limiter(soln_vector,
                                                  dofh.get_fe().degree);
    moment_limiter.apply_limiting(dofh.active_cell_iterators(), mapping,
                                  dofh.get_fe(), quadrature);
  }

}  // namespace dg

FELSPA_NAMESPACE_CLOSE
#endif  //  _FELSPA_PDE_PDE_TOOLS_IMPLEMENT_H_ //
